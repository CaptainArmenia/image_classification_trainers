import os
import argparse

from fastai.vision.all import *
from torchvision.models.quantization import mobilenet_v2
from fastai.callback.tracker import SaveModelCallback
from fastai.vision.augment import *
import torch
import torch.nn as nn
import torch.quantization


def create_grayscale_mobilenet_v2(n_out=2, pretrained=True, qat=True):
    # Load a pretrained MobileNetV2 model
    model = mobilenet_v2(pretrained=pretrained).features
    # Modify the first convolution layer
    model[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    # Add QAT if specified
    if qat:
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model = torch.quantization.prepare_qat(model)
    
    # Replace the classifier part of MobileNetV2
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(1280, n_out)
    )
    
    # Create a custom model combining the modified MobileNetV2 with the new classifier
    return nn.Sequential(model, classifier)

# Custom transforms compatible with MPS
class VerticalFlip(Transform):
    "A transform class that vertically flips images"
    def __init__(self, p=0.5): self.p = p
    def encodes(self, x: PILImage):
        if torch.rand(1) < self.p: 
            return x.transpose(Image.FLIP_TOP_BOTTOM)
        return x


class Rotate90(Transform):
    def encodes(self, x:PILImage):
        return x.rotate(-90, expand=True)
    

def rotate_image_90_deg(img:PILImage):
    """Rotate a PILImage by 90 degrees."""
    return img.rotate(90, expand=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/Users/andy/Desktop/datasets/cat_dataset_unsplit/')
    args = parser.parse_args()

    # Check if the dataset path exists
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist.")
    
    # Check if plots and checkpoints directories exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Set device
    device = torch.device('mps')

    # Augmentation transforms
    batch_tfms = [
        Rotate(max_deg=20, p=0.7), # Rotation
        #Flip(p=0.5), # Horizontal flipping
        VerticalFlip(p=0.5),
        #Dihedral(p=0.5), # Random flips in all eight possible directions
        Zoom(min_zoom=1.0, max_zoom=1.1, p=0.8), # Zoom
        #Warp(magnitude=0.2, p=0.8), # Warping
        Brightness(max_lighting=0.2, p=0.75), # Brightness
        Contrast(max_lighting=0.2, p=0.75), # Contrast
        Normalize.from_stats([0.5], [0.5]) # Normalizing to [-1,1]
    ]
    
    item_tfms=[Resize(224), Rotate90()]

    def label_func(f): return f.parent.name  # Define according to your data structure

    dblock = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                    get_items=get_image_files, 
                    splitter=RandomSplitter(valid_pct=0.2, seed=42),
                    get_y=label_func,
                    item_tfms=item_tfms,
                    batch_tfms=batch_tfms  # Normalizing to [-1,1]
                    )

    dls = dblock.dataloaders(args.dataset_path, bs=32, device=device)

    # dls = ImageDataLoaders.from_folder(args.dataset_path, valid_pct=0.2, seed=42,
    #                                 item_tfms=item_tfms,
    #                                 batch_tfms=batch_tfms,
    #                                 device=device)

    # Make plot directory
    os.makedirs('plots', exist_ok=True)

    # Save a plot of a train batch
    dls.show_batch(max_n=9, figsize=(7, 6))
    plt.savefig('plots/train_batch.png')

    # Save a plot of a validation batch
    dls.valid.show_batch(max_n=9, figsize=(7, 6))
    plt.savefig('plots/val_batch.png')

    # Create the modified model
    model = create_grayscale_mobilenet_v2()

    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

    # Print the device on which the model will be trained
    print(f"Training on {learn.dls.device}")

    learn.lr_find()

    # Plot the learning rate finder plot
    learn.recorder.plot_lr_find()
    plt.savefig('plots/lr_find.png')

    print("Training Batch Size:", dls.train.bs)
    print("Validation Batch Size:", dls.valid.bs)
    print("Loss Function:", learn.loss_func)
    print("Optimizer:", learn.opt_func)

    # Callback to save the model after each epoch
    cbs = [SaveModelCallback(monitor='accuracy', fname='/Users/andy/Desktop/repos/cat_classifier/checkpoints/cat_model_epoch', every_epoch=True)]

    learn.fit_one_cycle(100, 1e-3, cbs=cbs)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()

    # Save the model
    learn.export('cat_model.pkl')

