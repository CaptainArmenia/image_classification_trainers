import cv2
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # Correctly pass the image as a named argument
            augmented = self.transform(image=image)  # Note the named argument here
            image = augmented['image']
        
        return image, label
    

# Dataloader for the dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

# Define the transforms
train_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.05),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.05),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Rotate(limit=30, p=0.5),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    ToTensorV2(),
])

# Create the datasets
train_dataset = ImageClassificationDataset(image_paths=train_image_paths, labels=train_labels, transform=train_transform)
val_dataset = ImageClassificationDataset(image_paths=val_image_paths, labels=val_labels, transform=val_transform)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Define the model
