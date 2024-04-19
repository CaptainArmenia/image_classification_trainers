import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
#import tensorflow_model_optimization as tfmot


# Load and preprocess data
def load_data(directory, target_size=(224, 224), batch_size=64):
    datagen = ImageDataGenerator(preprocessing_function=lambda x: (x / 127.5) - 1.0, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        #color_mode='grayscale',
        class_mode='binary',
        subset='training')
    validation_generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        #color_mode='grayscale',
        class_mode='binary',
        subset='validation')
    return train_generator, validation_generator

def build_grayscale_mobilenet(input_shape=(96, 96, 1), num_classes=2):
    """
    Build a MobileNetV2 model adapted for 96x96 grayscale images.

    Args:
    input_shape (tuple): Shape of the input images.
    num_classes (int): Number of output classes, set to 1 for binary classification.

    Returns:
    TensorFlow Keras Model: A model configured for grayscale inputs.
    """
    # Input tensor for grayscale images
    input_tensor = Input(shape=input_shape)

    # Expand the grayscale image to mimic 3 channels needed by MobileNetV2
    x = Conv2D(3, (1, 1), padding='same', activation='relu', name='color_space_adjust')(input_tensor)

    # Load the MobileNetV2 model, excluding the top fully connected layer
    #base_model = MobileNetV2(include_top=False, weights=None, input_tensor=x, input_shape=(240, 240, 3), alpha=0.35)
    base_model = MobileNet(include_top=False, input_shape=(96, 96, 3), weights='imagenet', alpha=0.25)

    # Freeze the base model to prevent changes to its learned weights during training
    base_model.trainable = False

    # Add pooling and output layers
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)

    # Compile the model
    model = Model(inputs=input_tensor, outputs=output)

    return model

def build_rgb_mobilenet(input_shape=(224, 224, 3), num_classes=1):
    """
    Build a MobileNetV2 model adapted for 512x512 RGB images.

    Args:
    input_shape (tuple): Shape of the input images.
    num_classes (int): Number of output classes, set to 1 for binary classification.

    Returns:
    TensorFlow Keras Model: A model configured for RGB inputs.
    """
    # Load the MobileNetV2 model, excluding the top fully connected layer
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape, alpha=0.35)

    # Freeze the base model to prevent changes to its learned weights during training
    base_model.trainable = False

    # Add pooling and output layers
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)

    # Compile the model
    model = Model(inputs=base_model.input, outputs=output)

    return model


# Train the model and save best model
def train_model(model, train_gen, val_gen, epochs=50, num_classes=1):
    # Callback to save the best model
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    # Define the pruning configuration
    # pruning_params = {
    #     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
    #                                                             final_sparsity=0.5,
    #                                                             begin_step=0,
    #                                                             end_step=1000)
    # }

    # Apply pruning to the entire model
    #pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Train the pruned model
    model.compile(optimizer=Adam(), loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy', metrics=['accuracy'])
    
    # Callback to update the pruning steps during training
    callbacks = [
        #tfmot.sparsity.keras.UpdatePruningStep(),
        #tfmot.sparsity.keras.PruningSummaries(log_dir='/tmp/logs'),  # Log sparsity and other metrics in TensorBoard.
        checkpoint
    ]

    return model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)



# Main function to execute the pipeline
def main(data_dir):
    train_gen, val_gen = load_data(data_dir)
    model = build_rgb_mobilenet()
    model_history = train_model(model, train_gen, val_gen)
    print("Model trained and saved.")
    #model = tfmot.sparsity.keras.strip_pruning(model)
    #model.save('pruned_model.h5')

# Provide the path to your dataset directory
if __name__ == "__main__":
    main('/Users/andy/Desktop/datasets/cat_dataset_unsplit/')