import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load and preprocess data
def load_data(directory, target_size=(96, 96), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        subset='training')
    validation_generator = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        subset='validation')
    return train_generator, validation_generator

def representative_dataset_gen(train_gen, num_batches=100):
    for i, (x, _) in enumerate(train_gen):
        yield [x]
        if i >= num_batches - 1:
            break

# Export to TensorFlow Lite with quantization
def export_to_tflite(model, train_gen):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable post-training quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Define representative dataset for quantization
    converter.representative_dataset = lambda: representative_dataset_gen(train_gen)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert to TFLite format
    tflite_model = converter.convert()
    with open('model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)

# Update the main function to incorporate the changes
def main(data_dir):
    # load model from h5 file
    model = tf.keras.models.load_model('best_model.h5')
    train_gen, val_gen = load_data(data_dir)

    export_to_tflite(model, train_gen)
    print("Model trained and exported to TensorFlow Lite format with quantization.")

if __name__ == '__main__':
    data_dir = '/Users/andy/Desktop/datasets/cat_dataset_unsplit/'
    main(data_dir)