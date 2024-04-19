import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def load_data(directory, target_size=(224, 224), batch_size=1):
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

def load_tflite_model(tflite_model_path):
    """Load a TFLite model from a file."""
    with open(tflite_model_path, 'rb') as file:
        tflite_model = file.read()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter

def visualize_predictions(interpreter, eval_generator):
    """Visualize predictions on a grid of images from the first batch."""
    #input_index = interpreter.get_input_details()[0]['index']
    #output_index = interpreter.get_output_details()[0]['index']
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scale, zero_point = input_details[0]['quantization']

    images, labels = next(eval_generator)
    predictions = []

    plt.figure(figsize=(10, 10))  # Adjust size as needed

    for i in range(images.shape[0]):
        quantized_input = ((images[i] / scale) + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], quantized_input[np.newaxis, ...])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = (prediction > 0).astype(np.int32)[0, 0]
        predictions.append(predicted_label)

        plt.subplot(4, 8, i+1)  # Assuming batch size is 32; adjust subplot grid as needed
        plt.imshow(images[i].reshape(224, 224, 3))
        plt.title(f'Pred: {predicted_label}\nTrue: {int(labels[i])}')
        plt.axis('off')

    plt.suptitle('Predictions on First Batch')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('predictions_grid.png')
    plt.show()

def evaluate_model(interpreter, eval_generator):
    """Evaluate the model on the entire evaluation dataset."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scale, zero_point = input_details[0]['quantization']
    num_batches = len(eval_generator)

    correct = 0
    for i, (image, labels) in enumerate(eval_generator):
        quantized_input = ((image / scale) + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], quantized_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_labels = (prediction > 0).astype(np.int32)[0, 0]
        correct += np.sum(predicted_labels == labels)

        if i % 10 == 0:
            print(f'Batch {i+1}/{num_batches}')

        if i == num_batches - 1:
            break

    accuracy = correct / (num_batches * eval_generator.batch_size)
    print(f'Accuracy: {accuracy:.2%}')

def main(tflite_model_path, data_directory):
    _, eval_generator = load_data(data_directory)
    interpreter = load_tflite_model(tflite_model_path)
    visualize_predictions(interpreter, eval_generator)
    evaluate_model(interpreter, eval_generator)

if __name__ == '__main__':
    tflite_model_path = 'model_quantized.tflite'
    data_directory = '/Users/andy/Desktop/datasets/cat_dataset_unsplit/'
    main(tflite_model_path, data_directory)