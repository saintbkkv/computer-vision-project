# Import of libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Downloading dataset from TensorFlow 
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Classification labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalizing images to scale pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images to a vector
    tf.keras.layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 classes and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Model evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Predictions on test set
predictions = model.predict(test_images)

# Displaying a sample prediction
plt.imshow(test_images[0], cmap='gray')
plt.title(f"Predicted: {class_names[np.argmax(predictions[0])]}, Actual: {class_names[test_labels[0]]}")
plt.show()

# Accuracy and loss visualizations
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Saving the trained model
model.save('fashion_mnist_model.keras')  # Saving in Keras format

# Loading the model back (for demonstration)
from tensorflow.keras.models import load_model
model = load_model('fashion_mnist_model.keras')

# Predicting the class of a new test image
test_image = test_images[0]
test_image = np.expand_dims(test_image, axis=0)  # Reshaping to (1, 28, 28)

# Predict the class of the test image
prediction = model.predict(test_image)

# Displaying the predicted class
predicted_class = np.argmax(prediction)
print(f"Predicted class: {class_names[predicted_class]}")

# # Загрузим пользовательское изображение
# image_path = 'your_image.jpg'
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Сразу в оттенках серого
# img = cv2.resize(img, (28, 28))  # Изменим размер
# img = img / 255.0  # Нормализация
# img = np.expand_dims(img, axis=0)  # (1, 28, 28)

# # Предсказание
# prediction = model.predict(img)
# predicted_class = np.argmax(prediction)
# print(f"Предсказанный класс: {class_names[predicted_class]}")
