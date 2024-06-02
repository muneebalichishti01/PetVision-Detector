# PetVision Detector:

# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Summary of the model
cnn.summary()

# Part 4 - Making a single prediction

# Prediction of a single image
path = 'dataset/single_prediction/cat_or_dog_6.jpg'
test_image = image.load_img(path, target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255.0)
training_set.class_indices
if result[0][0] > 0.5:
  prediction = 'dog'
else:
  prediction = 'cat'

# Printiing the output prediction with bounding box
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(f"Prediction: {prediction}")
plt.show()

# Saving the model
cnn.save('my_model.keras')

# Activation Maps and Feature Visualization
def get_activations(model, image, layer_name):
    model = Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
    return model.predict(image)

def plot_feature_maps(activations, num_cols=8):
    num_features = activations.shape[-1]
    num_rows = num_features // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*1.5, num_rows*1.5))
    for i in range(num_features):
        ax = axes[i//num_cols, i%num_cols]
        ax.imshow(activations[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage:
activations = get_activations(cnn, test_image/255.0, 'conv2d_2')  # Change 'conv2d' to your layer's name
plot_feature_maps(activations)

# load the model
cnn = tf.keras.models.load_model('my_model.keras')

# Prediction of a single image
path = 'dataset/single_prediction/cat_or_dog_6.jpg'
test_image = image.load_img(path, target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255.0)
training_set.class_indices
if result[0][0] > 0.5:
  prediction = 'dog'
else:
    prediction = 'cat'
    
# Printiing the output prediction with bounding box
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(f"Prediction: {prediction}")
plt.show()