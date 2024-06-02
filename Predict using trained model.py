import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

# load the model
cnn = tf.keras.models.load_model('my_model.keras')

# Prediction of a single image
path = './dataset/single_prediction/cat_or_dog_5.jpg'
test_image = image.load_img(path, target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255.0)
if result[0][0] > 0.5:
  prediction = 'dog'
else:
    prediction = 'cat'
    
# Printing the output prediction with bounding box
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(f"Prediction: {prediction}")
plt.show()