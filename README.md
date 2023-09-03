# my_project1234
# contains codes for my projects

# Train data set
# remember, for image-wise classification, the requirements are tensorflow, and pillow. Install both via pip
# If pip is not recognised by cmd, reinstall by checking all the ticks in advance options
# Also since pycharm is a virtual ide, go to file/settings/interpreter/pus sing/search for plugins
# This code is only for image-wise identification based on the model trained

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model # alwyas convert Model to model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# very important, for file destination, always make the single \ into double \\ to run
image = Image.open("C:\\Users\\Wati\\Downloads\\Dataset\\test\\Black Soil\\black_1.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

