# Image Preprocessing Pipeline
# Converts raw image → tensor ready for CNN
import cv2
import torch
import numpy as np
from torchvision import transforms


#Load Image
def load_image(image_path):
    #Reads an image from disk and converts BGR to RGB
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found at path:", image_path)
    #convert image into RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


#Resize Image-because most of CNN architecture buid in fixed size i.e 224
def resize_image(image, size=(224, 224)):
    #Resizes the image to fixed size required by CNN
    image = cv2.resize(image, size)
    return image


#Normalize Image
def normalize_image(image):
    #Normalize pixel values from 255 to 1
    image = image / 255.0
    return image


#Convert to Tensor
def to_tensor(image):
    #Converts numpy image to PyTorch tensor
    transform = transforms.ToTensor()

    tensor = transform(image)

    return tensor


#Complete Preprocessing Pipeline
def preprocess_image(image_path):
    # Load image
    image = load_image(image_path)
    # Resize image
    image = resize_image(image)
    # Normalize pixels
    image = normalize_image(image)
    # Convert to tensor
    tensor = to_tensor(image)
    return tensor

#Testing the Pipeline
if __name__ == "__main__":
    test_image_path = "data/images/test.jpg"
    tensor = preprocess_image(test_image_path)
    print("Preprocessing Successful!")
    print("Tensor Shape:", tensor.shape)