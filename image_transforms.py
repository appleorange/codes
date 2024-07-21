import cv2
import numpy as np
import random
import torch
from torchvision import transforms
from PIL import Image


class ContourThicknessVariation:
    def __init__(self, max_thickness=1):
        self.max_thickness = max_thickness

    def __call__(self, image):
        image_np = np.array(image)
        # Convert RGB to grayscale
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # Convert to binary image
        binary =  cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)#cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Randomly choose to dilate or erode
        if random.choice([True, False]):
            kernel = np.ones((self.max_thickness, self.max_thickness), np.uint8)
            image = cv2.dilate(binary, kernel, iterations=1)
        else:
            kernel = np.ones((self.max_thickness, self.max_thickness), np.uint8)
            image = cv2.erode(binary, kernel, iterations=1)
        
        return Image.fromarray(image)
        #return image

# # Example usage
# contour_transforms = transforms.Compose([
#     ContourThicknessVariation(max_thickness=2),
#     transforms.ToTensor()
# ])

class ContourDilationErosion:
    def __init__(self, max_kernel_size=1):
        self.max_kernel_size = max_kernel_size

    def __call__(self, image):
        
        image_np = np.array(image)
        # Convert RGB to grayscale
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Ensure image is in grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 17, 3)
        
        # Randomly choose to dilate or erode
        if random.choice([True, False]):
            kernel_size = random.randint(1, self.max_kernel_size)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.dilate(binary, kernel, iterations=1)
        else:
            kernel_size = random.randint(1, self.max_kernel_size)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.erode(binary, kernel, iterations=1)
        
        return Image.fromarray(image)
        #return image
# # Example usage
# contour_transforms = transforms.Compose([
#     ContourDilationErosion(max_kernel_size=3),
#     transforms.ToTensor()
# ])
