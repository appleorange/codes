import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_transforms import ContourDilationErosion
from image_transforms import ContourThicknessVariation
from torchvision.transforms import v2
from PIL import Image


# Load a sample contour image
image_path = 'data65k/test/image/000000043.jpg'

greyscale = v2.Grayscale(num_output_channels=3)

original_image = Image.open(image_path).convert('RGB')
augmented_image = greyscale(original_image)

distort = v2.RandomPhotometricDistort()
augmented_image_distort = distort(original_image)

sharpness = v2.RandomInvert()
augmented_image_sharpness = sharpness(original_image)


# original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Create an instance of ContourDilationErosion
# augmentation = ContourDilationErosion(max_kernel_size=1)

# #augmentation = ContourThicknessVariation()

# # Apply the augmentation
# augmented_image = augmentation(original_image)

# Visualize the results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5))

ax1.imshow(original_image, cmap='gray')
ax1.set_title('Original Contour Image')
ax1.axis('off')

ax2.imshow(augmented_image, cmap='gray')
ax2.set_title('Augmented Contour Image')
ax2.axis('off')

ax3.imshow(augmented_image_distort, cmap='gray')
ax3.set_title('Augmented distort Image')
ax3.axis('off')

ax4.imshow(augmented_image_sharpness, cmap='gray')
ax4.set_title('Augmented sharpness Image')
ax4.axis('off')

plt.tight_layout()
plt.show()