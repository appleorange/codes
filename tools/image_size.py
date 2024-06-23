from PIL import Image
import os

# Define the directory containing the images
img_dir = '../cropped_/data/test/image'

# Initialize a dictionary to hold the count of each image size
size_count = {}

# Iterate over each file in the directory
for filename in os.listdir(img_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for common image file extensions
        img_path = os.path.join(img_dir, filename)
        with Image.open(img_path) as img:
            size = img.size  # Get image size as (width, height)
            if size in size_count:
                size_count[size] += 1  # Increment count if size already encountered
            else:
                size_count[size] = 1  # Initialize count for a new size

# Print the sizes and the count of images for each size
# sort the sizes by width and height
sorted_sizes = sorted(size_count.keys(), key=lambda x: (x[0], x[1])) 

for size in sorted_sizes:
    print(f"Size: {size}, Count: {size_count[size]}")

# get the median from the sorted sizes and 90th percentile of the sorted sizes
median_size = sorted_sizes[len(sorted_sizes) // 2]
percentile_90_size = sorted_sizes[int(len(sorted_sizes) * 0.9)]
print(f"Median Size: {median_size}")
print(f"90th Percentile Size: {percentile_90_size}")

