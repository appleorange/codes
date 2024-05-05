import cv2
import json

# Load the image
img = cv2.imread('image.jpg')

# Load the JSON file
with open('labels.json') as f:
    data = json.load(f)

# Iterate over the annotations in the JSON file
for annotation in data['annotations']:
    # Check if the annotation type is 'points'
    if annotation['type'] == 'points':
        # Get the keypoint coordinates
        keypoints = annotation['points']
        
        # Draw the keypoints on the image
        for x, y in keypoints:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

# Display the image with keypoints
cv2.imshow('Keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()