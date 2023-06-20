import cv2
from modules.SingleShotCamera import SingleShotCamera
import matplotlib.pyplot as plt

# Create a camera object
camera = SingleShotCamera()

# get image
img = camera.capture_image()

# Convert the image from BGR (OpenCV's default) to RGB (matplotlib's default)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.show()
