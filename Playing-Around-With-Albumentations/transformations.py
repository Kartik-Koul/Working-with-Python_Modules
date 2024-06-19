# Albumentations is a particularly useful library during the tasks of data segmentation and data augmentation.
# It can be used with deep learning libraries such as PyTorch and Tensorflow.

# Why use it with PyTorch?
# It is faster than Torchvision with its functioning on every benchmark. Torchvision is a PyTorch library for image and video processing.
# It has support with tasks like segmentation and detection.

import cv2 # OpenCV for all your image processing needs
import albumentations as A # The big A. It's how albumentations is often referred in these programs.
import numpy as np 
from PIL import Image # Python Imaging Library. Also for your image processing needs.
from utils import plot_examples

# Let us work with one of the greatest images to exist -- The Pedro meme face.

image = Image.open("images/pedro.jpg")

transform = A.Compose(
    [
        # Here we can specify the transformations we want to perform on the image in a sequential manner
        A.Resize(height=1080, width=1920),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.8, border_mode=cv2.BORDER_CONSTANT), # Setting the rotation limit of the image to be 40 degrees. The p argument specifies that the probability of this transformation. So rotation will occur in 80% of the cases. The use of the border_mode argument is best understood by studying rotated outputs with and without it. I will include both the outputs as images.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.RGBShift(r_shift_limit = 20, g_shift_limit= 20, b_shift_limit= 20, p=0.5), # Specifying a possible colour shift transformation with the change limit being 20 in each of the channels
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5)
        ], p = 1.0), # Essentially in all the cases, one of the image blur or colour jitter augmentations will be selected. They will then be applied based on their respective arguments specified here.
    ]
)

images_list = [image]

image = np.array(image)

for i in range(15):
    augmentations = transform(image = image)
    augmented_image = augmentations["image"]
    images_list.append(augmented_image)

plot_examples(images_list)
