import random
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A

def visualise(image):
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()

def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(15,15))
    columns = 4
    rows = 4

    for i in range(1, len(images)):
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
