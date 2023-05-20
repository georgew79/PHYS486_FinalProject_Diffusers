'''
Helper functions for visualizing data...

@Author: George Witt
'''

import matplotlib.pyplot as plt
import torchvision
import torch

from PIL import Image

def viz_mnist(images, titles=None, title='h'):
    '''
    @images: np array of 28 x 28 images to plot, MAY NOT be None.
    @titles: np vectror of the number associated with each, MAY be None.
    
    It is recommended to keep the num of images small.
    '''

    # Code largely similar to assignment 4

    num_images = len(images)

    fig, axes = plt.subplots(1, num_images, figsize = (50, 8))

    for i in range(num_images):
        axes[i].imshow(images[i], cmap = 'gray')
        
        if titles is not None:
            axes[i].set_title(f'Image of {titles[i]}', fontsize = 40)
        else:
            axes[i].set_title(f'Provided Image {i+1}', fontsize = 40)

    fig.savefig(title)

def viz_mnist_wchannel(images, titles=None):
    '''
    @images: np array of 1 x 28 x 28 images to plot, MAY NOT be None.
    @titles: np vectror of the number associated with each, MAY be None.
    
    It is recommended to keep the num of images small.
    '''

    viz_mnist(images[:, 0], titles)

def save_images(images, path):
    images = torch.from_numpy(images)
    g = torchvision.utils.make_grid(images)

    # Permute the images into a 1d row of images, conv. to cpu and numpy again
    np_arr = g.permute(1, 2, 0).to('cpu').numpy()

    print(np_arr.shape)

    # Convert grid to image
    im = Image.fromarray(np_arr)
    im.save(path)
