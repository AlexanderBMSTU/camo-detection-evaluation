import collections
import PIL.ImageDraw as ImageDraw
import scipy.misc	
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from _functools import partial, reduce
import numpy as np
import os
import six.moves.urllib as urllib
import sys

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from glob import glob

def generate_camo_in_boxes(boxes,scores,alfa_image_path,camo_image_path,min_score_thresh=.5):
    boxes_to_work = list()
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            boxes_to_work.append(box)

    alfa_image = Image.open(alfa_image_path).convert("RGBA")
    camo = Image.open(camo_image_path).convert("RGBA")
    background = Image.new("RGBA", alfa_image.size)
    for box in boxes_to_work:
        ymin, xmin, ymax, xmax = box
        im_width, im_height = background.size
        (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                  int(ymin * im_height), int(ymax * im_height))
        camo_layer_size = max(right-left,bottom-top)
        camo_layer = camo.resize([camo_layer_size,camo_layer_size], Image.BILINEAR)
        camo_layer = camo_layer.crop([0, 0, camo_layer_size/2,bottom-top])
        background.paste(camo_layer, (left, top), camo_layer)
    background.paste(alfa_image, (0, 0), alfa_image)
    #plt.figure(figsize=(12, 8))
    #plt.imshow(background)
    background = background.convert('RGB')
    #scipy.misc.imsave('test_images/1.jpg',background)
    return background
    
def create_camo_images(boxes,scores,alfa_image_path):
    dirName = 'test_images/camo/'
    camo_name_list = filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), os.listdir(dirName))
    for name in camo_name_list:
        generated_image = generate_camo_in_boxes(boxes,scores,alfa_image_path,dirName+name)
        scipy.misc.imsave(dirName+name[:len(name)-4]+'/3.jpg',generated_image)
        
def average(l):
    return reduce(lambda x, y: x + y, l) / len(l)

def plot_camo(labels, count):
    images = [mpimg.imread("test_images/camo/" + labels[i] + ".jpg") for i in range(count)]

    plt.figure(figsize=(30,20))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.xlabel(labels[i])
        plt.axis('off')
        plt.imshow(image)

def crop_img(image_paths_list):
	for image_path in image_paths_list:
		image = Image.open(image_path)	
		image = image.crop([100, 100, image.size[0]-100, image.size[1]-50])
		rgb_im = image.convert('RGB')
		rgb_im.save(image_path)
		cnt+=1
	print(str(cnt) + ' images processed')