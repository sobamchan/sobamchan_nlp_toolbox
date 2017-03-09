from PIL import Image
from PIL import ImageOps
import numpy as np
import os

class Img(object):

    def __init__(self):
        pass

    @staticmethod
    def gray(origin_path, output_path):
        origin_image = Image.open(origin_path)
        output_image = ImageOps.grayscale(origin_image)
        output_image.save(output_path)

    @staticmethod
    def resize(origin_path, output_path, size):
        origin_image = Image.open(origin_path, 'r')
        resized_image = origin_image.resize(size)
        resized_image.save(output_path)

    @staticmethod
    def gray_dir(origin_dir, output_dir):
        files = os.listdir(origin_dir)
        for f in files:
            if not f.startswith('.'):
                Img.gray('{}/{}'.format(origin_dir, f), '{}/{}'.format(output_dir, f))

    @staticmethod
    def resize_dir(origin_dir, output_dir, size):
        files = os.listdir(origin_dir)
        for f in files:
            if not f.startswith('.'):
                Img.resize('{}/{}'.format(origin_dir, f), '{}/{}'.format(output_dir, f), size=size)

    @staticmethod
    def np_array_dir(input_dir):
        image_arrays = []
        files = os.listdir(input_dir)
        for f in files:
            if not f.startswith('.'):
                img = Image.open('{}/{}'.format(input_dir, f))
                img_arr = np.asarray(img)
                image_arrays.append(img_arr)

        return image_arrays
