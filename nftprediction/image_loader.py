import glob
from dataclasses import dataclass
from os.path import join, sep

import numpy as np
from PIL import Image
from numpy import ndarray


class ImageLoader:
    __path: str

    def __init__(self, path: str):
        self.__path = path

    def load_images(self):
        return (self.__load_image(file_path) for file_path in self.find_png_files())

    def find_png_files(self):
        return glob.glob(join(self.__path, '**', '*.png'), recursive=True)

    def __load_image(self, file_path: str):
        image = Image.open(file_path)
        image = image.convert('RGB') if image.mode != 'RGB' else image
        return ImageAndId(np.asarray(image), int(file_path.split(sep)[-1].split('.')[0]))


@dataclass
class ImageAndId:
    image: ndarray
    id: int
