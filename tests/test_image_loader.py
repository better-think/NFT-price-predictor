from os.path import join
from unittest import TestCase

from image_loader import ImageLoader


class TestImageLoader(TestCase):
    __base_path = 'image-loader-data'

    def test_load_images(self):
        images = ImageLoader(join(self.__base_path, 'images')).load_images()
        [self.assert_image_has_3_channels(image_and_id.image) for image_and_id in images]

    def assert_image_has_3_channels(self, image):
        self.assertEquals(image.shape[2], 3)

    def test_find_png_files(self):
        base_path = join(self.__base_path, 'images')
        expected_files = [join(base_path, file_name) for file_name in [
            join('partition=0', '195000.png'),
            join('partition=0', '34359969000.png'),
            join('partition=1', '195001.png'),
            join('partition=1', '34359969001.png')
        ]]
        self.assertCountEqual(ImageLoader(base_path).find_png_files(), expected_files)
