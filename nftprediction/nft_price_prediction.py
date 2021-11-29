import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame

from alexnet import AlexNet
from image_loader import ImageLoader


def read_image_and_prices(images_path, prices_path) -> DataFrame:
    """
    :param images_path: Path of the images, you can read the whole or part by constraining with a partition
                        e.g. path-to-images/partition=0
    :return: a DataFrame of image_loader.ImageAndId
    """
    images_df = pd.DataFrame(ImageLoader(images_path).load_images()).set_index('id')
    images_df['image'] = images_df['image'].apply(normalize_images)
    prices_df = pd.read_pickle(prices_path, compression='gzip')[['id', 'Price_USD']].set_index('id')
    return pd.merge(images_df, prices_df, on='id')


def normalize_images(image: ndarray):
    return tf.image.resize(tf.image.per_image_standardization(image), (300, 300))  # TODO Make shape parametric


def main():
    image_and_prices_df = read_image_and_prices('../data/images',
                                                '../data/nft')
    images = image_and_prices_df['image'].to_numpy()
    prices = image_and_prices_df['Price_USD'].to_numpy()
    images = np.array(images.tolist())
    prices = prices.reshape((prices.shape[0], 1))

    images_length = images.shape[0]
    percent_60 = int(images_length * .6)
    percent_80 = int(images_length * .8)
    train_images, validation_images, test_images = images[:percent_60], images[percent_60:percent_80], images[
                                                                                                       percent_80:]
    train_prices, validation_prices, test_prices = prices[:percent_60], prices[percent_60:percent_80], prices[
                                                                                                       percent_80:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_prices)).batch(batch_size=2)
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_prices)).batch(batch_size=2)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_prices)).batch(batch_size=2)
    alexnet = AlexNet((300, 300, 3))
    alexnet.model.fit(train_ds, epochs=2, validation_data=validation_ds)
    print(alexnet.output_layer.get_weights())

    alexnet.model.evaluate(test_ds)


if __name__ == '__main__':
    main()
