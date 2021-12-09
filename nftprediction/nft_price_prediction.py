import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame

from alexnet import AlexNet
from image_loader import ImageLoader


def run(images_path, prices_path):
    image_and_prices_df = read_image_and_prices(images_path, prices_path)
    images = np.array(image_and_prices_df['image'].to_numpy().tolist())
    prices = image_and_prices_df['Price_USD'].to_numpy()
    reshaped_prices = prices.reshape((prices.shape[0], 1))

    train_images, validation_images, test_images = split_data_as_train_validation_test(.6, .2, images)
    train_prices, validation_prices, test_prices = split_data_as_train_validation_test(.6, .2, reshaped_prices)

    train_ds = create_tf_dataset(train_images, train_prices)
    validation_ds = create_tf_dataset(validation_images, validation_prices)
    test_ds = create_tf_dataset(test_images, test_prices)

    alexnet = AlexNet((300, 300, 3))
    alexnet.model.fit(train_ds, epochs=2, validation_data=validation_ds)
    alexnet.model.evaluate(test_ds)
    print(alexnet.output_layer.get_weights())


def split_data_as_train_validation_test(train_ratio, validation_ratio, data):
    data_length = data.shape[0]
    train_part = int(data_length * .6)
    validation_part = int(data_length * (train_ratio + validation_ratio))
    return data[:train_part], data[train_part:validation_part], data[validation_part:]


def create_tf_dataset(data, labels):
    return tf.data.Dataset.from_tensor_slices((data, labels)).batch(batch_size=2)  # TODO Parameterize batch_size


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


if __name__ == '__main__':
    run('path-to-images', 'path-to-nft')
