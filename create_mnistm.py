"""
create_mnistm.py: Creates the MNIST-M Dataset

Adapted from https://github.com/pumpikano/tf-dann/blob/master/create_mnistm.py
"""

import tarfile
import os
import numpy as np
import skimage
import skimage.io
import urllib.request
import tensorflow as tf
import h5py

URL = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
BST_PATH = './Datasets/BSR_bsds500.tgz'
MNIST_M_PATH = './Datasets/MNIST_M/mnistm.h5'

(mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = tf.keras.datasets.mnist.load_data()

rand = np.random.RandomState(42)

background_data = []



def check_file_or_download():
    if not os.path.isfile(BST_PATH):
        print(f"Download BSDS500 Dataset from {URL}\n [y/(n)]?")
        response = input().lower()
        
        if response == 'y':
            os.makedirs(os.path.dirname(BST_PATH), exist_ok=True)
            urllib.request.urlretrieve(URL, BST_PATH)
        else:
            print(f"Download the BSDS500 Dataset and place at {BST_PATH}")
        
        if os.path.isfile(BST_PATH):
            print(f"BSR Dataset Downloaded at {BST_PATH}")
        else:
            raise FileNotFoundError(f"{BST_PATH} does not exist!")


def process_bsr():
    check_file_or_download()
    
    with tarfile.open(BST_PATH) as f:
        train_files = []
        
        for name in f.getnames():
            if name.startswith('BSR/BSDS500/data/images/train/'):
                train_files.append(name)

        print('Loading BSR training images')

        for name in train_files:
            try:
                fp = f.extractfile(name)
                bg_img = skimage.io.imread(fp)
                background_data.append(bg_img)
            except:
                continue


def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    
    bg = background[x:x+dw, y:y+dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
    
        if i % 1000 == 0:
            print(f'Processing {i}/{X.shape[0]}...', end='\r')

        bg_img = rand.choice(background_data)

        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d

    return X_



def convert_dict_to_h5dataset(d: dict, f):
    """
    Convert Nested Dictionaries containing Numpy Arrays to HDF5 Dataset
    """
    for item in d.items():
        if isinstance(item[1], dict):
            convert_dict_to_h5dataset(item[1], f.create_group(item[0]))
        else:
            assert(isinstance(item[1], np.ndarray))
            f.create_dataset(item[0], data=item[1])


def write_h5Dataset(dataset_dict, filename):
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, 'w') as f:
        convert_dict_to_h5dataset(dataset_dict, f)
        
        print(f'\n{filename} written')


if __name__ == '__main__':
    process_bsr()
    
    print('Building Training Set...')
    train_x = create_mnistm(mnist_train_x)
    
    print('\nBuilding Testing Set...')
    test_x = create_mnistm(mnist_test_x)
    
    dataset = {
        'train': {
            'X': train_x,
            'Y': mnist_train_y
        },
        'test' : {
            'X' : test_x,
            'Y' : mnist_test_y
        }
    }
    
    write_h5Dataset(dataset, MNIST_M_PATH)
