import os
import numpy as np
import glob
import cv2
from collections import defaultdict


# using opencv2 for loading images in grayscale. Can be substituted to any other loading method that keeps format
# [image height, image width, channels]
def imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def save_imgs_to_ndy(data_dir_path, dataset_part, binary_file_dir_path):
    """
    saves images into numpy binary file. For consistency with original repository purposes. Keeping Omniglot formatting,
    as implemented in original repository - batcher.py. Our classes are load one after another, from class digit 0 to
    class digit 9.
    :param data_dir_path: path to folder with images to be converted
    :param dataset_part: test or train, to keep them separated
    :param binary_file_dir_path: path to dir
    """
    # get all images in directory
    data_paths = glob.glob(os.path.join(data_dir_path, '*.png'))
    # get data shape from first image [img_count, img_h, img_w, channels]
    data_shape = (len(data_paths),) + imread(data_paths[0]).shape
    data_array = np.zeros(data_shape, dtype='uint8')

    for datum_index in range(len(data_paths)):
        # read all images and save them in data array
        data_array[datum_index] = imread(data_paths[datum_index])
    # save data array to binary numpy file
    np.save(os.path.join(binary_file_dir_path, dataset_part + '_data.npy'), data_array, allow_pickle=False)

    labels = defaultdict(int)
    for datum_index in range(len(data_paths)):
        # split path to image by path separator and then split image name by '-'. First substring contains label.
        label = data_paths[datum_index].split(os.path.sep)[-1].split('-')[0]
        # increase given label counter
        labels[label] += 1

    label_counts_array = np.zeros([len(labels.keys())], dtype='int32')
    # write dictionary to label_counts_array
    for key in labels.keys():
        label_counts_array[int(key)] = labels[key]

    # save label counts to binary numpy file
    np.save(os.path.join(binary_file_dir_path, dataset_part + '_sizes.npy'), label_counts_array, allow_pickle=False)
    # compute start of each class within array
    class_starts = np.cumsum(label_counts_array)
    class_starts = np.roll(class_starts, 1)
    class_starts[0] = 0
    # and save it to the binary numpy file
    np.save(os.path.join(binary_file_dir_path, dataset_part + '_starts.npy'), label_counts_array, allow_pickle=False)


if __name__ == '__main__':
    # process training images# compute start of each class within array
    train_dataset_path = os.path.join('data', 'letters_learn_es_pp')
    save_imgs_to_ndy(train_dataset_path, 'train', 'data')
    # process training images# compute start of each class within array
    test_dataset_path = os.path.join('data', 'letters_recogn_es_pp')
    save_imgs_to_ndy(test_dataset_path, 'test', 'data')
