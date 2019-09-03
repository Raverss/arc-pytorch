import os
import numpy as np
import glob
import cv2
from collections import defaultdict


def imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def save_imgs_to_ndy(path, file_name, file_path):
    tx_file_paths = glob.glob(path)
    # get data shape from first image [img_count, img_w, img_h, channels]
    data_shape = (len(tx_file_paths),) + imread(tx_file_paths[0]).shape
    tx = np.zeros(data_shape)

    labels = defaultdict(int)
    for tx_file_path_index in range(len(tx_file_paths)):
        label = tx_file_paths[tx_file_path_index].split(os.path.sep)[-1].split('-')[0]
        labels[label] += 1

    for tx_file_path_index in range(len(tx_file_paths)):
        tx[tx_file_path_index] = imread(tx_file_paths[tx_file_path_index])

    ty = np.zeros([len(labels.keys())])
    for key in labels.keys():
        ty[int(key)] = labels[key]

    np.save(os.path.join(file_path, file_name + '_data.npy'), tx, allow_pickle=False)
    np.save(os.path.join(file_path, file_name + '_counts.npy'), ty, allow_pickle=False)


if __name__ == '__main__':
    path = os.path.join('data', 'letters_learn_es_pp', '*.png')
    save_imgs_to_ndy(path, 'train.npy', 'data')
    path = os.path.join('data', 'letters_recogn_es_pp', '*.png')
    save_imgs_to_ndy(path, 'test.npy', 'data')
