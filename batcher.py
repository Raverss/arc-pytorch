"""
taken and modified from https://github.com/pranv/ARC
"""

import os

import numpy as np
import torch
from numpy.random import choice
from torch.autograd import Variable

from image_augmenter import ImageAugmenter

use_cuda = False


class Digits(object):
    def __init__(self, batch_size=128):
        """
        batch_size: the output is (2 * batch size, 1, image_size, image_size)
                    X[i] & X[i + batch_size] are the pair
        image_size: size of the image
        ---------------------
        Data Augmentation Parameters:
            flip: here flipping both the images in a pair
            scale: x would scale image by + or - x%
            rotation_deg
            shear_deg
            translation_px: in both x and y directions
        """
        # load training examples and labels
        tx = np.load(os.path.join('data', 'train_data.npy'))
        tx_starts = np.load(os.path.join('data', 'train_starts.npy'))
        tx_sizes = np.load(os.path.join('data', 'train_sizes.npy'))
        # load testing examples and labels
        tsx = np.load(os.path.join('data', 'test_data.npy'))
        tsx_starts = np.load(os.path.join('data', 'test_starts.npy'))
        tsx_sizes = np.load(os.path.join('data', 'test_sizes.npy'))

        # [H: 28, W: 26]
        image_size = tx.shape[1:3]
        # We'll be using training mean only
        self.mean_train = tx.mean() / 255.0  # used later for mean subtraction

        # remain after Omniglot, sampling probability for each alphabet given by number of its characters.
        def size2p(size):
            s = np.array(size).astype('float64')
            return s / s.sum()

        self.size2p = size2p

        self.data = {'train': tx, 'test': tsx}
        self.a_start = {'train': tx_starts, 'test': tsx_starts}
        self.a_size = {'train': tx_sizes, 'test': tsx_sizes}
        self.image_size = image_size
        self.batch_size = batch_size

        flip = True
        scale = 0.1
        rotation_deg = 10
        shear_deg = 5
        translation_px = 2
        self.augmentor = ImageAugmenter(image_size[1], image_size[0],
                                        hflip=flip, vflip=flip,
                                        scale_to_percent=1.0 + scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                                        translation_x_px=translation_px, translation_y_px=translation_px)

    def fetch_batch(self, phase):
        """
            This outputs batch_size number of pairs
            Thus the actual number of images outputted is 2 * batch_size
            Say A & B form the half of a pair
            The Batch is divided into 4 parts:
                Dissimilar A 		Dissimilar B
                Similar A 			Similar B

            Corresponding images in Similar A and Similar B form the similar pair
            similarly, Dissimilar A and Dissimilar B form the dissimilar pair

            When flattened, the batch has 4 parts with indices:
                Dissimilar A 		0 - batch_size / 2
                Similar A    		batch_size / 2  - batch_size
                Dissimilar B 		batch_size  - 3 * batch_size / 2
                Similar B 			3 * batch_size / 2 - 2 * batch_size

        """
        pass

    def fetch_test_batch(self):
        """
            Creates batch [number of training images, 2, image height, image width]. [:, 0, ...] contains all training
             images and [:, 1, ...] contains copy of i-th test image. Therefore we get similarity measure of i-th image
             with all training images. Format remains same as in fetch_batch(self, phase)
        """
        pass


class Batcher(Digits):
    def __init__(self, batch_size=128):
        Digits.__init__(self, batch_size)

        starts = self.a_start
        sizes = self.a_size

        size2p = self.size2p

        p = {'train': size2p(sizes['train']), 'test': size2p(sizes['test'])}

        self.starts = starts
        self.sizes = sizes
        self.p = p
        self.test_batch_index = 0

    def fetch_batch(self, phase, batch_size: int = None):

        if batch_size is None:
            batch_size = self.batch_size

        if phase == 'test':
            X, Y = self._fetch_test_batch(batch_size)
        else:
            X, Y = self._fetch_batch(phase, batch_size)

        X = Variable(torch.from_numpy(X)).view(2 * batch_size, self.image_size[0], self.image_size[1])

        X1 = X[:batch_size]  # (B, h, w)
        X2 = X[batch_size:]  # (B, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, h, w)

        Y = Variable(torch.from_numpy(Y))

        if use_cuda:
            X, Y = X.cuda(), Y.cuda()

        return X, Y

    def _fetch_batch(self, phase, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        data = self.data
        starts = self.starts[phase]
        sizes = self.sizes[phase]
        p = self.p[phase]
        image_size = self.image_size
        # number of classes
        num_digits = len(starts)

        # batch array
        X = np.zeros((2 * batch_size, image_size[0], image_size[1]), dtype='uint8')
        for i in range(batch_size // 2):
            # choose first digit class
            base_digit_class = choice(num_digits, p=p)
            dissimilar_digit_class = base_digit_class
            # choose dissimilar digit class different from base digit class
            while dissimilar_digit_class == base_digit_class:
                dissimilar_digit_class = choice(num_digits, p=p)
            # select random digit from from base_digit_class class
            base_digit = data[phase][starts[base_digit_class] + choice(sizes[base_digit_class], 1)]
            # select random digit from from base_digit_class class. With high probability different from base_digit
            similar_digit = data[phase][starts[base_digit_class] + choice(sizes[base_digit_class], 1)]
            # select random digit from from dissimilar_digit_class class
            dissimilar_digit = data[phase][starts[dissimilar_digit_class] + choice(sizes[dissimilar_digit_class], 1)]

            # save into batch array
            X[i], X[i + batch_size] = base_digit, dissimilar_digit
            X[i + batch_size // 2], X[i + 3 * batch_size // 2] = base_digit, similar_digit

        y = np.zeros((batch_size, 1), dtype='int32')
        # first half of batch are dissimilar digits
        y[:batch_size // 2] = 0
        # second half of batch are similar digits
        y[batch_size // 2:] = 1

        if phase == 'train':
            # augmentator rescale intensities from [0-255] to [0.0 - 1.0]!
            X = self.augmentor.augment_batch(X)
            X = X - self.mean_train
        else:
            X = X / 255.0
            X = X - self.mean_train
        # for stacking purposes
        X = X[:, np.newaxis]
        X = X.astype("float32")

        return X, y

    def fetch_test_batch(self):
        """
        Creates batch for testing batch_index-th test image.
        :return: batch array in format of [batch size, 2 (pair to be compared), image height, image width]
        """
        data = self.data
        # size of train dataset
        num_train = data['train'].shape[0]
        image_size = self.image_size
        # index of test image that is being classified in this batch
        batch_index = self.test_batch_index

        # create batch array
        X = np.zeros([2 * num_train, image_size[0], image_size[1]], dtype='uint8')
        # first half are all training images
        X[:num_train, ...] = data['train']
        # second half is copy of a batch_index-th test image to be classified
        X[num_train:, ...] = data['test'][batch_index, ...]
        # true label is extracted from array of indexes where particular class start
        test_label = np.argmax(self.starts['test']>batch_index) - 1

        # rescale intensities and center
        X = X / 255.0
        X = X - self.mean_train

        X = X[:, np.newaxis]
        X = X.astype("float32")

        self.test_batch_index += 1

        X = Variable(torch.from_numpy(X)).view(2 * num_train, self.image_size[0], self.image_size[1])

        # stack batch by second axis to [batch size, 2 (pair to be compared), image height, image width]
        X1 = X[:num_train]  # (B, h, w)
        X2 = X[num_train:]  # (B, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, h, w)

        if use_cuda:
            X = X.cuda()
        # using test dataset size and current index for controlling test loop in test_model.py
        return X, test_label, data['test'].shape[0], self.test_batch_index
