import argparse
import os

import numpy as np
import torch

from batcher import Batcher
from models import ArcBinaryClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for loading model'
                                                 'and saving images')
parser.add_argument('--load', required=True, help='the model to load from.')

opt = parser.parse_args()

if opt.name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                    "cuda" if opt.cuda else "cpu")

# initialise the batcher
batcher = Batcher(batch_size=opt.batchSize)

if __name__ == "__main__":
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    discriminator.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load)))
    # holds results of classification of test dataset
    results = []
    # get first test image classification, to get first index variable value
    X, true_label, size, index = batcher.fetch_test_batch()
    pred = discriminator(X).data.numpy()
    # get position of train image that is the most similar to 'index-th' test image. Divide by 100 because first 100
    # images are 0-th class, second 100 images are 1-th class and so on. Compared to true label (0 - 9) interval. To
    # use with unequal classes, use array with indexes where each class starts.
    if pred.argmax() // 100 == true_label:
        # correctly classified
        results.append(1)
    else:
        # incorrectly classified
        results.append(0)

    while index < size:
        # each 200 test images, print how many images was classified/tested so far and what is average precision.
        if index % 200 == 0:
            print('processing {} image from test dataset ... avg. success rate is {}'.format(index,
                                                                                             np.array(results).mean()))
        X, true_label, size, index = batcher.fetch_test_batch()
        pred = discriminator(X).data.numpy()
        if pred.argmax() // 100 == true_label:
            results.append(1)
        else:
            results.append(0)
    # convert list to array
    results = np.array(results)
    print(results.mean())
