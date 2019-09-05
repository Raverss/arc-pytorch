# arc-pytorch
PyTorch implementation of [Attentive Recurrent Comparators](https://arxiv.org/abs/1703.00767) by Shyam et al.

A [blog](https://medium.com/@sanyamagarwal/understanding-attentive-recurrent-comparators-ea1b741da5c3) explaining Attentive Recurrent Comparators

This is repository for testing ARC on custom (digits) dataset.

### How to run?

#### Download data
Download our [data](graphicwg.irafm.osu.cz/storage/ft1-pca-app.zip) and unzip two folders into data folder of this repository.

#### Train
```
python train.py --cuda
```
The training should achieve around 85%+ accuracy on test data.

#### Visualize
```
python viz.py --cuda --load name_of_model --same
```
Run with exactly the same parameters as train.py and specify the model to load. The script dumps images to a directory in visualization. The name of directory is taken from --name parameter if specified, else name is a function of the parameters of network. ```name_of_model``` should be in ```data``` folder.

#### Test accuracy of model
```
python test_model.py --cuda --load name_of_model
```
```name_of_model``` should be in ```data``` folder.
