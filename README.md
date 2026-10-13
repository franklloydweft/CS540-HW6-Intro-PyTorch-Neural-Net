# CS 540 HW6 Spring 2023

## Important note
The training data for this neural net is not included in this repository. In the original assignment, I had it stored in a "data" directory. For your own reference, this project uses the [Fashion-MNIST}](https://github.com/zalandoresearch/fashion-mnist) dataset.
I would also recommend setting up your environment in a Python virtual environment, as that is what the assignment specification recommends.

## Required Packages
- Most all of these imports rely on having PyTorch installed and configured. That was the purpose of this assignment.
- `import torch`
- `import torch.nn as nn`
- `import torch.nn.functional as F`
- `import torch.optim as optim`
- `from torchvision import datasets, transforms`
- `import time` for tracking training epochs

## Implemented functions
This file implements 6 functions, all dividing up the process of building, training, and evaluating a model.
- `get_data_loader(training=True)` Returns the dataloader for training set (boolean true) and test set (boolean false).
- `build_model` Returns an untrained neural net model with four layers: a flatten layer, dense 128 node ReLU layer, dense 64 node ReLU layer, and a final 10 node dense layer.
- `train_model(model,train_loader,criterion,T)` Parameters specify the model to be trained, the training loader, the training criteria (cross entropy), and number of epochs.
- `evaluate_model(model,test_loader,criterion,show_loss=True)` Parameters specify the trained model, the test loader, the evaluation criteria (cross entropy), and boolean to show or hide loss, default show.
- `predict_label(model,test_images,index)` Function predicts labels on given test images with a softmax function on the final layer of the trained model. Parameters specify the model, tensor of test images, and index of a particular image to test.
- `debug()` "Main" function from which all other functions run to set up model. This debug function is called directly in the actual main function. All setup functions are called here and specific parameters are assigned.

*one of my predicted labels in this script ended up being wrong, as a note*
