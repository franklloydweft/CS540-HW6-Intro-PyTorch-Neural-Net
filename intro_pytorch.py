

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time

start_time = time.time()

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./ data', train=True, download=True, transform=custom_transform)
    test_set = datasets.FashionMNIST('./ data', train=False, transform=custom_transform)

    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)
        return loader
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size=64)
        return loader


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10)
                          )
    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()

    for epoch in range(T):
        running_loss = 0.0
        correct_class = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_class += (predicted == labels).sum().item()

        print(f'Train Epoch: {epoch} Accuracy: {correct_class}/{total}({100 * correct_class / total:.2f}%) '
              f'Loss: {(running_loss / total):.3f}')


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if show_loss:
        print(f'Average loss: {(running_loss / total):.4f}')
    print(f'Accuracy: {100 * correct / total:.2f}%')


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle Boot']
    image = test_images[index]
    logits = model(image)
    prob = F.softmax(logits, dim=1)
    values, indices = torch.topk(prob, 3)
    for i in range(3):
        print(f'{class_names[indices[0,i]]}: {values[0,i]*100:.2f}%')


def debug():
    # setup
    my_criterion = nn.CrossEntropyLoss()
    training = get_data_loader()
    testing = get_data_loader(False)
    my_model = build_model()

    # training and eval
    epochs = 5
    train_model(my_model, training, my_criterion, epochs)
    evaluate_model(my_model, testing, my_criterion, show_loss=True)

    # predict labels
    test_images, test_labels = next(iter(testing))
    predict_label(my_model, test_images, 1)


if __name__ == '__main__':
    debug()

print("--- %s seconds ---" % (time.time() - start_time))
