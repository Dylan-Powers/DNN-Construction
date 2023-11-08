import torch
import numpy as np # For loading cached datasets
# import matplotlib.pyplot as plt
# import torchvision # For loading initial datasets
#                    # Commented out because Spring 2023 this is failing to load
#                    # in the conda-cs3450 environment
import time
import warnings
import os.path

from network import *
from layers import *

# warnings.filterwarnings('ignore')  # If you see warnings that you know you can ignore, it can be useful to enable this.

EPOCHS = 5
# For simple regression problem
TRAINING_POINTS = 1000

# For fashion-MNIST and similar problems
DATA_ROOT = '/data/cs3450/data/'
FASHION_MNIST_TRAINING = '/data/cs3450/data/fashion_mnist_flattened_training.npz'
FASHION_MNIST_TESTING = '/data/cs3450/data/fashion_mnist_flattened_testing.npz'
CIFAR10_TRAINING = '/data/cs3450/data/cifar10_flattened_training.npz'
CIFAR10_TESTING = '/data/cs3450/data/cifar10_flattened_testing.npz'
CIFAR100_TRAINING = '/data/cs3450/data/cifar100_flattened_training.npz'
CIFAR100_TESTING = '/data/cs3450/data/cifar100_flattened_testing.npz'

# With this block, we don't need to set device=DEVICE for every tensor.
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
     torch.cuda.set_device(0)
     torch.set_default_tensor_type(torch.cuda.FloatTensor)
     print("Running on the GPU")
else:
     print("Running on the CPU")

def create_linear_training_data():
    """
    This method simply rotates points in a 2D space.
    Be sure to use L2 regression in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_folded_training_data():
    """
    This method introduces a single non-linear fold into the sort of data created by create_linear_training_data. Be sure to REMOVE the final softmax layer before testing on this data!
    Be sure to use MSE in the place of the final softmax layer before testing on this
    data!
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    x = torch.randn((2, TRAINING_POINTS))
    x1 = x[0:1, :].clone()
    x2 = x[1:2, :]
    x2 *= 2 * ((x2 > 0).float() - 0.5)
    y = torch.cat((-x2, x1), axis=0)
    return x, y


def create_square():
    """
    This is a square example in which the challenge is to determine
    if the points are inside or outside of a point in 2d space.
    insideness is true if the points are inside the square.
    :return: (points, insideness) the dataset. points is a 2xN array of points and insideness is true if the point is inside the square.
    """
    win_x = [2,2,3,3]
    win_y = [1,2,2,1]
    win = torch.tensor([win_x,win_y],dtype=torch.float32)
    win_rot = torch.cat((win[:,1:],win[:,0:1]),axis=1)
    t = win_rot - win # edges tangent along side of poly
    rotation = torch.tensor([[0, 1],[-1,0]],dtype=torch.float32)
    normal = rotation @ t # normal vectors to each side of poly
        # torch.matmul(rotation,t) # Same thing

    points = torch.rand((2,2000),dtype = torch.float32)
    points = 4*points

    vectors = points[:,np.newaxis,:] - win[:,:,np.newaxis] # reshape to fill origin
    insideness = (normal[:,:,np.newaxis] * vectors).sum(axis=0)
    insideness = insideness.T
    insideness = insideness > 0
    insideness = insideness.all(axis=1)
    return points, insideness


def load_dataset_flattened(train=True,dataset='Fashion-MNIST',download=False):
    """
    :param train: True for training, False for testing
    :param dataset: 'Fashion-MNIST', 'CIFAR-10', or 'CIFAR-100'
    :param download: True to download. Keep to false afterwords to avoid unneeded downloads.
    :return: (x,y) the dataset. x is a torch tensor where columns are training samples and
             y is a torch tensor where columns are one-hot labels for the training sample.
    """
    if dataset == 'Fashion-MNIST':
        if train:
            path = FASHION_MNIST_TRAINING
        else:
            path = FASHION_MNIST_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-10':
        if train:
            path = CIFAR10_TRAINING
        else:
            path = CIFAR10_TESTING
        num_labels = 10
    elif dataset == 'CIFAR-100':
        if train:
            path = CIFAR100_TRAINING
        else:
            path = CIFAR100_TESTING
        num_labels = 100
    else:
        raise ValueError('Unknown dataset: '+str(dataset))

    if os.path.isfile(path):
        print('Loading cached flattened data for',dataset,'training' if train else 'testing')
        data = np.load(path)
        x = torch.tensor(data['x'],dtype=torch.float32)
        y = torch.tensor(data['y'],dtype=torch.float32)
        pass
    else:
        class ToTorch(object):
            """Like ToTensor, only redefined by us for 'historical reasons'"""

            def __call__(self, pic):
                return torchvision.transforms.functional.to_tensor(pic)

        if dataset == 'Fashion-MNIST':
            data = torchvision.datasets.FashionMNIST(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-10':
            data = torchvision.datasets.CIFAR10(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        elif dataset == 'CIFAR-100':
            data = torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=ToTorch(), download=download)
        else:
            raise ValueError('This code should be unreachable because of a previous check.')
        x = torch.zeros((len(data[0][0].flatten()), len(data)),dtype=torch.float32)
        for index, image in enumerate(data):
            x[:, index] = data[index][0].flatten()
        labels = torch.tensor([sample[1] for sample in data])
        y = torch.zeros((num_labels, len(labels)), dtype=torch.float32)
        y[labels, torch.arange(len(labels))] = 1
        np.savez(path, x=x.numpy(), y=y.numpy())
    return x, y

class Timer(object):
    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.2f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename,'a') as file:
                print(str(datetime.datetime.now())+": ",message,file=file)


# Training loop -- fashion-MNIST
# def main_linear():
# if __name__ == '__main__':
    # Once you start this code, comment out the method name and uncomment the
    # "if __name__ == '__main__' line above to make this a main block.
    # The code in this section should NOT be in a helper method.
    #
    # In particular, your client code that uses your classes to stitch together a specific network 
    # _should_ be here, and not in a helper method.  This will give you access
    # to the layers of your network for debugging purposes.

    # with Timer('Total time'):
    #     # TODO: You may wish to make each TODO below its own pynb cell.
    #     # TODO: Build your network.
    #     dataset = 'Fashion-MNIST'
    
    #     # TODO: Select your datasource.
    #     x_train, y_train = create_linear_training_data()
    #     # x_train, y_train = create_folded_training_data()
    #     # x_train, y_train = load_dataset_flattened(train=True, dataset=dataset, download=True)
    
    #     # TODO: Train your network.
    #     with Timer('Training time'):
    #         pass # Replace with your code to train
    
    #     # TODO: Sanity-check the output of your network.
    #     # Compute the error on this test data:
    #     x_test, y_test = create_linear_training_data()
    #     # x_test, y_test = create_folded_training_data()
    #     # x_test, y_test = load_dataset_flattened(train=False, dataset=dataset)
    
    #     # Report on GPU memory used for this script:
    #     peak_bytes_allocated = torch.cuda.memory_stats()['active_bytes.all.peak']
    #     print(f"Peak GPU memory allocated: {peak_bytes_allocated} Bytes")

dataset = 'Fashion-MNIST'

# x_train, y_train = create_linear_training_data()
# x_train, y_train = create_folded_training_data()
x_train, y_train = load_dataset_flattened(train=True, dataset=dataset, download=True)

# %%
lambda_ = 0.000
epsilon = 1e-8
step_size = 0.001
batch_size = 1
hidden_nodes = [80]
num_train_samples = x_train.shape[1]
num_features = x_train.shape[0]
num_output_classes = y_train.shape[0]

network = Network()

## Input Layer
# The second dimension of the input does not matter right now
# since we can work off different batch sizes later
input = Input(size=(num_features, 1))
network.set_input(input)

## Hidden Layer 0
W_0 = Input(trainable=True, size=(hidden_nodes[0], num_features))
network.add(W_0)
b_0 = Input(trainable=True, size=(hidden_nodes[0], 1))
network.add(b_0)
linear_0 = Linear(W_0, b_0, input)
network.add(linear_0)

## ReLU
relu_0 = ReLU(linear_0)
network.add(relu_0)

## Output Layer
W_1 = Input(trainable=True, size=(num_output_classes, hidden_nodes[0]))
network.add(W_1)
b_1 = Input(trainable=True, size=(num_output_classes, 1))
network.add(b_1)
linear_1 = Linear(W_1, b_1, relu_0)
network.add(linear_1)

## Regularization
# r_0 = Regularization(W_0)
# network.add(r_0)
# r_1 = Regularization(W_1)
# network.add(r_1)

## Softmax and Cross Entropy Loss
# The second dimension does not matter right now
# since we can work off different batch sizes later
true_label_layer = Input(size=(num_output_classes, 1))
network.set_true_label(true_label_layer)
# mse_loss = MSELoss(linear_1, true_label_layer)
softmax = Softmax(linear_1, y=true_label_layer, epsilon=epsilon)
network.set_output(softmax)

# Sum the regularization terms and the cross entropy loss
# reg_sum = Sum(softmax, lambda_=lambda_)
# r_0, r_1, 
# network.add(reg_sum)

# reg_sum should be the loss layer when the regularization terms are added
network.set_loss(softmax)

# %%
epochs_to_train = EPOCHS
samples_per_accuracy_check = 100
train_accuracy_list = []
train_loss_list = []
test_accuracy_list = []
test_loss_list = []

with Timer('Training time'):
    print("Hypterparameters:")
    print("lambda: ", lambda_)
    print("step size: ", step_size)
    print("epsilon: ", epsilon)
    print("batch_size: ", batch_size)
    print("hidden_nodes: ", hidden_nodes)
    # print("number of learnable parameters: ", (hidden_nodes * num_features) + (num_output_classes * hidden_nodes) + hidden_nodes + num_output_classes)
    
    for j in range(epochs_to_train):
        samples_since_last_accuracy_check = 0
        num_correct_preds = 0
        loss = 0
        for i in range(batch_size, num_train_samples + 1, batch_size):
            # create the input vector from x
            sample = x_train[:, i-batch_size:i].clone()
            sample = sample.view(num_features, batch_size)

            # create the true label vector from y
            true_label = y_train[:, i-batch_size:i].clone()
            true_label = true_label.view(num_output_classes, batch_size)

            classifications, loss = network.forward(sample, true_label)
            network.backward()
            network.step(step_size=step_size)
            #print(f"Loss 1 for batch {i / batch_size} of {num_train_samples / batch_size}: {loss}")
            # print(W_0.output_grad)
            # print(W_1.output_grad)
            # print(W_1.output)
            # print(W_0.output)
            #classifications, loss = network.forward(sample, true_label)
            #print(f"Loss 2 for batch {i / batch_size} of {num_train_samples / batch_size}: {loss}")

            network.clear_grad()

            # Accuracy Check for classification problems (comment out for regression problems)
            pred = torch.argmax(classifications, 0)
            true = torch.argmax(y_train[:, i-batch_size:i].view(num_output_classes, batch_size), 0)
            num_correct_preds += torch.sum(pred == true).item()

            # if (samples_since_last_accuracy_check >= samples_per_accuracy_check):
                
            # samples_since_last_accuracy_check += batch_size

        train_accuracy = num_correct_preds / num_train_samples
        print("training accuracy in epoch", str(j), ":",  train_accuracy)
        print("training loss in epoch", str(j), ":",  loss)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(loss)

        x_test, y_test = load_dataset_flattened(train=False, dataset=dataset)

        test_correct_preds = 0
        num_test_samples = x_test.shape[1]
        test_loss = 0

        for i in range(batch_size, num_test_samples + 1, batch_size):
            # create the input vector from x
            sample = x_test[:, i-batch_size:i].clone()
            sample = sample.view(num_features, batch_size)

            # create the true label vector from y
            true_label = y_test[:, i-batch_size:i].clone()
            true_label = true_label.view(num_output_classes, batch_size)

            classifications, test_loss = network.forward(sample, true_label)

            pred = torch.argmax(classifications, 0)
            true = torch.argmax(y_test[:, i-batch_size:i].view(num_output_classes, batch_size), 0)
            test_correct_preds += torch.sum(pred == true).item()

        print("test accuracy in epoch", j, ":",  test_correct_preds / num_test_samples)
        print("test loss in epoch", j, ":",  test_loss)
        test_accuracy_list.append(test_correct_preds / num_test_samples)
        test_loss_list.append(test_loss)

print("training accuracy list:", train_accuracy_list)
print("training loss list:", train_loss_list)
print("test accuracy list:", test_accuracy_list)
print("test loss list:", test_loss_list)


# %%
# Compute the error on this test data:
# x_test, y_test = create_linear_training_data()
# x_test, y_test = create_folded_training_data()


# %%
# Report on GPU memory used for this script:
# peak_bytes_allocated = torch.cuda.memory_stats()['active_bytes.all.peak']
# print(f"Peak GPU memory allocated: {peak_bytes_allocated} Bytes")


