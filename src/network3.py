"""
Got the code from https://github.com/MichalDanielDobrzanski/DeepLearningPython/pull/14/
"""

"""network3.py
~~~~~~~~~~~~~~
A Pytorch-based program for training and running simple neural
networks.
Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).
When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.
Now that the code is based on Pytorch, it ressembles network.py and
network2.py more than when based on Theano.  However, where possible
I have tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return F.relu(z)
from torch.nn.functional import sigmoid, tanh



GPU = True
if GPU:
    print("Trying to run under a GPU. If this is not desired,\nthen modify network3.py to set the GPU flag to False.")
    if torch.cuda.is_available():
        print(f"CUDA device found: {torch.cuda.get_device_name(0)}\n")
        device = torch.device("cuda")
        torch.set_default_device("cuda")
    else:
        print("Pytorch did not find CUDA device. Using CPU.\n")
        device = torch.device("cpu")
        torch.set_default_device("cpu")
else:
    print("Running with a CPU. If this is not desired, then the modify network3.py to set\nthe GPU flag to True.")
    device = torch.device("cpu")
    torch.set_default_device("cpu")

def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    def shared(data):
        shared_x = torch.tensor(data[0], dtype=torch.float32)
        shared_y = torch.tensor(data[1], dtype=torch.float32)
        return shared_x, shared_y.type(torch.int32)

    return [list(shared(training_data)), list(shared(validation_data)), list(shared(test_data))]

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn

        # initialize weights and biases
        n_out = np.prod(filter_shape)/np.prod(poolsize)
        self.w = torch.normal(0., np.sqrt(1.0/n_out), filter_shape, requires_grad=True, dtype=torch.float32, device=device)
        self.b = torch.randn(filter_shape[0], requires_grad=True, dtype=torch.float32)
        self.params = [self.w, self.b]

    # set_inpt is just the feedforward function
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = F.conv2d(self.inpt, self.w)
        pooled_out = F.max_pool2d(conv_out, self.poolsize)
        self.output = self.activation_fn(pooled_out + (self.b.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)) # Pytorch's way of doing Theano's dimshuffle is nested unsqueezes
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        # Initialize weights and biases
        self.w = torch.normal(0., np.sqrt(1.0/n_out), (n_in,n_out), requires_grad=True, dtype=torch.float32, device=device)
        self.b = torch.randn(n_out, requires_grad=True, dtype=torch.float32)
        self.params = [self.w, self.b]

    # set_inpt is just the feedforward function
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1-self.p_dropout)*torch.matmul(self.inpt, self.w) + self.b)
        self.y_out = torch.argmax(self.output, dim=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(torch.matmul(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        return torch.mean((self.y_out == y).float())

class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout

        # Initialize weights and biases
        self.w = torch.zeros((n_in,n_out), requires_grad=True, dtype=torch.float32)
        self.b = torch.zeros(n_out, requires_grad=True, dtype=torch.float32)
        self.params = [self.w, self.b]

    # set_inpt is just the feedforward function
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = F.softmax((1-self.p_dropout)*torch.matmul(self.inpt, self.w) + self.b, dim=1)
        self.y_out = torch.argmax(self.output, dim=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = F.softmax(torch.matmul(self.inpt_dropout, self.w) + self.b, dim=1)

    def cost(self, net):
        "Return the log-likelihood cost."
        eps = 1e-8
        # To avoid log(0), force all activations to be between eps and 1.0
        probs = torch.clamp(self.output_dropout, eps, 1.0)
        return -torch.mean(torch.log(probs)[torch.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return torch.mean((self.y_out == y).float())

#### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers: FullyConnectedLayer | ConvPoolLayer | SoftmaxLayer, mini_batch_size: int):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]

    def feedforward(self, x, y):
        """Theano works with symbolic variables, while Pytorch works with explicit values. This is why I turned part of the __init__() method into a feedforward() method.
        Although the only input parameter manipulated here is ``x``, I still included ``y`` to stay consistent with the original Theano code. Otherwise the natural way to write this method would be ``feedforward(self, x)`` and instead move ``y`` to ``SoftmaxLayer.cost(self, net)``
        """
        self.x = x
        self.y = y
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)): # xrange() was renamed to range() in Python 3.
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def zero_grad(self):
        """An imitation of the built-in Pytorch function that resets the gradients. To be used between training epochs.
        Setting to None has the same effect as setting to torch.zeros() but is said to have slightly better performance. See the docs for torch.optim.Optimizer.zero_grad.
        """
        for p in self.params:
            p.grad = None

    # Theano symbolic functions had to be turned into real functions.
    # Also got moved out of Network.SGD().
    def mb_train(self, i, data_x, data_y, eta, lmbda, n):
        mb_range = range(i*self.mini_batch_size,(i+1)*self.mini_batch_size)
        self.feedforward(data_x[mb_range], data_y[mb_range])

        # define the (regularized) cost function and gradients
        l2_norm_squared = sum([torch.sum(layer.w**2) for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5*lmbda*l2_norm_squared/n
        cost.backward()

        with torch.no_grad():
            for layer in self.layers:
                layer.w -= (eta/self.mini_batch_size) * layer.w.grad
                layer.b -= (eta/self.mini_batch_size) * layer.b.grad

    def mb_accuracy(self, i, data_x, data_y):
        mb_range = range(i*self.mini_batch_size,(i+1)*self.mini_batch_size)
        self.feedforward(data_x[mb_range], data_y[mb_range])
        return self.layers[-1].accuracy(self.y)

    def mb_predictions(self, i, data_x, data_y):
        mb_range = range(i*self.mini_batch_size,(i+1)*self.mini_batch_size)
        self.feedforward(data_x[mb_range], data_y[mb_range])
        return self.layers[-1].y_out

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent.
        """
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            training_x, training_y = data_shuffle(training_data)
            for mini_batch in range(num_training_batches):
                iteration = num_training_batches*epoch + mini_batch
                if iteration % 1000 == 0:
                    print(f"Training mini-batch number {iteration}")

                self.mb_train(mini_batch, training_x, training_y, eta, lmbda, num_training_batches)

                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean([self.mb_accuracy(j, validation_x, validation_y).cpu() for j in range(num_validation_batches)])
                    print(f"Epoch {epoch}: validation accuracy {100*validation_accuracy:.2f}%")
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean([self.mb_accuracy(j, test_x, test_y).cpu() for j in range(num_test_batches)])
                            print(f"The corresponding test accuracy is {100*test_accuracy:.2f}%")
            self.zero_grad()
            print("")

        print("Finished training network.")
        print(f"Best validation accuracy of {100*best_validation_accuracy:.2f}% obtained at iteration {best_iteration}")
        print(f"Corresponding test accuracy of {100*test_accuracy:.2f}%")


# Miscellaneous
def size(data):
    return data[0].shape[0]

def dropout_layer(layer, p_dropout):
    """I'm lazy. I just used Pytorch's function which does the trick.
    """
    return F.dropout(layer, p_dropout)

def data_shuffle(data):
    """Randomly shuffles with a different seed on every run. Can be used on training_data, validation_data and test_data independently.
    """
    shuffled_indexes = torch.randperm(size(data))
    data[0][:] = data[0][shuffled_indexes]
    data[1][:] = data[1][shuffled_indexes]
    return data
