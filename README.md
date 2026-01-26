## Fork overview

This is a fork of [unexploredtest's repo](https://github.com/unexploredtest/neural-networks-and-deep-learning) that uses Pytorch instead of Theano.
The main motivation was that Theano has been deprecated since Python 3.5.

Only `src/network3.py` has been updated.
`src/network.py` and `src/network2.py` are unchanged as they use Numpy to simulate the neural networks.

#### Details on the changes

wip

#### TODO

- [x] Fix `Network.feedforward()` not working when the network starts with a layer type different from `ConvPoolLayer`
- [x] (NOT RESETTING. INTENDED BEHAVIOR) Reset the weights to fix `Network.SGD` starting with already good weights after a rerun
- [ ] Fix vram accumulation when changing the layer structure of `Network` (only current solution is to restart the whole ipython kernel)
- [ ] Update remaining files to Pytorch.

## Acknowledgements

[unexploredtest](https://github.com/unexploredtest): for the Python3 compatibility. ([repo](https://github.com/unexploredtest/neural-networks-and-deep-learning))

[Michael Nielsen](https://github.com/mnielsen): for the original resource. ([repo](https://github.com/mnielsen/neural-networks-and-deep-learning))

## Original Readme: Code samples for "Neural Networks and Deep Learning"

This repository contains code samples for my book on ["Neural Networks
and Deep Learning"](http://neuralnetworksanddeeplearning.com).

The code is written for Python 2.6 or 2.7. There is a version for
Python 3.8-3.10 [here](https://github.com/unexploredtest/neural-networks-and-deep-learning).
I will not be updating the current repository for Python 3 compatibility.

The program `src/network3.py` uses version 0.6 or 0.7 of the Theano
library.  It needs modification for compatibility with later versions
of the library.  I will not be making such modifications.

As the code is written to accompany the book, I don't intend to add
new features. However, bug reports are welcome, and you should feel
free to fork and modify the code.
