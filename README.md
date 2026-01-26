## Fork overview

This is a fork of [unexploredtest's repo](https://github.com/unexploredtest/neural-networks-and-deep-learning) that uses Pytorch instead of Theano.
The main motivation was that Theano has been deprecated since Python 3.5.

Only `src/network3.py` has been updated.
`src/network.py` and `src/network2.py` are unchanged as they use Numpy to simulate the neural networks.

> [!NOTE]
> **You can still follow Nielsen's book as if nothing changed.**
>
> This was the most important requirement. You can copy-paste the console commands given in the book and they will run.
>
> Only when you'll be asked to modify `network3.py` yourself in chapter 6 you'll have to understand how it works.
> Hopefully I didn't make it too difficult.

> [!NOTE]
> **Hardware compatibility :**
>
> So far only tested on Python 3.10.19 + torch 2.5.1+cu121 on an RTX 4050.
> Will test more recent versions of Python in the future.
> Feel free to send a PR for README.md if you succesfully tested on a different configuration.
>
> Running on a non-Nvidia GPU will trigger CPU mode.

### Details on the changes

#### Requirements

- Numpy version is no longer capped to 1.22
- Theano removed
- Torch added: 2.2.0 minimum for Numpy 2.0 compatibility

#### Theano to Pytorch

Theano uses symbolic variables while Pytorch uses explicit values. This is what caused most of the changes.

Most notable differences:

- In `Network` class, the original `__init()__` function as been split into `__init()__` and `feedforward()`.
- Modified the `theano.function([i], cost, updates=updates, givens={...})` as methods of the `Network` class.

### TODO

- [x] Fix `Network.feedforward()` not working when the network starts with a layer type different to `ConvPoolLayer`
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
