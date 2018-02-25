---
layout: default
---

![](assets/pictures/coffee_picaso_transfer.jpg)

**EDXNet** is a deep learning library independently developed by [Edward Liu](http://behindthepixels.io/). It is developed in C++ and only provides C++ interface. It's built with the static computational graph approach like TensorFlow, and implements automatic-differentiation.

A numpy like multi-dimensional tensor library is also included. Template expression is used to make tensor expression evaluation more efficient. 

Currently it only supports some of the most common operators like fully connected, convolution, pooling, batchnorm etc. With these simple operators EDXNet can already do many interesting tasks such as training a digit recognition neural net, image classification and even style transfer.

The source code of EDXNet is highly self-contained and does not depend on any external library other than [EDXUtil](https://github.com/behindthepixels/EDXUtil), which is a utility library developed by Edward Liu.

### Features
- Statically built net works for image recognition and style transfer
- Pre-trained weights for LeNet and VGG19 included

### Operators
- Convolution
- Fully connected
- Min/Max/Avg pooling
- Relu
- Batch Normalization
- Dropout

### Optimizers
- SGD
- Adam
- LBFGS

### More Examples

![](assets/pictures/coffee_wave_transfer.jpg)