# EDXNet
EDXNet is a deep learning library independently developed by [Edward Liu](http://behindthepixels.io/). It is developed in C++ and only provides C++ interface. It's built with the static computational graph approach like TensorFlow, and implements automatic-differentiation.

A numpy like multi-dimensional tensor library is also included. Template expression is used to make tensor expression evaluation more efficient. 

Currently it only supports some of the most common operators like fully connected, convolution, pooling, batchnorm etc. With these simple operators EDXNet can already do many interesting tasks such as training a digit recognition neural net, image classification and even style transfer. The following is one sample result of running style transfer with EDXNet.

<div align="center">
 <img src="https://raw.githubusercontent.com/behindthepixels/EDXNet/master/StyleTransfer/coffee.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/behindthepixels/EDXNet/master/StyleTransfer/picasso_selfport1907.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/behindthepixels/EDXNet/master/StyleTransfer/Coffee_Picaaso.jpg height="223px">
</div>

More operators and features such as RNN, as well as GPU support is planned in the near future.
