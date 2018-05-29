# Convolutional Neural Networks (CNNs / ConvNets)


### Reference:
- http://cs231n.github.io/convolutional-networks/
- Géron, Aurélien. Hands-on machine learning with Scikit-Learn and TensorFlow: concepts, tools, and techniques to build intelligent systems. " O'Reilly Media, Inc.", 2017.
- Goodfellow, Ian, et al. Deep learning. Cambridge: MIT press, 2016.
- Convolutional Neural Network (CNNs) by Andrew Ng  
https://www.youtube.com/watch?v=Z91YCMvxdo0&list=PLBAGcD3siRDjBU8sKRk0zX9pMz9qeVxud




<hr>

### Table of Contents:

1. Overview
2. Convolutional Layer
3. Pooling Layer
4. Normalization Layer
5. Fully-connected layer
6. TensorFlow Implementation


<hr>      

## 1. Overview  
### 1.1 Computer Vision Problem
Image Classification/ Recognization: takes an input image, and classify it under certain categories (Eg., Dog, Cat, Tiger, Lion).

Object Detection:

Neural Style Transfer:

convolution operation

Convolution is a specialized kind of linear operation. Convolutional networks are neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

### 1.2 Convolution Operation
The convolution operation is typically denoted with an asterisk `*`.


### 1.3 CNN Architectures
Typical CNN architectures stack a few convolutional layers (each one generally followed by a ReLU layer), then a pooling layer, then another few convolutional layers (+ReLU), then another pooling layer, and so on. The image gets smaller and smaller as it progresses through the network, but it also typically gets deeper and deeper (i.e., with more feature maps) thanks to the convolutional layers. At the top of the stack, a regular feedforward neural network is added, composed of a few fully connected layers (+ReLUs), and the final layer outputs the prediction (e.g., a softmax layer that outputs estimated class probabilities).

<p align='center'><img src='/images/typical CNN architecture.png' width="100%"></img></p><p align='center'>Typical CNN architecture</p>


## 2. Convolutional Layer
### 2.1 Architecture
Neurons in the first convolutional layer are not connected to every single pixel in the input image, but only to a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the **receptive field** of the neuron (equivalently this is the filter size). A **filter** that applies on the receptive field is also referred to as a **kernel** or a **neuron**. The output is called an **activation map** or **feature map**.

<p align='center'><img src='/images/convolution schematic.gif' width="60%"></img></p><p align='center'>The convolution operation. The output matrix is called Convolved Feature (or Feature Map, Activation Map). http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution. </p>

> It is important to emphasize the asymmetry in how we treat the spatial dimensions (width and height) and the depth dimension: The connections are local in space (along width and height), but always full along the entire depth of the input volume. In turn, each neuron in the second convolutional layer is connected only to neurons located within a small rectangle in the first layer. This architecture allows the network to concentrate on low-level features in the first hidden layer, then assemble them into higher-level features in the next hidden layer, and so on.


The convolutional layer’s parameters consist of a set of learnable filters. Every filter is small spatially (along width and height), but extends through the full depth of the input volume. For example, a typical filter on a first layer of a ConvNet might have size 5x5x3 (i.e. 5 pixels width and height, and depth 3, 3 color channels). As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. We will have an entire set of filters in each convolutional layer (e.g. 12 filters), and each of them will produce a separate 2-dimensional activation map. We will stack these 12 activation maps along the depth dimension and produce the output volume.

<p align='center'><img src='/images/convolutional layer.png' width="50%"></img></p><p align='center'>Example of a convolutional layer. Note, there are multiple neurons along the depth, all looking at the same region in the input. </p>

**Examples:**
1. Suppose that the input volume has size [32x32x3]. If the receptive field (or the filter size) is 5x5, then each neuron in the convolutional layer will have weights to a [5x5x3] region in the input volume, for a total of 5x5x3 = 75 weights (and +1 bias parameter). Notice that the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.
2. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the convolutional layer would now have a total of 3x3x20 = 180 connections to the input volume. Notice that, again, the connectivity is local in space (e.g. 3x3), but full along the input depth (20).

### 2.2 Spatial Arrangement
Three hyperparameters control the size of the output volume: the **depth**, **stride** and **padding**.

The **depth** corresponds to the number of filters we would like to use, each learning to look for something different in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color.

The distance between two consecutive receptive fields is called the **stride**. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the filters jump 2 pixels at a time as we slide them around. This will produce smaller output volumes spatially.

<p align='center'><img src='/images/one stride.png' width="60%"></img></p><p align='center'>Connections between layers: reducing dimensionality using a stride</p>

Sometimes it will be convenient to pad the input volume with zeros around the border. This is called **zero-padding**.
1. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes. For example, the size of the input volume keeps decreasing after applying convolutional layers. In order to preserve as much information about the original input and design deeper networks, we can use padding to preserve the size of the input volume so the input and output width and height are the same.
2. Additionally, padding improves performance by keeping information near the edge of the image (at the borders) since pixels on the corners are used much less in the output.

<p align='center'><img src='/images/zero padding.png' width="60%"></p><p align='center'>Connections between layers: zero-padding</p>

We can compute the spatial size of the output volume as a function of the input volume size (width: ***W***, height: ***H***, depth: ***D***), the receptive field size of the convolutional layer neurons / filter size (***F***), the stride with which they are applied (***S***), the amount of zero padding used (***P***) on the border, and the number of filters (***K***). The number of neurons is given by width = (***W-F+2P***)/***S+1***, height = (***H-F+2P***)/***S+1***, depth = ***K***.  

A common setting of the hyperparameters is ***F***=3, ***S***=1, ***P***=1.

**Examples:**
1. *Constraints on strides.* The spatial arrangement hyperparameters have mutual constraints. For example, when the input has size W=10, no zero-padding is used P=0, and the filter size is F=3, then it would be impossible to use stride S=2, since (W−F+2P)/S+1=(10−3+0)/2+1=4.5, i.e. not an integer, indicating that the neurons don’t “fit” neatly and symmetrically across the input.
2. *Real-world example.* The Krizhevsky et al. architecture that won the ImageNet challenge in 2012 accepted images of size [227x227x3]. On the first convolutional layer, it used neurons with receptive field size F=11, stride S=4 and no zero padding P=0. Since (227 - 11)/4 + 1 = 55, and since the convolutional layer had a depth of K=96, the output volume had size [55x55x96]. Each of the 55x55x96 neurons in this volume was connected to a region of size [11x11x3] in the input volume. Moreover, all 96 neurons in each depth column are connected to the same [11x11x3] region of the input, but of course with different weights.


### 2.3 Parameter Sharing
Using the real-world example above, we see that there are 55x55x96 = 290,400 neurons in the first convolutional layer, and each has 11x11x3 = 363 weights and 1 bias. Together, this adds up to 290400 x 364 = 105,705,600 parameters on the first layer of the ConvNet alone. Clearly, this number is very high. It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: if one feature is useful to compute at some spatial position, then it should also be useful to compute at a different position. In other words, denoting a single 2-dimensional slice of depth as a depth slice (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias. With this parameter sharing scheme, the first convolutional layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96x11x11x3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55x55 neurons in each depth slice will now be using the same parameters. (In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice.)

### 2.4 Examples
Suppose that the input volume `X` has shape `X.shape: (11,11,4)`. Suppose further that we use no zero padding (***P=0***), that the filter size is ***F=5***, and that the stride is ***S=2***. The output volume would therefore have spatial size (***11-5***)/***2+1 = 4***, giving a volume with width and height of 4. The activation map in the output volume (call it `V`) would then look as follows (only some of the elements are computed in this example):

`V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0`  
`V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0`  
`V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0`  
`V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0`

Remember that in numpy, the operation `*` above denotes elementwise multiplication between the arrays. The weight matrix `W0` is assumed to be of shape `W0.shape: (5,5,4)`, since the filter size is 5 and the depth of the input volume is 4. To construct a second activation map in the output volume, we would have:  

`V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1`  
`V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1`  
`V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1`  
`V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1`

`V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1`  (example of going along y)  
`V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1`  (or along both)  

When computing the second activation map, and that a different set of parameters (`W1`) is used. Additionally, recall that these activation maps are often followed elementwise through an activation function such as ReLU, but this is not shown here.

A more concrete example is shown below. The input volume (in blue), the weight volumes (in red), and the output volume (in green) are visualized with each depth slice stacked in rows. The input volume is of size ***W1=5***, ***H1=5***, ***D1=3***, and the convolutional layer parameters are ***K=2***, ***F=3***, ***S=2***, ***P=1***. That is, we have two filters of size 3×3, and they are applied with a stride of 2. Therefore, the output volume size has spatial size (5 - 3 + 2)/2 + 1 = 3. The visualization below iterates over the output activations (green), and shows that each element is computed by elementwise multiplying the highlighted input (blue) with the filter (red), summing it up, and then offsetting the result by the bias.

<p align='center'><img src='/images/conv example.png' width="80%"></p><p align='center'>Convolutional Layer</p>

### 2.5 Implementation
#### Matrix Multiplication
The convolution operation essentially performs dot products between the filters and local regions of the input. The local regions in the input image are stretched out into columns in an operation commonly called **im2col**.

For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 filters at stride 4, then we would take [11x11x3] blocks of pixels in the input and stretch each block into a column vector of size 11x11x3 = 363. Iterating this process in the input at stride of 4 gives (227-11)/4+1 = 55 locations along both width and height, leading to an output matrix `X_col` of **im2col** of size [363 x 3025], where every column is a stretched out receptive field and there are 55x55 = 3025 of them in total. Note that since the receptive fields overlap, every number in the input volume may be duplicated in multiple distinct columns.

The weights of the convolutional layer are similarly stretched out into rows. For example, if there are 96 filters of size [11x11x3] this would give a matrix `W_row` of size [96 x 363].

The result of a convolution is now equivalent to performing one large matrix multiplication `np.dot(W_row, X_col)`, which evaluates the dot product between every filter and every receptive field location. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each filter at each location. The result must finally be reshaped back to its proper output dimension [55x55x96].

This approach has the downside that it can use a lot of memory, since some values in the input volume are replicated multiple times in `X_col`.

#### Backpropagation
...

## 3. Pooling Layer
It is common to periodically insert a **pooling layer** in-between successive convolutional layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also limit the risk of overfitting. Reducing the input image size also makes the neural network tolerate a little bit of image shift (location invariance).

Just like in convolutional layers, each neuron in a pooling layer is connected to the outputs of a limited number of neurons in the previous layer, located within a small rectangular receptive field. You must define its size, the stride, and the padding type, just like before. However, a pooling neuron has no weights; all it does is aggregate the inputs using an **aggregation function** such as the max or mean. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation, which has been shown to work better in practice.

The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. Note that it is not common to use zero-padding for Pooling layers.

<p align='center'><img src='/images/pooling layer.png' width="90%"></p><p align='center'>Pooling layer downsamples the volume spatially, independently in each depth slice of the input volume.</p>


More generally, the pooling layer accepts a value of size ***W*** x ***H*** x ***D***. It requires two hyperparameters: the filter ***F*** and the stride ***S***. It produces a volume of size ***W'*** x ***H'*** x ***D'***, where:  ***W'*** = ***(W - F)***/***S + 1***, ***H'*** = ***(H - F)***/***S + 1***, ***D'*** = ***D***.

> Many people dislike the pooling operation and think that we can get away without it. To reduce the size of the representation they suggest using larger stride in convolutional layer once in a while. Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers.

## 4. Normalization Layer



## 5. Fully-connected layer


## 6. TensorFlow Implementation

### 6.1 Convolutional Layer
In TensorFlow, each **input image** is typically represented as a 3D tensor of shape `[height, width, channels]`. A **mini-batch** is represented as a 4D tensor of shape `[mini-batch size, height, width, channels]`. The **weights (filters)** of a convolutional layer are represented as a 4D tensor of shape `[filter height, filter weight, number of feature maps (number of filters), number of feature maps in the previous layer (number of filters in the previous layer)]`. The **bias** term of a convolutional layer is simply represented as a 1D tensor of shape `[number of feature maps]`.

Let’s look at a simple example. The following code loads two sample images, using Scikit-Learn’s load_sample_images(). Then it creates two 7 × 7 filters (one with a vertical white line in the middle, and the other with a horizontal white line), and applies them to both images using a convolutional layer built using TensorFlow’s `conv2d()` function (with zero padding and a stride of 2).

```python
import numpy as np
from sklearn.datasets import load_sample_images

# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create 2 filters
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32) filters_test[:, 3, :, 0] = 1   # vertical line
filters_test[3, :, :, 1] = 1   # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
plt.show()
```


- Strides is a four-element 1D array, where the two central elements are the vertical and horizontal strides. The first and last elements must currently be equal to 1. They may one day be used to specify a batch stride (to skip some instances) and a channel stride (to skip some of the previous layer’s feature maps or channels).


- Padding must be either "VALID" or "SAME":
   - "Valid" convolution: no padding.
   - "Same" convolution: pad so that output size is the same as the input size.

<p align='center'><img src='/images/padding options.png' width="60%"></p><p align='center'>Padding options—input width =13, filter width =6, stride =5</p>


### 6.2 Pooling Layer

Here we implement a max pooling layer in TensorFlow. The following code creates a max pooling layer using a 2 × 2 kernel (filter), stride 2, and no padding, then applies it to all the images in the dataset:

```python
[...] # load the image dataset, just like above

# Create a graph with input X plus a max pooling layer
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(output[0].astype(np.uint8)) # plot the output for the 1st image
plt.show()
```

The `ksize` argument contains the kernel shape along all four dimensions of the input tensor: `[batch size, height, width, channels]`. TensorFlow currently does not support pooling over multiple instances, so the first element of ksize must be equal to 1. Moreover, it does not support pooling over both the spatial dimensions (height and width) and the depth dimension, so either ksize[1] and ksize[2] must both be equal to 1, or ksize[3] must be equal to 1.

To create an average pooling layer, just use the `avg_pool()` function instead of `max_pool()`.
