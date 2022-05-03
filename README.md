# Top 50 On MNIST CNN 

A short demo that shows how to create, train, test, save and reload a CNN model. 

## MNIST Benchmark
<p align="center">
  <img src="https://github.com/grensen/top_50_mnist/blob/main/figures/top_50_mnist_benchmark.png?raw=true">
</p>

This CNN ranks under the top 50 in the [MNIST benchmark](https://paperswithcode.com/sota/image-classification-on-mnist). For that, my basic NN is used, extended and turned into a [CNN](https://github.com/grensen/convolutional_neural_network). The result was quite good with 99,20% accuracy in the MNIST test. This CNN model here is quite similar and based on the previous demo. But with an additional technique, [Infinity Dropout](https://github.com/grensen/easy_regression#infinity-regression), and a few more epochs of training.

## The Demo
<p align="center">
  <img src="https://github.com/grensen/top_50_mnist/blob/main/figures/top_50_mnist_demo.png?raw=true">
</p>

With 99.42% accuracy in the MNIST test, it was even much better, nice.

To run the demo program, you must have [VisualStudio2022](https://visualstudio.microsoft.com/downloads/) installed on your machine. Then just start a console application with .NET 6, copy the code and change from debug to release mode and run the demo. MNIST data and network are then managed by the code. 







