![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)

# Model Zoo

A collection of standalone TensorFlow and PyTorch models in Jupyter Notebooks.

## Classifiers

- Perceptron [[TensorFlow](tensorflow_ipynb/perceptron.ipynb)] [[PyTorch](pytorch_ipynb/perceptron.ipynb)]
- Logistic Regression [[TensorFlow](tensorflow_ipynb/logistic-regression.ipynb)] [[PyTorch](pytorch_ipynb/logistic-regression.ipynb)]
- Softmax Regression (Multinomial Logistic Regression) [[TensorFlow](tensorflow_ipynb/softmax-regression.ipynb)] [[PyTorch](pytorch_ipynb/softmax-regression.ipynb)]
- Multilayer Perceptron [[TensorFlow](tensorflow_ipynb/multilayer-perceptron.ipynb)] [[PyTorch](pytorch_ipynb/multilayer-perceptron.ipynb)]
- Multilayer Perceptron with Dropout [[TensorFlow](tensorflow_ipynb/multilayer-perceptron-dropout.ipynb)] [[PyTorch](pytorch_ipynb/multilayer-perceptron-dropout.ipynb)]
- Multilayer Perceptron with Batch Normalization [[TensorFlow](tensorflow_ipynb/multilayer-perceptron-batchnorm.ipynb)] [[PyTorch](pytorch_ipynb/multilayer-perceptron-batchnorm.ipynb)]
- Multilayer Perceptron with Backpropagation from Scratch [[TensorFlow](tensorflow_ipynb/multilayer-perceptron-lowlevel.ipynb)] [[PyTorch](pytorch_ipynb/mlp-fromscratch__sigmoid-mse.ipynb)]

### Convolutional Classifiers

**Concepts**

- Replacing Fully-Connnected by Equivalent Convolutional Layers [[PyTorch](pytorch_ipynb/fc-to-conv.ipynb)]

**Basic**

- Convolutional Neural Network [[TensorFlow](tensorflow_ipynb/convnet.ipynb)] [[PyTorch](pytorch_ipynb/convnet.ipynb)]
- Convolutional Neural Network with He Initialization  [[PyTorch](pytorch_ipynb/convnet-he-init.ipynb)]

**All Convolutional**

- All-Convolutional Neural Network [[PyTorch](pytorch_ipynb/convnet-allconv.ipynb)]

**AlexNet**

- AlexNet on CIFAR-10 [[PyTorch](pytorch_ipynb/convnet-alexnet-cifar10.ipynb)]

**VGG**

- Convolutional Neural Network VGG-16 [[TensorFlow](tensorflow_ipynb/convnet-vgg16.ipynb)] [[PyTorch](pytorch_ipynb/convnet-vgg16.ipynb)]
- VGG-16 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/convnet-vgg16-celeba.ipynb)]
- Convolutional Neural Network VGG-19 [[PyTorch](pytorch_ipynb/convnet-vgg19.ipynb)]

**ResNet**

- Convolutional ResNet and Residual Blocks [[PyTorch](pytorch_ipynb/resnet-ex-1.ipynb)]
- ResNet-18 Digit Classifier Trained on MNIST [[PyTorch](pytorch_ipynb/convnet-resnet18-mnist.ipynb)]
- ResNet-18 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/convnet-resnet18-celeba-dataparallel.ipynb)]
- ResNet-34 Digit Classifier Trained on MNIST [[PyTorch](pytorch_ipynb/convnet-resnet34-mnist.ipynb)]
- ResNet-34 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/convnet-resnet34-celeba-dataparallel.ipynb)]
- ResNet-50 Digit Classifier Trained on MNIST [[PyTorch](pytorch_ipynb/convnet-resnet50-mnist.ipynb)]
- ResNet-50 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/convnet-resnet50-celeba-dataparallel.ipynb)]
- ResNet-101 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/convnet-resnet101-celeba.ipynb)]
- ResNet-152 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/convnet-resnet152-celeba.ipynb)]

**Network in Network**

- Network in Network CIFAR-10 Classifier [[PyTorch](pytorch_ipynb/nin-cifar10.ipynb)] 

## Metric Learning

- Siamese Network with Multilayer Perceptrons [[TensorFlow](tensorflow_ipynb/siamese-1.ipynb)]

## Autoencoders

**Regular Autoencoders**

- Autoencoder [[TensorFlow](tensorflow_ipynb/autoencoder.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder.ipynb)]
- Convolutional Autoencoder with Deconvolutions [[TensorFlow](tensorflow_ipynb/autoencoder-deconv.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder-deconv.ipynb)]
- Convolutional Autoencoder with Deconvolutions (without pooling operations) [[PyTorch](pytorch_ipynb/autoencoder-deconv-2.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation [[TensorFlow](tensorflow_ipynb/autoencoder-conv.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder-conv.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on CelebA [[PyTorch](pytorch_ipynb/autoencoder-conv-2.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on Quickdraw [[PyTorch](pytorch_ipynb/autoencoder-conv-quickdraw-1.ipynb)]

**Variational Autoencoders**

- Variational Autoencoder [[PyTorch](pytorch_ipynb/autoencoder-var.ipynb)]
- Convolutional Variational Autoencoder [[PyTorch](pytorch_ipynb/autoencoder-cnn-var.ipynb)]

**Conditional Variational Autoencoders**

- Conditional Variational Autoencoder (with labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder-cvae.ipynb)]
- Conditional Variational Autoencoder (without labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder-cvae_no-out-concat.ipynb)]
- Convolutional Conditional Variational Autoencoder (with labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder-cnn-cvae.ipynb)]
- Convolutional Conditional Variational Autoencoder (without labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder-cnn-cvae_no-out-concat.ipynb)]

## General Adversarial Networks

- General Adversarial Networks [[TensorFlow](tensorflow_ipynb/gan.ipynb)] [[PyTorch](pytorch_ipynb/gan.ipynb)]
- Convolutional General Adversarial Networks [[TensorFlow](tensorflow_ipynb/gan-conv.ipynb)]

## Recurrent Neural Networks (RNNs)


**Sentiment Analysis / Classification**

- A simple single-layer RNN (IMDB) [[PyTorch](pytorch_ipynb/rnn_simple_imdb.ipynb)]
- A simple single-layer RNN with packed sequences to ignore padding characters (IMDB) [[PyTorch](pytorch_ipynb/rnn_simple_packed_imdb.ipynb)]
- An RNN with LSTM cells (IMDB) [[PyTorch](pytorch_ipynb/rnn_lstm_packed_imdb.ipynb)]
- An RNN with LSTM cells and Own Dataset in CSV Format (IMDB) [[PyTorch](pytorch_ipynb/rnn_lstm_packed_own_csv_imdb.ipynb)]
- An RNN with GRU cells (IMDB) [[PyTorch](pytorch_ipynb/rnn_gru_packed_imdb.ipynb)]
- A multilayer bi-directional RNN (IMDB) [[PyTorch](pytorch_ipynb/rnn_gru_packed_imdb.ipynb)]

**Sequence-to-Sequence**

- A simple character RNN to generate new text (Charles Dickens) [[PyTorch](pytorch_ipynb/rnn_lstm_bi_imdb.ipynb)]


## Tips and Tricks

- Cyclical Learning Rate [[PyTorch](pytorch_ipynb/cyclical-learning-rate.ipynb)]

## PyTorch Workflows

**Datasets**

- [Using PyTorch Dataset Loading Utilities for Custom Datasets -- CSV files converted to HDF5](pytorch_ipynb/custom-data-loader-csv.ipynb)
- [Using PyTorch Dataset Loading Utilities for Custom Datasets -- Face Images from CelebA](pytorch_ipynb/custom-data-loader-celeba.ipynb)
- [Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from Quickdraw](pytorch_ipynb/custom-data-loader-quickdraw.ipynb)
- [Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from the Street View House Number (SVHN) Dataset](pytorch_ipynb/custom-data-loader-svhn.ipynb)
- [Pinned Memory](pytorch_ipynb/convnet-resnet34-cifar10-pinmem.ipynb)
- [Standardizing Images](pytorch_ipynb/convnet-standardized.ipynb)
- [Image Transformation Examples](pytorch_ipynb/torchvision-transform-examples.ipynb)
- [Char-RNN with Own Text File](pytorch_ipynb/rnn_lstm_bi_imdb.ipynb)
- [Sentiment Classification RNN with Own CSV File](pytorch_ipynb/rnn_lstm_packed_own_csv_imdb.ipynb)


**Parallel Computing**

- [Using Multiple GPUs with DataParallel -- VGG-16 Gender Classifier on CelebA](pytorch_ipynb/convnet-vgg16-celeba-data-parallel.ipynb)

**Other**

- [Sequential API and hooks](pytorch_ipynb/mlp-sequential.ipynb)
- [Getting Gradients of an Intermediate Variable in PyTorch](pytorch_ipynb/manual-gradients.ipynb)
- [Weight Sharing Within a Layer](pytorch_ipynb/convnet-weight-sharing.ipynb)
- [Plotting Live Training Performance in Jupyter Notebooks with just Matplotlib](pytorch_ipynb/plot-jupyter-matplotlib.ipynb)

## TensorFlow Workflows

- [Saving and Loading Trained Models -- from TensorFlow Checkpoint Files and NumPy NPZ Archives](tensorflow_ipynb/saving-and-reloading-models.ipynb)
- [Chunking an Image Dataset for Minibatch Training using NumPy NPZ Archives](tensorflow_ipynb/image-data-chunking-npz.ipynb)
- [Storing an Image Dataset for Minibatch Training using HDF5](tensorflow_ipynb/image-data-chunking-hdf5.ipynb)
- [Using Input Pipelines to Read Data from TFRecords Files](tensorflow_ipynb/tfrecords.ipynb)
- [Using Queue Runners to Feed Images Directly from Disk](tensorflow_ipynb/file-queues.ipynb)
- [Using TensorFlow's Dataset API](tensorflow_ipynb/dataset-api.ipynb)
