This is the first console version of the application developed using pure python scientific libraries like numpy and sklearn.
The motive was to implemnt a basic neural network without using any available libraries and to understand the working thoroughly.


trainData.py-->used for preprocessing of Data
net.py-->First version of the program
net2.py-->Revised version of net.py with focus on matrix operations
net3.py-->Final version of the console applicaiton which supports minibatch and regularization concepts.
          Minibatch highly aids in reducing computaional time by vectorising code which is handled by numpy.

This version implements the neural network in the popular Kera fraework which is implemented on top of Theano.

The basic objective of omplementation is to parallelize the neural networl trianing on the GPU.

trainingData.py --> File to preprocess the data using python scientific libraries like pandas and sklearn

net.py -->Actual implementation of the neural network using Keras

temp.csv-->Superstore sales Data

housing.csv-->housing dataset from https://archive.ics.uci.edu/ml/datasets/Housing
