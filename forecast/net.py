from lasagne import layers,nonlinearities
from lasagne.updates import momentum
from nolearn.lasagne import NeuralNet,PrintLayerInfo
import numpy as np
from nolearn.lasagne.base import BatchIterator
from matplotlib import pyplot
import math
import theano

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()





class network(object):
    def __init__(self,X_train, Y_train):
        #self.__hidden=0

        self.__hidden=int(math.ceil((2*(X_train.shape[1]+ 1))/3))
        self.net= NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer)
            ],
            input_shape=( None, X_train.shape[1] ),
            hidden_num_units=self.__hidden,
            #hidden_nonlinearity=nonlinearities.tanh,
            output_nonlinearity=None,
            batch_iterator_train=BatchIterator(batch_size=256),
            output_num_units=1,

            on_epoch_finished=[EarlyStopping(patience=50)],
            update=momentum,
            update_learning_rate=theano.shared(np.float32(0.03)),
            update_momentum=theano.shared(np.float32(0.8)),
            regression=True,
            max_epochs=1000,
            verbose=1,
        )

        self.net.fit(X_train,Y_train)

    def predict(self,X):
        return self.net.predict(X)

    def showMetrics(self):
        train_loss = np.array([i["train_loss"] for i in self.net.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in self.net.train_history_])
        pyplot.plot(train_loss, linewidth=3, label="training")
        pyplot.plot(valid_loss, linewidth=3, label="validation")
        pyplot.grid()
        pyplot.legend()
        pyplot.xlabel("epoch")
        pyplot.ylabel("loss")
        # pyplot.ylim(1e-3, 1e-2)
        pyplot.yscale("log")
        pyplot.show()

    def saveNet(self,fname):
        self.net.save_params_to(fname)

    def loadNet(self,fname):
        self.net.load_params_from(fname)


