from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.regularizers import *
from keras.callbacks import EarlyStopping
import numpy as np

class nnnet(object):


    def __init__(self,X_train,Y_train,X_test, Y_test,n_hidden,l1=0.,l2 = 0.):
        reg = l1l2(l1=l1, l2=l2)
        self.model = Sequential()
        self.X_test = X_train

        self.model.add(Dense(
            input_dim=X_train.shape[1],
            output_dim=n_hidden,
            W_regularizer=reg,
            init='uniform',
            activation="tanh"
        ))

        self.model.add(Dense(
            input_dim=n_hidden,
            output_dim=n_hidden,
            W_regularizer=reg,
            init="uniform",
            activation='tanh'
        ))

        self.model.add(Dense(
            input_dim=n_hidden,
            output_dim=Y_train.shape[1],
            W_regularizer=reg,
            init='uniform',
            activation='linear'
        ))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
        self.model.compile(sgd,'mse')

        self.model.fit(X_train,Y_train,256,100,callbacks=[],show_accuracy=True)
        y_pred = self.model.predict(X_train)

        #train_acc = np.sum((Y_train- y_pred), axis=0) / X_train.shape[0]
        #print('Training accuracy: %.2f%%' % (train_acc * 100))

    def Blah(self):
        #temp = self.model.predict_on_batch(self.X_test)
        #return temp
        y_pred = self.model.predict(self.X_test)
        return y_pred
























