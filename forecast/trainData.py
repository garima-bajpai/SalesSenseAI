import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plot
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing


class trainingData(object):
    def __init__(self, data):
        self.__df = data
        self.__col_list = self.__df.columns.values.tolist()
        self.__del_col_list=[]
        self.__cat_col_list=[]
        self.__final_col_list = []


    def plot_corr(self):
        corMat = DataFrame(self.__df.corr())
        plot.pcolor(corMat)
        plot.show()

    def getColNmaes(self):
        return self.__col_list

    def preprocess(self,predictors, target, unique = 3):
        assert str(target)
        """
        :param target: variable to be predicted
        :param unique: tolerance for number of unique values in categorial columns. \
        If unique count in categorial columns is greater than this count those particular columns are dropped
        :return: none
        """
        for name in self.__col_list:
            if name == target:
                self.__temp = name

            if self.__df[name].dtype == 'O':
                if len(self.__df[name].unique()) > unique:
                    self.__del_col_list.append(self.__col_list.index(name))
                else:
                    self.__cat_col_list.append(name)

            if self.__df[name].dtype == 'int64':
                self.__df[name] = self.__df[name].astype(float)

        #templist = []
        for value in self.__col_list:
            if value not in predictors and self.__col_list.index(value) not in self.__del_col_list and value != target:
                self.__del_col_list.append(self.__col_list.index(value))


        #drop unwqanted columns
        self.__df.drop(self.__df.columns[self.__del_col_list],axis=1,inplace=True)


        #drop null values
        self.__df.dropna(axis=1,how='any',inplace = True)

        #prepare target df
        self.__target_df= self.__df[self.__temp]
        self.__df.drop(self.__temp,axis = 1, inplace = True)



        #train test split
        self.trainX ,self.testX, self.trainY, self.testY = sklearn.cross_validation.train_test_split(self.__df,self.__target_df,test_size=0.30)

        #get final column list for mappers
        self.__final_col_list = self.__df.columns.values.tolist()
        self.__num_col_list = [item for item in self.__final_col_list if item not in self.__cat_col_list]
        #print self.num_col_list

        self.mapfunc = []
        for name in self.__final_col_list:
            if self.__df[name].dtype == "O":
                self.mapfunc.append(([name],sklearn.preprocessing.LabelBinarizer()))
            else:
                self.mapfunc.append(([name], sklearn.preprocessing.StandardScaler(copy=False)))

        #io mappers
        self.in_mapper = DataFrameMapper(self.mapfunc)
        self.out_mapper = sklearn.preprocessing.RobustScaler(with_centering=False,copy=False)


        self.trainX = np.array(self.in_mapper.fit_transform(self.trainX),np.float32)
        self.trainY = np.array(self.out_mapper.fit_transform(self.trainY.reshape(-1,1)),np.float32)

        self.testX = np.array(self.in_mapper.transform(self.testX),np.float32)
        self.testY = np.array(self.out_mapper.transform(self.testY.reshape(-1,1)),np.float32)

        self.tindex = self.trainX.shape[0]


    def expt(self,name):
        """Export train or test Files...for debugging purposes """
        if name == "trainX":
            __df = pd.DataFrame(self.trainX)
            __df.to_csv("trainX.csv")
        elif name == "trainY":
            __df = pd.DataFrame(self.trainY)
            __df.to_csv("trainY.csv")
        elif name == "testX":
            __df = pd.DataFrame(self.testX)
            __df.to_csv("testX.csv")
        elif name == "testY":
            __df = pd.DataFrame(self.testX)
            __df.to_csv("testX.csv")
        else:
            raise ValueError


