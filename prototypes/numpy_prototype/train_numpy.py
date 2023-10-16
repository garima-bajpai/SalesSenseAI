import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper

import sklearn.preprocessing

class trainingData(object):
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.col_list = self.df.columns.values.tolist()
        self.del_col_list=[]
        self.cat_col_list=[]
        self.final_col_list = []

    def getColNmaes(self):
        return self.col_list

    def preprocess(self,target, unique = 3):
        for name in self.col_list:
            if name == target:
                temp = name
            if self.df[name].dtype == 'O':
                if len(self.df[name].unique()) > unique:
                    self.del_col_list.append(self.col_list.index(name))
                else:
                    self.cat_col_list.append(name)

            if self.df[name].dtype == 'int64':
                self.df[name] = self.df[name].astype(float)
        print self.cat_col_list

        #drop unwqanted columns
        self.df.drop(self.df.columns[self.del_col_list],axis=1,inplace=True)

        #drop null values
        self.df.dropna(axis=1,how='any',inplace = True)

        #prepare target df
        self.target_df= self.df[temp]
        self.df.drop(temp,axis = 1, inplace = True)


        #train test split
        self.trainX ,self.testX, self.trainY, self.testY = sklearn.cross_validation.train_test_split(self.df,self.target_df,test_size=0.30)

        #get final column list for mappers
        self.final_col_list = self.df.columns.values.tolist()
        self.num_col_list = [item for item in self.final_col_list if item not in self.cat_col_list]
        print self.num_col_list

        self.mapfunc = []
        for name in self.final_col_list:
            if self.df[name].dtype == "O":
                self.mapfunc.append(([name],sklearn.preprocessing.LabelBinarizer()))
            else:
                self.mapfunc.append(([name], sklearn.preprocessing.StandardScaler(False)))

        in_mapper = DataFrameMapper(self.mapfunc)
        out_mapper = sklearn.preprocessing.StandardScaler()


        self.trainX = np.array(in_mapper.fit_transform(self.trainX),np.float32)
        self.trainY = np.array(out_mapper.fit_transform(self.trainY.reshape(-1,1)),np.float32)
        self.tindex = self.trainX.shape[0]

#td = trainingData(r"C:\Users\hites\Documents\Visual Studio 2015\Projects\Engine\csv_title_io\temp.csv")
#td.preprocess('Sales')


