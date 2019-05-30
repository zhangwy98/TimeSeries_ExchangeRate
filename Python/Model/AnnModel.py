import pandas as pd
import numpy as np
import keras
import keras.backend
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Activation
import tensorflow as tf
import random

class AnnModel:
    def __init__(self,first_layer=7,second_layer=6,third_layer=1,feature_size=8,startPrediction=100):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.third_layer = third_layer
        self.feature_size = feature_size
        self.startPrediction = startPrediction
        
    def fit(self,data):
        keras.backend.clear_session()
        np.random.seed(1024)
        random.seed(1024)
        tf.set_random_seed(1024)
        
        self.data = data
        self.model = Sequential()
        self.model.add(Dense(self.first_layer,input_shape=(self.feature_size,)))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(self.second_layer))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(self.third_layer))
        
        feature_vec = []
        label_vec = []
        for i in range(self.feature_size,len(data)):
            label_vec.append(data[i])
            val = data[i-self.feature_size:i]
            feature_vec.append(val)
        x_train = np.array(feature_vec)
        y_train = np.array(label_vec)
        
        self.model.compile(loss='mean_absolute_error',optimizer=keras.optimizers.Adam())
        self.model.fit(x_train,y_train,epochs=100,verbose=0)
        return 
    
    def predict(self,next_n):
        ret = []
        for i in range(next_n):
            x = self.data[-self.feature_size:]
            x = np.array(x)
            x = x.reshape((-1,self.feature_size))
            y = self.model.predict(x)[0][0]
            self.data.append(y)
            ret.append(y)
        return ret