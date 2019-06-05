import pandas as pd
import numpy as np
import math
import keras
import keras.backend
from Model.LstmModel import LstmModel
from Model.AnnModel import AnnModel
import tensorflow as tf
import random

def mean_error(pred,real):
    return np.mean([pred[i]-real[i] for i in range(len(pred))])
def mean_absolute_error(pred,real):
    return np.mean([np.abs(pred[i]-real[i]) for i in range(len(pred))])
def root_mean_square_error(pred,real):
    return np.sqrt(np.mean([pow(pred[i]-real[i],2) for i in range(len(pred))]))


#You can change dataset here
dataset = 'EXUSEU'
data = pd.read_csv('Data/'+dataset+".csv")
tsValues = list(data[dataset])
startPrediction = 100
totalData = len(tsValues)
#log data
tsValuesLog = [math.log(ele) for ele in tsValues]
#first difference
tsValuesLog_d_1 = [0]
tsValuesLog_d_1 += [tsValuesLog[i+1]-tsValuesLog[i] for i in range(totalData-1)]

PredictionOutputMAE = pd.DataFrame(index=['monthly','3_month','6_month','12_month'])
PredictionOutputRMSE = pd.DataFrame(index=['monthly','3_month','6_month','12_month'])

#benchmark Random Walk
mae_ = []
rmse_ = []
for l in [1,3,6,12]:
    pred = [0]*(startPrediction+l-1)
    for i in range(totalData-startPrediction-l+1):
        pred.append(tsValuesLog[startPrediction-1+i])
    mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
    rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
PredictionOutputMAE['RandomWalk'] = mae_
PredictionOutputRMSE['RandomWalk'] = rmse_

#Test pure ANN performance/use Random Walk basic model
for feature_len in [1,2]:
    mae_ = []
    rmse_ = []
    for l in [1,3,6,12]:
        pred = [0]*(startPrediction+l-1)
        for i in range(totalData-startPrediction-(l-1)):
            model = AnnModel(feature_size=feature_len)
            model.fit(tsValuesLog[i:i+startPrediction])
            pred.append(model.predict(l)[l-1])
            #print(i,"end")
        mae = mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        rmse = root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        mae_.append(mae)
        rmse_.append(rmse)
    name = 'Ann_Pure_'+str(feature_len)
    PredictionOutputMAE[name] = mae_
    PredictionOutputRMSE[name] = rmse_

#Test hybrid ANN performance/use Random Walk basic model
for feature_len in [1,2,3,4]:
    mae_ = []
    rmse_ = []
    for l in [1,3,6,12]:
        pred = [0]*(startPrediction+l-1)
        for i in range(totalData-startPrediction-(l-1)):
            model = AnnModel(feature_size=feature_len)
            model.fit(tsValuesLog_d_1[i:i+startPrediction])
            noise_list = model.predict(l)
            pred.append(tsValuesLog[i+startPrediction-1]+sum(noise_list))
        mae = mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        rmse = root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        mae_.append(mae)
        rmse_.append(rmse)
    name = 'Ann_Hybrid_'+str(feature_len)
    PredictionOutputMAE[name] = mae_
    PredictionOutputRMSE[name] = rmse_

#Test Pure LSTM performance/use Random Walk basic model
for feature_len in [1,2]:
    mae_ = []
    rmse_ = []
    for l in [1,3,6,12]:
        pred = [0]*(startPrediction+l-1)
        for i in range(totalData-startPrediction-(l-1)):
            model = LstmModel(sample_num=100-feature_len,feature_length_used=feature_len)
            model.fit(tsValuesLog[i:i+startPrediction])
            pred.append(model.predict(l)[l-1])
        mae = mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        rmse = root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        mae_.append(mae)
        rmse_.append(rmse)
    name = 'LSTM_Pure_'+str(100-feature_len)+'_'+str(feature_len)
    PredictionOutputMAE[name] = mae_
    PredictionOutputRMSE[name] = rmse_

#Test hybrid LSTM performance/use Random Walk basic model
for feature_len in [1,2]:
    mae_ = []
    rmse_ = []
    for l in [1,3,6,12]:
        pred = [0]*(startPrediction+l-1)
        for i in range(totalData-startPrediction-(l-1)):
            model = LstmModel(sample_num=100-feature_len,feature_length_used=feature_len)
            model.fit(tsValuesLog_d_1[i:i+startPrediction])
            noise_list = model.predict(l)
            noise_sum = sum(noise_list)
            pred.append(tsValuesLog[i+startPrediction-1]+noise_sum)
            #print("prediction:",)
        mae = mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        rmse = root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])
        mae_.append(mae)
        rmse_.append(rmse)
    name = 'LSTM_Hybrid'+str(100-feature_len)+'_'+str(feature_len)
    PredictionOutputMAE[name] = mae_
    PredictionOutputRMSE[name] = rmse_

print(PredictionOutputMAE)
print(PredictionOutputRMSE)