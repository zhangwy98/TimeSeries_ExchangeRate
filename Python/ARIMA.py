import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
from Model.UnobservedComponentModel import UnobservedComponentModel
from matplotlib import pyplot as plt
import math

def mean_error(pred,real):
    return np.mean([pred[i]-real[i] for i in range(len(pred))])
def mean_absolute_error(pred,real):
    return np.mean([np.abs(pred[i]-real[i]) for i in range(len(pred))])
def root_mean_square_error(pred,real):
    return np.sqrt(np.mean([pow(pred[i]-real[i],2) for i in range(len(pred))]))

#You can change dataset here
dataset = 'EXUSEU'
data = pd.read_csv("Data/"+dataset+".csv")
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

#Test Performance of Random Walk with Drift
mae_ = []
rmse_ = []
for l in [1,3,6,12]:
    pred = [0]*(startPrediction+l-1)
    for i in range(totalData-startPrediction-l+1):
        monthly_change = [tsValuesLog[j]-tsValuesLog[j-l] for j in range(i+l,i+startPrediction)]
        mean_monthly_change = np.mean(monthly_change)
        pred.append(tsValuesLog[startPrediction+i-1]+mean_monthly_change)
    mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
    rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])) 
PredictionOutputMAE['RandomWalkWithDrift'] = mae_
PredictionOutputRMSE['RandomWalkWithDrift'] = rmse_

#Test Performance of ARMA Model
for p,q in [[1,1],[2,0]]:
    mae_ = []
    rmse_ = []
    for l in [1,3,6,12]:
        pred = [0]*(startPrediction+l-1)
        for i in range(totalData-startPrediction-l+1):
            try:
                model = ARMA(tsValuesLog[i:i+startPrediction],order=(p,q))
                result = model.fit()
                pred.append(result.predict(start=startPrediction,end=startPrediction+l-1)[l-1])
            except ValueError:
                pred.append(tsValuesLog[i+startPrediction-1])
        mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
        rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])) 
    PredictionOutputMAE['ARMA_'+str(p)+'_'+str(q)] = mae_
    PredictionOutputRMSE['ARMA_'+str(p)+'_'+str(q)] = rmse_


#Test Performance of ARIMA Model
for p,q in [[1,1],[1,0],[0,1]]:
    mae_ = []
    rmse_ = []
    for l in [1,3,6,12]:
        pred = [0]*(startPrediction+l-1)
        for i in range(totalData-startPrediction-l+1):
            try:
                model = ARMA(tsValuesLog_d_1[i:i+startPrediction],order=(p,q))
                result = model.fit()
                pred.append(sum(result.predict(start=startPrediction,end=startPrediction+l-1))+tsValuesLog[i+startPrediction-1])
            except ValueError:
                pred.append(tsValuesLog[i+startPrediction-1])
        mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
        rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:])) 
    PredictionOutputMAE['ARIMA_'+str(p)+'_'+str(q)] = mae_
    PredictionOutputRMSE['ARIMA_'+str(p)+'_'+str(q)] = rmse_

#Test Unobserved Model
mae_ = []
rmse_ = []
for l in [1,3,6,12]:
    pred = [0]*(startPrediction+l-1)
    for i in range(totalData-startPrediction-(l-1)):
        model = UnobservedComponentModel()
        model.fit(tsValuesLog[i:i+startPrediction])
        pred.append(model.predict(l)[l-1])
    mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
    rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
PredictionOutputMAE["UnobservedComponentModel"] = mae_
PredictionOutputRMSE["UnobservedComponentModel"] = rmse_

print(PredictionOutputMAE)
