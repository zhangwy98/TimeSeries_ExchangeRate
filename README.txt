The code is also available on github:
https://github.com/zhangwy98/TimeSeries_ExchangeRate/
The code consists of two parts: R language and Python language.

********R code********
We use R mainly in model fitting (Section 3 and 4 in our report).
The executable R code is Fit.R.
We also keep our record in six notebooks.
You may check our results open the .ipynb files in Jupyter.

We are sorry for that we are not familiar with IO in R language, and thus we do some work with the help of python, mainly:
a) we input the data which we have already processed with python. You may check our work in Python/Forecast/Forecast_LinearModel_prediciton_*.ipynb
b) when dealing with out-of-sample prediction and evaluation, we have difficulty with R package. Thus we use python to generate script and evaluate the result. You may check our work in Python/Forecast/CH_GARCH.ipynob and Python/Forecast/EU_GARCH.ipynb

********python code********
We use python mainly in data pre-processing, observation and nonlinear model (networks and hybrid models) fitting.

a) Observations
You may run Python/BasicObs.py and change the dataset in the code (marked in the .py file)
Or you can simply check our result in Python/Observation_CH.ipynb Python/Observaion_EU.ipynb
b) Check cointergration
You may check the result in Python/Fit_cointergration.ipynb
c) Fit Exogenous ARIMA
You may run Python/ExoARMA.py
Or you can simply check our result in Python/Fit_Macro_Exogenous_ARIMA(1,0,1).ipynb
d) Build Network and Compare Forecast 
	1) Run python ARIMA.py, you can get result for ARIMA models
	   You may check our result in Forecast/Forecast_LinearModel_prediciton_*.ipynb
	2) Run python Network.py, you can get result for networks and hybrid models.
	   It requires some time. :)
	3) The total result can be found in excels in Result.
	   If you are interested in detail, you may check Result/Detail, here are some raw .csv files. :)
	   The final aggregated result can be found in Result/total_mas.xlsx and Result/total_rmse.xlsx
	   
********Python packages********
pandas==0.19.2
numpy==1.11.3
statsmodels==0.8.0
scikit-learn==0.18.1
scipy==0.18.1
keras
tensorflow