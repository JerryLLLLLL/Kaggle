# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 00:03:25 2018

@author: Jerry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import xgboost as xgb
from xgboost import sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
#import seaborn as sns
#import statsmodels.api as sm
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.stattools import acf,pacf
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
#from statsmodels.tsa.arima_model import ARIMA

###################data process#######################
path = r"D:/Script/Data"
dateparser = lambda date: pd.datetime.strptime(date,"%Y-%m-%d")
train_file = pd.read_csv(path+"/train.csv")
test_file = pd.read_csv(path+"/test.csv")
date_train = train_file["date"].apply(dateparser)
date_test = test_file["date"].apply(dateparser)
#Turn index into time format
train_file = train_file.set_index(date_train)
test_file = test_file.set_index(date_test)


train = train_file.copy()
test = test_file.copy()


#After notice, we decide to split the "Date"
#Considering some special day, such as weekday, weekend, beginning of month, end of month, we can add some new features.
#Below function is to add new columns
def expand_df(df):
    data = df.copy()
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["wday"] = data.index.dayofweek
    data = data.drop("date",axis=1,inplace=False)
    return data
train_x = expand_df(train).drop("sales",axis=1)
train_y = expand_df(train)["sales"]

test_x = expand_df(test)

#Simple Analyse & Plot
#Show Time--Store Sales & Time--Item Sales Plot
train_all = expand_df(train)
agg_year_item = pd.pivot_table(train_all,index="year",columns="item",values="sales",aggfunc=np.mean).values
agg_year_store = pd.pivot_table(train_all, index='year', columns='store',values='sales', aggfunc=np.mean).values

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.show()


#handling dummy varibles
train_x_dummy = pd.get_dummies(train_x,prefix=["store","item","wday"],columns=["store","item","wday"])
test_x_dummy = pd.get_dummies(test_x,prefix=["store","item","wday"],columns=["store","item","wday"])
test_x_dummy.drop("id",axis=1,inplace=True)

###################build model: xgboost##############################
#Measurement
def SMAPE(pred,sales):
    if len(pred)!=len(sales):
        raise BaseException("Length of two variables are different!")
    n = len(pred)
    smape = 0.0
    for i in range(n):
        temp = abs(pred[i]-sales[i])/((abs(pred[i])+abs(sales[i]))/2)
        smape += temp
    return "error",smape

"""
#Model Function, return Model&SMAPE
def test_model(param,data_x,data_y,metric=SMAPE):
    model = sklearn.XGBRegressor(**param)
    #train_x = data_x[data_x.index.year!=2017]
    #test_x = data_x[data_x.index.year==2017]
    #train_y = data_y[data_y.index.year!=2017]
    #test_y = data_y[data_y.index.year==2017]
    train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.2,random_state=28)
    print("Start fitting...")
    model.fit(train_x,train_y,eval_metric="auc")
    print("Fitting done. Start predicting...")
    pred = model.predict(test_x)
    print("Calculating SMAPE...")
    test_plot = test_x.copy()
    test_plot["pred"] = pred
    test_plot["sales"] = test_y.values
    if "date" in test_plot.columns:
        test_plot.drop("date",axis=1,inplace=True)
    test_plot[(test_plot["store"]==2)&(test_plot["item"]==2)][["sales","pred"]].plot()
    smape = metric(pred,test_y)
    return model,smape
"""    
#param = {"learning_rate":0.1,"n_estimators":1000,"max_depth":5,
# "min_child_weight":1,"gamma":0,"subsample":1,"colsample_bytree":1,
# "objective":'reg:linear',"nthread":4,"scale_pos_weight":1,"seed":27}
param = {"learning_rate":0.8,"gamma":0,"subsample":1,"colsample_bytree":1,"max_depth":5,
         "objective":'reg:linear',"seed":27}
model = sklearn.XGBRegressor(**param)
"""
param_cv_1 = {"learning_rate":[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15]}
param_cv_2 = {"n_estimators":[int(x) for x in np.linspace(100,2000,20)]}
param_cv_3 = {"max_depth":[3,4,5,6,7,8]}
param_cv_4 = {"min_child_weight":[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]}
param_cv_5 = {"scale_pos_weight":[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]}

def choose_best_param(model,param_cv,data_x,data_y):
    clf = GridSearchCV(estimator=model,param_grid=param_cv,error_score=SMAPE)
    clf.fit(data_x,data_y)
    return clf.best_params_


param_all = [param_cv_1,param_cv_2,param_cv_3,param_cv_4,param_cv_5]
best_param = param.copy()
for p in param_all:
    temp_param = choose_best_param(model,p,train_x_dummy,train_y)
    best_param.update(temp_param)


update_model = sklearn.XGBRegressor(**best_param)
update_model.fit(train_x_dummy,train_y)  
"""
model.fit(train_x_dummy,train_y)
pred_test = model.predict(test_x_dummy) 
uid = test_x["id"]

result = pd.DataFrame({"id":uid,"pred":pred_test})
result.index = range(len(result))
result.to_csv(path+"/Prediction.csv")

