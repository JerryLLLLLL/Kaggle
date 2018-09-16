import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import xgboost as xgb
from xgboost import sklearn
import math
from sklearn.model_selection import train_test_split
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
#将index转化为时间
train_file = train_file.set_index(date_train)
test_file = test_file.set_index(date_test)

'''
date2int = lambda date: int(date[0:4]+date[5:7]+date[8:10])
train = train_file
test = test_file
train["date"] = train["date"].apply(date2int)
test["date"] = test["date"].apply(date2int)
train_x = train[["store","item","date"]]
test_x = test[["store","item","date"]]
train_y = train["sales"]
'''
train = train_file.copy()
test = test_file.copy()

#观察数据后,决定对日期值进行拆分
#可以考虑日期的特殊性,如周中,周末,月初,月末,增加新特征
#用于添加列
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

#简单分析 画图
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
    return smape


#Model Function to choose best parameters
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
    
param = {"learning_rate":0.1,"n_estimators":1000,"max_depth":5,
 "min_child_weight":1,"gamma":0,"subsample":1,"colsample_bytree":1,
 "objective":'reg:linear',"nthread":4,"scale_pos_weight":1,"seed":27}
model,smape = test_model(param,train_x_dummy,train_y)

        


    
        
pred_test = model.predict(test_x) 
