#! /usr/python 
#https://www.kaggle.com/gpreda/santander-value-prediction-extensive-eda
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

path = 'E:/DATA/Santander Value Prediction'
print(os.listdir(path))

######read data######
train = pd.read_csv(path+'/train.csv')
test = pd.read_csv(path+'/test.csv')

######glimpse data######
train.head()
test.head()
#conclusion:
#ID:people's numbers with 9 digits
#target:target variable, numerical
#other 4991 columns:there are hexa large numbers with 9 digits.
#                   Most of the columns have 0 values, the dataset is 
#                   sparse. The columns types seems to be integers and
#                   reals. 


######Check missing data######
def check_nulls(df):
    nulls = df.isnull().sum(axis=0).reset_index()
    nulls.columns = ['columns','missings']
    nulls = nulls[nulls['missings']>0]
    nulls = nulls.sort_values(by='missings')
    return nulls

check_nulls(train)#no missing data
check_nulls(test)#no missing data

           
#######Check data sparsity######
def check_sparsity(df):
    non_zeros = (df.ne(0).sum(axis=1)).sum()
    total = df.shape[0]*df.shape[1]
    zeros = total - non_zeros
    sparsity = round(zeros/total*100,2)
    density = round(non_zeros/total*100,2)
    
    print("Zeros:%d\nNon_zeros:%d\nTotal:%d\nSparsity:%.2f\nDensity:%.2f"
          %(zeros,non_zeros,total,sparsity,density))
    return density

d1=check_sparsity(train)
#Zeros:21554760
#Non_zeros:709027
#Total:22263787
#Sparsity:96.82
#Density:3.18
               
d2=check_sparsity(test)
#Zeros:242805367
#Non_zeros:3509897
#Total:246315264
#Sparsity:98.58
#Density:1.42

######Data Exploration######
#Feature Type
types = train.dtypes.reset_index()
types.columns = ['Count','Types']
types.groupby('Types').aggregate('count').reset_index()
#     Types  Count
#0    int64   3147
#1  float64   1845
#2   object      1

data = []
for feature in train.columns:
    if feature == 'ID':
        use = 'ID'
    elif feature == 'target':
         use = 'target'
    else:
        use = 'input'
    
    keep = True
    if feature == 'ID':
        keep = False
    
    dtype = train[feature].dtype
                 
    feature_dictionary = {
            'variable':feature,
            'use':use,
            'keep':keep,
            'dtype':dtype
            }
    data.append(feature_dictionary)
metadata = pd.DataFrame(data,columns = ['variable','use','keep','dtype'])
metadata.set_index('variable')
metadata.head(10)

pd.DataFrame({'count':metadata.groupby(['dtype'])['dtype'].size()}).reset_index()

#Data sparsity per column type
int_data = []
var = metadata[(metadata['dtype']=='int64')&(metadata['use']=='input')].index
density_int_train = check_sparsity(train[var])
density_int_test = check_sparsity(test[var-1])

var = metadata[(metadata['dtype']=='float64')&(metadata['use']=='input')].index
density_float_train = check_sparsity(train[var])
density_float_test = check_sparsity(test[var-1])

temp = {'Dataset':['Train','Test'],'All':[d1,d2],
        'Integer':[density_int_train,density_int_test],
        'Float':[density_float_train,density_float_test]}
density_data = pd.DataFrame(temp,columns = ['Dataset','All','Integer','Float'])
density_data =density_data.set_index('Dataset')

#Target Variable
def plot_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s"%feature)
    sns.distplot(df[feature].dropna(),color=color,kde=True,bins=100)
    plt.show()

def plot_log_distribution(df,feature,color):
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(np.log1p(df[feature]).dropna(),color=color, kde=False,bins=100)
    plt.title("Distribution of log(target)")
    plt.show()
    
plot_distribution(train,"target","blue")
plot_log_distribution(train,"target",'blue')

#Distribution of non-zero features values per row

non_zeros = (train.ne(0).sum(axis=1))
plt.figure(figsize=(10,6))
plt.title("Distribution of log(number of non-zeros per row) - train set")
sns.distplot(np.log1p(non_zeros),color='red',kde=False,bins=100)
plt.show()







#Highly correlated features





    