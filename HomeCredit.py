
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


path = r'E:\DATA\CreditRisk'
os.listdir(path)

application_test = pd.read_csv(path+"/application_test.csv")
application_train = pd.read_csv(path+"/application_train.csv")

print('Size of application_train data', application_train.shape)

####check missing data####
total = application_train.isnull().sum().sort_values(ascending=False)
percent = total/application_train.shape[0] * 100
missing_application_train_data = pd.concat([total,percent],axis=1,join="inner",keys=["Total","Percent"])



remove_list = missing_application_train_data[missing_application_train_data.Percent > 60].index.values

app_train = application_train.drop(remove_list,axis=1)
target = app_train.TARGET

app_train.dtypes.value_counts()

object_set_counts = app_train.select_dtypes(["object"]).apply(pd.Series.nunique,axis=0)
two_label_column = object_set_counts[object_set_counts==2].index.values
multi_label_column = object_set_counts[object_set_counts>2].index.values                                    





























                           