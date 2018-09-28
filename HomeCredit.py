import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


path = r"E:/DATA/HousePrice/"
train_file = pd.read_csv(path+"train.csv")
test_file = pd.read_csv(path+"test.csv")
SalePrice = train_file["SalePrice"]
test_id = test_file.Id


############################EDA#############################
#check missing value
train_file.isnull().sum().sort_values(ascending=False)
#PoolQC MiscFeature Alley Fence have more than 1000 missing values
#Drop these columns from train dataset
train = train_file.drop(["PoolQC","MiscFeature","Alley","Fence"],axis=1)
test = test_file.drop(["PoolQC","MiscFeature","Alley","Fence"],axis=1)

train.drop("Id",axis=1,inplace=True)
test.drop("Id",axis=1,inplace=True)

#transform some columns type based on its defination
train.MSSubClass.astype("object")


#Target variable -- SalePrice
sns.distplot(SalePrice)
plt.title("SalePrice")
print("SalePrice Skewness: %10.6f"%train.SalePrice_log.skew())
print("SalePrice Kurtosis: %10.6f"%train.SalePrice_log.kurt())


train["SalePrice_log"] = np.log1p(train.SalePrice)
sns.distplot(train.SalePrice)
plt.title("log1p(SalePrice)")
print("SalePrice_log Skewness: %10.6f"%train.SalePrice_log.skew())
print("SalePrice_log Kurtosis: %10.6f"%train.SalePrice_log.kurt())
train.drop("SalePrice",axis=1,inplace=True)


################numerical columns###################
numerical_columns = list(train.select_dtypes(["float64","int64"]).columns)
train[numerical_columns].isnull().sum().sort_values(ascending=False)
#LotFrontage GarageYrBlt MasVnrArea
train["LotFrontage"].fillna(train["LotFrontage"].mean(),inplace=True)
train["GarageYrBlt"].fillna(train["GarageYrBlt"].mean(),inplace=True)
train["MasVnrArea"].fillna(train["MasVnrArea"].mean(),inplace=True)

test[numerical_columns].isnull().sum().sort_values(ascending=False)
#MasVnrArea TotalBsmtSF garageYrBlt GarageCars
test["MaxVnrArea"].fillna(test["MaxVnrArea"].mean(),inplace=True)
test["TotalBsmtSF"].fillna(test["TotalBsmtSF"].mean(),inplace=True)
test["garageYrBlt"].fillna(test["garageYrBlt"].mean(),inplace=True)
test["GarageCars"].fillna(test["GarageCars"].mean(),inplace=True)



#37 columns
nrows=12
ncols=3
#fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(nrows*4,ncols*3))
num_col_pear = []
for r in range(nrows):
    for c in range(ncols):
        idx = r*3+c
        if idx < len(numerical_columns)-1:
#            sns.regplot(train[numerical_columns[idx]],train["SalePrice_log"],ax=axes[r][c])
            pearson = stats.pearsonr(train[numerical_columns[idx]],train["SalePrice_log"])
            num_col_pear.append(abs(pearson[0]))
#           axes[r][c].set_title("r:%2f  p-value:%2f"%(pearson[0],pearson[1]))
#plt.show()

num_col_pear = pd.DataFrame({"Feature":numerical_columns[:-1],"Pearson":num_col_pear})
#check pearson correlation,choose a Threshold value(0.4 here) for filtering
num_col_pear.sort_values(by="Pearson",ascending=False)
weak_pear_col = list(num_col_pear[num_col_pear.Pearson<0.4]["Feature"])
strong_pear_col = list(num_col_pear[num_col_pear.Pearson>=0.4]["Feature"])

#heatmap
#just for strong-relation columns
sns.heatmap(train[strong_pear_col+["SalePrice_log"]].corr(),cbar=True, square=True, fmt='.2f',annot=True,annot_kws={"size":10})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
#from the plot, we can know maybe some multiple mutual linear problem
#corr > 0.8, we consider it as high-relation columns
#[GrLivArea,TotRmsAbvGrd] [GarageArea,GarageCars]
#compare their correlation with SalePrice_log,
#save the higher one, drop the lower one
#[GrLivArea 0.70,TotRmsAbvGrd 0.53] [GarageArea 0.65,GarageCars 0.68]

train.drop(weak_pear_col,axis=1,inplace=True)
test.drop(weak_pear_col,axis=1,inplace=True)

train.drop(["TotRmsAbvGrd","GarageArea"],axis=1,inplace=True)
test.drop(["TotRmsAbvGrd","GarageArea"],axis=1,inplace=True)

################object columns###################
object_columns = list(train.select_dtypes(["object"]).columns)
#plot: show boxplot of SalePrice_log and each object column
#will update later


#check distribution of each feature
#remove the feature if one class
unbalance_columns = [] 
for col in object_columns:
    temp = train.groupby(col)["SalePrice_log"].count()/train.shape[0]
    print(temp)
    if max(temp.values) > 0.95:
        unbalance_columns.append(col)
    print("#"*50)
print(unbalance_columns)

train.drop(unbalance_columns,axis=1,inplace=True)
test.drop(unbalance_columns,axis=1,inplace=True)


object_nunique = train.select_dtypes(["object"]).apply(pd.Series.nunique)
two_label_columns = list(object_nunique.index[object_nunique<=2])
multi_label_columns = list(object_nunique.index[object_nunique>2])

###################preprocessing#############################
def RMSE(pred,y):
    y = list(y)
    pred = list(pred)
    sum0 = 0.0
    for i in range(len(pred)):
        sum0 += (pred[i]-y[i])*(pred[i]-y[i])
    return math.sqrt(sum0/len(pred))

df_train = train.drop("SalePrice_log",axis=1)
df_test = test.copy()
for col in two_label_columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

df_train = pd.get_dummies(df_train,columns=multi_label_columns)
df_test = pd.get_dummies(df_test,columns=multi_label_columns)

df_train,df_test = df_train.align(df_test,join="inner",axis=1)

train_x, test_x, train_y, test_y = train_test_split(df_train,train["SalePrice_log"])


###########xgboost############
xgb = XGBRegressor(objective="reg:linear",max_depth=10,learning_rate=0.1)
xgb.fit(train_x,train_y)
pred_x = xgb.predict(test_x)
RMSE(pred_x,test_y)
pred_xgb = np.expm1(xgb.predict(df_test))


##########randomForest#############
rf = RandomForestRegressor(n_estimators=100)
rf.fit(train_x,train_y)
pred_x = rf.predict(test_x)
RMSE(pred_x,test_y)
pred_rf = np.expm1(rf.predict(df_test))


best_pred = pd.DataFrame({"Id":test_id,"SalePrice":pred_xgb})
best_pred.to_csv(path+"/prediction.csv",index=False)







