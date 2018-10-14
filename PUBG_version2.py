import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = r"E:\DATA\PUBG"
train = pd.read_csv(path+"/train.csv")
test= pd.read_csv(path+"/test.csv")


features = list(train.columns)
for i in ["Id","groupId","matchId","numGroups","winPlacePerc"]:
    features.remove(i)
print(features)

#We can divide the features into several parts:
#Personel Kill[damageDealt,DBNOs,headshotKills,killPlace,killPoints,kills,killStreaks,longestKill,roadKills]
#TeamWork[assists,heals]
#Survive[boosts,revives]
#Run[rideDistance,swimDistance,walkDistance]
#Others[vehicleDestroys,weaponsAcquierd,winPoints,maxPlace,teamKills]

##############personal##############
#Kills & DBNOs
data = train.copy()
data.loc[data.kills > 10,"kills"] = 10
data.loc[data.DBNOs > 10,"DBNOs"] = 10       

sns.distplot(data["kills"],kde=False)
sns.boxplot(x="kills",y="winPlacePerc",data=data)        
sns.boxplot(x="DBNOs",y="winPlacePerc",data=data)


#damageDealt
data.damageDealt.describe().drop("count")
sns.jointplot(x="winPlacePerc",y="damageDealt",data=data)
#slightly trends


#longestKill
data.longestKill.describe().drop("count")
sns.jointplot(x="winPlacePerc",y="longestKill",data=data)
#should good at sniper rifle

#############teamwork#############
#teamkills
team = train.groupby("groupId").mean()
team.drop(["Id","matchId","numGroups"],axis=1,inplace=True)

lists = ["kills","DBNOs","assists","boosts"]
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
for i in range(2):
    for j in range(2):
        idx = i*2+j
        if idx < 4:
            sns.boxplot(x=lists[idx],y="winPlacePerc",data=data,ax=axes[i][j])
            axes[i][j].set_title(lists[idx])
fig.suptitle()
plt.tight_layout()
plt.show()

