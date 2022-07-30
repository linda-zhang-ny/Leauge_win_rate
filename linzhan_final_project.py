# -*- coding: utf-8 -*-
"""
Linda Zhang
Class: CS 677
Date:4/22/2022
Final Project

In my project I want to look at the game League of Legends. The question that
I want to see if the features that I selected are significant in terms of predicting
the win rate of one team.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


#importing data into python
df=pd.read_csv('Challenger_Ranked_Games.csv')

#select the columns that are going to be used
df = df[['blueWins','blueWardPlaced','blueKills','blueAvgLevel','blueTotalGold','blueKillingSpree']]
#randomly select 2000 columns to use
df = df.sample(600,random_state=42)
print(df.head())

#seperate the the blue team wins vs where blue team lost
df_win=df[df['blueWins']==1]
df_lose=df[df['blueWins']==0]
train, test=train_test_split(df, test_size=0.75, random_state=1)

train_win=train[train['blueWins']==1]

train_lost=train[train['blueWins']==0]

#plot out the team wins and we can see that Total Gold and Average Level has highest accuracy
sns.set_context( rc={"axes.labelsize":21})
win_plot=sns.pairplot(train_win[['blueWardPlaced','blueKills','blueAvgLevel','blueTotalGold','blueKillingSpree']],height=3.5)
win_plot.fig.suptitle('Team Wins')
plt.show()

lose_plot=sns.pairplot(train_lost[['blueWardPlaced','blueKills','blueAvgLevel','blueTotalGold','blueKillingSpree']])
lose_plot.fig.suptitle('Team Lost')
plt.show()


#Creating a table to keep track of hte mean and standard deviation
def graph_calc(data):
    master_list=[]
    for i in [1,2,3,4,5]:
        m=round(data.mean()[i],3)
        sd=round(data.std()[i],3)
        master_list.append(m)
        master_list.append(sd)
    return master_list

sd_and_mean_table = pd.DataFrame(
    [
     graph_calc(train_lost),
     graph_calc(train_win),
     graph_calc(df)
     ],
    index=[0,1,'all'],
    columns=['u(Wards Placed)','sd(Wards Placed)','u(Kills)','sd(Kills)','u(Avg Level)','sd(Avg Level)','u(Total Gold)','sd(Total Gold)','u(Killing Spree)','sd(Killing Spree)'],
)
print(sd_and_mean_table)



#Using K-nn classifier
x_test=test[['blueWardPlaced','blueKills','blueAvgLevel','blueTotalGold','blueKillingSpree']].values
x_train=train[['blueWardPlaced','blueKills','blueAvgLevel','blueTotalGold','blueKillingSpree']].values

scaler = StandardScaler()
scaler.fit(x_train)

x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


y_test=test['blueWins'].values
y_train=train['blueWins'].values

accurate=[]

#Using the highest K value we p
for k in range(2,20,2):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_predictor=knn.predict(x_test)
    accuracy=metrics.accuracy_score(y_predictor,y_test)
    accurate.append(accuracy)
    print('k=' + str(k) + ', Accuracy:' + str(round(accuracy,4)))

plt.clf()
plt.plot(range(2,20,2),accurate,marker='o')
plt.title('K vs Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()


knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
new_y_predictor=knn.predict(x_test)
n_tp = 0
n_fp = 0
n_tn = 0
n_fn = 0

for i in range(len(y_predictor)):
    if y_predictor[i]==1:
        if y_test[i]==1:
            n_tp +=1
        else:
            n_fp +=1
    else:
        if y_test[i]==1:
            n_fn +=1
        else:
            n_tn +=1
    i+=1
n_tpr = round(n_tp / (n_tp + n_fn),2)
n_tnr = round(n_tn / (n_tn + n_fp),2)

accurate_table = pd.DataFrame(
    {
        'tp': [n_tp],
        'fp': [n_fp],
        'tn': [n_tn],
        'fn': [n_fn],
        'accuracy': max(accurate),
        'tpr': [n_tpr],
        'tnr': [n_tnr]
    }
)

print(accurate_table)


def feature_selection(columns):
    scaler = StandardScaler()
    new_x_test=test[columns]
    scaler.fit(new_x_test)
    
    new_y_test=(test['blueWins'].values)

    feature_knn=KNeighborsClassifier(n_neighbors=4)
    feature_knn.fit(new_x_test,new_y_test)
    new_feautre_pred=feature_knn.predict(new_x_test)
    accuracy=round(metrics.accuracy_score(new_feautre_pred,new_y_test),3)
    return accuracy
print('feature selection-4_____________________________________')
print('Kills ,Average Level,Total Gold,Killing Spree')
print(feature_selection(['blueKills','blueAvgLevel','blueTotalGold','blueKillingSpree']))

print('Ward Placed,Average Level,Total Gold,Killing Spree')
print(feature_selection(['blueWardPlaced','blueAvgLevel','blueTotalGold','blueKillingSpree']))

print('Ward Placed,Kills,Total Gold,Killing Spree')
print(feature_selection(['blueWardPlaced','blueKills','blueTotalGold','blueKillingSpree']))
print('Ward Placed,Kills,Average Level,Total Gold')
print(feature_selection(['blueWardPlaced','blueKills','blueAvgLevel','blueTotalGold']))


'''
For KNN I noticed that the highest accuracy happens where wards placed was 
used so I have decided to keep wards placed when it came to selecting three
features. 
'''

print('feature selection-3_____________________________________')
print('Ward Placed,Kills,Average Level')
print(feature_selection(['blueWardPlaced','blueKills','blueAvgLevel']))

print('Ward Placed,Kills,Total Gold')
print(feature_selection(['blueWardPlaced','blueKills','blueTotalGold']))

print('Ward Placed,Kills,Killing Spree')
print(feature_selection(['blueWardPlaced','blueKills','blueKillingSpree']))

print('Ward Placed,Average Level,TotalGold')
print(feature_selection(['blueWardPlaced','blueAvgLevel','blueTotalGold']))

print('Ward Placed,Average Level,Killing Spree')
print(feature_selection(['blueWardPlaced','blueAvgLevel','blueKillingSpree']))

print('Ward Placed,Total Gold,Killing Spree')
print(feature_selection(['blueWardPlaced','blueTotalGold','blueKillingSpree']))


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(x_train,y_train)
logistic_prediction=log_reg_classifier.predict(x_test)

l_tp = 0
l_fp = 0
l_tn = 0
l_fn = 0
for i in range(len(logistic_prediction)):
    if logistic_prediction[i]==1:
        if y_test[i]==1:
            l_tp +=1
        else:
            l_fp +=1
    else:
        if y_test[i]==1:
            l_fn +=1
        else:
            l_tn +=1
    i+=1
l_tpr = round(l_tp / (l_tp + l_fn),2)
l_tnr = round(l_tn / (n_tn + l_fp),2)
log_accuracy=round(metrics.accuracy_score(logistic_prediction,y_test),3)
log_accurate_table = pd.DataFrame(
    {
        "tp": [l_tp],
        "fp": [l_fp],
        "tn": [l_tn],
        "fn": [l_fn],
        "accuracy": [log_accuracy],
        "tpr": [l_tpr],
        "tnr": [l_tnr]
    }
)
print('Logistic Regression_____________________________________')
print(log_accurate_table )
def log_feature_selection(columns):
    scaler = StandardScaler()
    new_log_x_test=test[columns]
    scaler.fit(new_log_x_test)
    
    new_log_y_test=(test['blueWins'].values)
    
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(new_log_x_test,new_log_y_test)
    new_logistic_prediction=log_reg_classifier.predict(new_log_x_test)
    accuracy=round(metrics.accuracy_score(new_logistic_prediction,new_log_y_test),3)
    return accuracy

print('feature selection-4_____________________________________')
print('Kills ,Average Level,Total Gold,Killing Spree')
print(log_feature_selection(['blueKills','blueAvgLevel','blueTotalGold','blueKillingSpree']))

print('Ward Placed,Avgerage Level,Total Gold,Killing Spree')
print(log_feature_selection(['blueWardPlaced','blueAvgLevel','blueTotalGold','blueKillingSpree']))

print('Ward Placed,Kills,Total Gold,Killing Spree')
print(log_feature_selection(['blueWardPlaced','blueKills','blueTotalGold','blueKillingSpree']))

print('Ward Placed,Kills,Avgerage Level,Total Gold')
print(log_feature_selection(['blueWardPlaced','blueKills','blueAvgLevel','blueTotalGold']))


'''
Using three features selecting kills, average level, total gold, and killing
spree because those four had the highest accuracy so far. I thought it would
do something similar like it did in Knn features selection. But to my suprise
the accuracy actually went lower. 
'''
print('feature selection-3_____________________________________')
print('Kills ,Average Level,Total Gold')
print(log_feature_selection(['blueKills','blueAvgLevel','blueTotalGold']))

print('Kills ,Average Level,Killing Spree')
print(log_feature_selection(['blueKills','blueAvgLevel','blueKillingSpree']))

print('Kills ,Average Level,Total Gold,Killing Spree')
print(log_feature_selection(['blueKills','blueTotalGold','blueKillingSpree']))

print('Average Level,Total Gold,Killing Spree')
print(log_feature_selection(['blueAvgLevel','blueTotalGold','blueKillingSpree']))


