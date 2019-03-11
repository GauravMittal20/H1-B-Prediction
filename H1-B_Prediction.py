import pandas as pd
import numpy as np
import pickle
import string
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

#Importing Dataset
df3 = pd.read_csv('C:/Users/Gaurav Mittal/Desktop/577/New folder/Working_data.csv')
df3.isnull().sum(axis=0) 
data =df3.dropna()
##data.info()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

y = data.iloc[:, 0].values
labelencoder_y = LabelEncoder()
y_data= labelencoder_y.fit_transform(y)


x_data= data.iloc[:,0:13]
x_data['CASE_STATUS']= labelencoder_y.fit_transform(x_data['CASE_STATUS'])
x_data['FULL_TIME_POSITION']= labelencoder_y.fit_transform(x_data['FULL_TIME_POSITION'])
x_data['H1B_DEPENDENT']= labelencoder_y.fit_transform(x_data['H1B_DEPENDENT'])
x_data['WILLFUL_VIOLATOR']= labelencoder_y.fit_transform(x_data['WILLFUL_VIOLATOR'])
x_data['WILLFUL_VIOLATOR']= labelencoder_y.fit_transform(x_data['WILLFUL_VIOLATOR'])
x_data['AGENT_REPRESENTING_EMPLOYER']= labelencoder_y.fit_transform(x_data['AGENT_REPRESENTING_EMPLOYER'])

##Converting 2 column STATE_ZONE and TOTAL_WORKERS into dummy variables
x_data['STATE_ZONE']= labelencoder_y.fit_transform(x_data['STATE_ZONE'])
Dummy1= pd.get_dummies(x_data['STATE_ZONE'], prefix= 'Zone')
x_data = pd.concat([x_data,Dummy1], axis=1)

x_data['TOTAL_WORKERS']= labelencoder_y.fit_transform(x_data['TOTAL_WORKERS'])
Dummy2= pd.get_dummies(x_data['TOTAL_WORKERS'], prefix= 'Workers')
x_data = pd.concat([x_data,Dummy2], axis=1)
del x_data['STATE_ZONE']
del x_data['TOTAL_WORKERS']
del x_data['EMPLOYER_COUNTRY']
del x_data['VISA_CLASS']

Final_Data=x_data.copy(deep=True) 
Final_Data.to_csv('Main.csv')

#correlation matrix
print(Final_Data.corr())
Corelation = Final_Data.corr()
Corelation.to_csv('matrix.csv')

#Heat map
#Create Correlation df
corr = Final_Data.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
#Generate Color Map, red & blue
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()
del Final_Data['Workers_1']


########## assigining x and y variable
y3 = Final_Data.iloc[:, 0].values
x3 = Final_Data.drop(['CASE_STATUS'], axis=1)
#Splitting into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x3, y3, test_size = 0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


###################### Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc,roc_auc_score, log_loss

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 100)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred1 = classifier.predict_proba(X_test)
##Confusion MAtrix
from sklearn.metrics import confusion_matrix
Cm_DT = confusion_matrix(y_test, y_pred)
#Accuracy
Acc_Dt= accuracy_score(y_test, y_pred)
print acc
#Loss
y_pred1 = classifier.predict_proba(X_test)
loss_Dt = log_loss(y_test, y_pred1)
loss= log_loss(y_test, y_pred1, eps=1e-15, normalize=True, sample_weight=None, labels=None)
##ROC
skplt.metrics.plot_roc_curve(y_test, y_pred1)
plt.show()

# ###########################Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred2 = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
Cm_Rf = confusion_matrix(y_test, y_pred2)
#Accuracy
Acc_Rf= accuracy_score(y_test, y_pred2)
#Loss
y_pred3 = classifier.predict_proba(X_test)
loss_Rf = log_loss(y_test, y_pred3)
loss= log_loss(y_test, y_pred3, eps=1e-15, normalize=True, sample_weight=None, labels=None)
#ROC
skplt.metrics.plot_roc_curve(y_test, y_pred3)
plt.show()


##################################K-nearestNeighbours
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred4 = classifier.predict(X_test)
y_pred5 = classifier.predict_proba(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_Knn = confusion_matrix(y_test, y_pred4)
#Accuracy
Acc_Knn= accuracy_score(y_test, y_pred4)
print acc
#Loss
y_pred5 = classifier.predict_proba(X_test)
loss_Knn = log_loss(y_test, y_pred5)
loss= log_loss(y_test, y_pred5, eps=1e-15, normalize=True, sample_weight=None, labels=None)
##ROC
skplt.metrics.plot_roc_curve(y_test, y_pred5)
plt.show()



#### Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred6 = classifier.predict(X_test)
y_pred7 = classifier.predict_proba(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_Xg = confusion_matrix(y_test, y_pred6)
#Accuracy
acc_Xg= accuracy_score(y_test, y_pred6)
print acc
#Loss
loss_Xg = log_loss(y_test, y_pred7)
loss= log_loss(y_test, y_pred7, eps=1e-15, normalize=True, sample_weight=None, labels=None)
##Roc
skplt.metrics.plot_roc_curve(y_test, y_pred7)
plt.show()
