import pandas as pd
import numpy as np
import pickle
import string
import math
import matplotlib.pyplot as plt
import seaborn as sns
#Importing Dataset
df3 = pd.read_csv('C:/Users/Gaurav Mittal/Desktop/577/New folder/Working_data.csv')
df3.isnull().sum(axis=0) 
data =df3.dropna()
data.info()

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
## how to save figure fig.savefig('heat.pdf')

##how to clear mirro image of heat map
'''def heatmap(self,df,mirror):
corr = Final_Data.corr()
    # Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
    # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
   
    if mirror == True:
       #Generate Heat Map, allow annotations and place floats in map
       sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
       #Apply xticks
       plt.xticks(range(len(corr.columns)), corr.columns);
       #Apply yticks
       plt.yticks(range(len(corr.columns)), corr.columns)
       #show plot

    else:
       # Drop self-correlations
       dropSelf = np.zeros_like(corr)
       dropSelf[np.triu_indices_from(dropSelf)] = True# Generate Color Map
       colormap = sns.diverging_palette(220, 10, as_cmap=True)
       # Generate Heat Map, allow annotations and place floats in map
       sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
       # Apply xticks
       plt.xticks(range(len(corr.columns)), corr.columns);
       # Apply yticks
       plt.yticks(range(len(corr.columns)), corr.columns)
    # show plot
    plt.show()
'''


Corr_Data =x_data.copy(deep=True) 
del Corr_Data['Workers_0']

#Create Correlation df
corr = Corr_Data.corr()
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

#Standarize
y1 = Corr_Data.iloc[:, 0].values
x1 = Corr_Data.drop(['CASE_STATUS','VISA_CLASS' ], axis=1)
x1 = StandardScaler().fit_transform(x1)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
principalComponents = pca.fit_transform(x1)

principaldf = pd.DataFrame(data = principalComponents, 
                           columns =['principal Component 1' , 'principal Component 2',
                                     'principal Component 3']  )




