from math import sqrt
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

# path operations
import os
dirname = os.path.dirname(__file__)
def path(rel):
    return os.path.join(dirname, rel)

# Input data
df= pd.read_csv(path("./../dat/forest_fires.csv"))
print(df.head())
print(df.describe())

# Check dataset for null values
# plt.figure(figsize=(10,10))
# sns.heatmap(df.isnull(),cbar=False)
# plt.show()
## NO DATA MISSING. CONTINUE WITHOUT PLOTTING THIS

# remove every data whose area is 0
df = df[df.area > 0]

# show description of data that is left
print(df.describe())

# Abandoning features that don't make sense.
#x1,x2,x3 are the name of features
drop_elements = ['FFMC', 'DMC', 'DC', 'ISI','X','Y']
df = df.drop(drop_elements, axis = 1)

# show description of data that is left
print(df.describe())

#
# Split incoming data
#
from sklearn.model_selection import train_test_split
x = df.drop(['area'],axis=1)
y = df['area']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


#
# FEATURE SELECTION
#
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
##
## Random Forest
## 
plt.figure()
rf = RandomForestRegressor ()
rf.fit(x,y) # x is train data, y is target data
score = rf.score(x_test, y_test)
print(score)
# importances = pd.DataFrame({'feature':x.columns,'importance':np.round(rf.feature_importances_,3)})
# importances = importances.sort_values('importance',ascending=False).set_index('feature')
# importances.head(25).plot(kind='bar',alpha=1,width = 0.8,facecolor = 'yellowgreen',edgecolor = 'white',lw=1)
# plt.tick_params(labelsize=18)
# font1 = {'family': 'Times New Roman','weight': 'normal'}
# legend = plt.legend(prop=font1)
# font2 = {'family': 'Times New Roman','weight': 'normal'}
# plt.xlabel('Features', font2)
# font3 = {'family': 'Times New Roman','weight': 'normal'}
# plt.ylabel('Correlation', font3)

# plt.show()

# result = rf.predict(np.array([8,6,7,2,28.2,29,1.8,0]).reshape(1,-1)) # result should be 5.86
# print(result)