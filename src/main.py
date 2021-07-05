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

def monthToNum(shortMonth):
    return {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9, 
            'oct': 10,
            'nov': 11,
            'dec': 12
    }[shortMonth]

def dayToNum(shortDay):
    return {
            'mon': 1,
            'tue': 2,
            'wed': 3,
            'thu': 4,
            'fri': 5,
            'sat': 6,
            'sun': 7
    }[shortDay]

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
drop_elements = ['FFMC', 'DMC', 'DC', 'ISI']
df = df.drop(drop_elements, axis = 1)

# show description of data that is left
print(df.describe())


plt.figure()
plt.plot(range(len(df['area'])), df['area'])
plt.yscale('log')
plt.show()

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
rf = RandomForestRegressor (max_features=100)
rf.fit(x,y) # x is train data, y is target data
importances = pd.DataFrame({'feature':x.columns,'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(25).plot(kind='bar',figsize=(30,20),alpha=1,width = 0.8,facecolor = 'yellowgreen',edgecolor = 'white',lw=1,fontsize=40)
plt.tick_params(labelsize=18)
font1 = {'family': 'Times New Roman','weight': 'normal','size': 50,}
legend = plt.legend(prop=font1)
font2 = {'family': 'Times New Roman','weight': 'normal','size': 50,}
plt.xlabel('Features', font2)
font3 = {'family': 'Times New Roman','weight': 'normal','size': 50,}
plt.ylabel('Correlation', font3)

plt.show()