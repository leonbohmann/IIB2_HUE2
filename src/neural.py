from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd

# path operations
import os
dirname = os.path.dirname(__file__)
def path(rel):
    return os.path.join(dirname, rel)

# Input data
df= pd.read_csv(path("./../dat/forest_fires.csv"))
print(df.head())
print(df.describe())

# remove every data whose area is 0
#df = df[df.area > 0]

# show description of data that is left
print(df.describe())

# Abandoning features that don't make sense.
drop_elements = ['FFMC', 'DMC', 'DC', 'ISI']
df = df.drop(drop_elements, axis = 1)

# show description of data that is left
print(df.describe())

#
# Split incoming data
#
from sklearn.model_selection import train_test_split
x = df.drop(['area'],axis=1)
y = df['area']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

y = X_train, X_test, y_train, y_test = x_train, x_test, y_train, y_test
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)
reg = MLPRegressor(hidden_layer_sizes=(128,128,128),activation="relu" ,random_state=1, max_iter=20000).fit(X_trainscaled, y_train)
y_pred=reg.predict(X_testscaled)
print("The Score with ", (r2_score(y_pred, y_test)))