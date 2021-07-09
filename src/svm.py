import pandas as pd
import numpy as np

# path operations
import os
dirname = os.path.dirname(__file__)
def path(rel):
    return os.path.join(dirname, rel)

# Input data
df= pd.read_csv(path("./../dat/forest_fires.csv"))
print(df.head())
print("Original: Size of CSV")
print(df.size)

# Remove Date columns
df.drop(["month","day"],axis=1,inplace =True)
print("Date removed: Size of CSV")
print(df.size)

# Normalising the data as there is scale difference
x = df.drop(['area'],axis=1)
y = df['area']


def norm_func(i):
    norm = (i-i.min())/(i.max()-i.min())
    return (norm)

norm_x = norm_func(x)

def round_func(i):
    return (np.ceil(i))

round_y = round_func(y)

# Split incoming data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, round_y, test_size = 0.15)

# Creat Model
from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)
print("linear SVC resulst:")
print(np.mean(pred_test_linear==y_test)) 

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)
print("poly SVC resulst in %")
print(np.mean(pred_test_poly==y_test)) 

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)
print("rbf SVC resulst in %")
print(np.mean(pred_test_rbf==y_test)) 

# 'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)
print("sigmoid SVC resulst in %")
print(np.mean(pred_test_sig==y_test)) 

