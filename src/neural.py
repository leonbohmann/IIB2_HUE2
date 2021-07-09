from keras.models import Sequential
import keras.optimizers as opti
from keras.layers import Dense, Activation,Dropout
import pandas as pd
import numpy as np

model = Sequential()
model.add(Dense(100, input_dim=12))
model.add(Activation('selu'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Activation('selu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(1))
model.summary()

learning_rate=0.001
optimizer = opti.RMSprop(lr=learning_rate)
model.compile(optimizer=optimizer,loss='mse', metrics=['accuracy'])

# path operations
import os
dirname = os.path.dirname(__file__)
def path(rel):
    return os.path.join(dirname, rel)

# Input data
df= pd.read_csv(path("./../dat/forest_fires.csv"))
print(df.head())
print(df.describe())

#
# Split incoming data
#
from sklearn.model_selection import train_test_split
x = df.drop(['area'],axis=1)
y = df['area']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

data=x_train
target = y_train
model.fit(data, target, epochs=100, batch_size=10,verbose=1, validation_data=(x_test,y_test))

a=model.predict(x_test)
print("RMSE for Deep Network:",np.sqrt(np.mean((y_test-a.reshape(a.size,))**2)))