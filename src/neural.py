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

# apply a learning rate to the model compiler
optim = opti.RMSprop(lr=0.001)
model.compile(optimizer=optim,loss='mse', metrics=['accuracy'])

# path operations
import os
dirname = os.path.dirname(__file__)
def path(rel):
    return os.path.join(dirname, rel)

# Input data
df= pd.read_csv(path("./../dat/forest_fires.csv"))
print(df.head())
print(df.describe())
df['area']=np.log10(df['area']+1)

#
# Split incoming data
#
from sklearn.model_selection import train_test_split
x = df.drop(['area'],axis=1)
y = df['area']
# take 20% of data for testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)

# fit model with available data
model.fit(x_train, y_train, epochs=100, batch_size=10,verbose=1, validation_data=(x_test,y_test))

# test the model
nn_result = model.predict(x_test)
print("RMSE for Deep Network:",np.sqrt(np.mean((y_test-nn_result.reshape(nn_result.size,))**2)))