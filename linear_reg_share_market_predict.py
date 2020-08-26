import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

dftrain = pd.read_csv("OHLC.csv")
dfeval = pd.read_csv("new.csv")
print(dftrain.dtypes)
y_train = dftrain.pop('High')
y_eval = dfeval.pop('High')
#PrintRowZero
#print(dftrain.loc[0],y_train.loc[0])

#CategoricalData is non numerical data

CATEGORICAL_COLUMNS = ['Symbol','Date','Vol']
NUMERIC_COLUMNS = ['Open','Low','Close']
feature_column = []
dftrain.fillna('', inplace=True)
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() 
    #Assigning unique value to each non numeric keyword
    feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    #Converting all numbers to float 32
    feature_column.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

#Feeding in form of batches for faster computation
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_column)
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data
print(result)
print("<><><><><><><><><><><>")
clear_output()  # clears consoke output
  # the result variable is simply a dict of stats about our model
pred_dicts = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0])

a=0
for i in range(0,len(pred_dicts)):
    print(pred_dicts[i]['predictions']) 
    print(y_eval.loc[i])
    print(pred_dicts[i]['predictions'][0]-y_eval.loc[i])
    print("--------------------------")
    if(pred_dicts[i]['predictions'][0]-y_eval.loc[i]<3 and pred_dicts[i]['predictions'][0]-y_eval.loc[i]>-3):
        a=a+1
  
print((a/len(pred_dicts))*100)
