import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

dftrain = pd.read_csv("weatherdata_bkt.csv")
dfeval = pd.read_csv("weathercheck.csv")
print(dftrain.dtypes)
y_train = dftrain.pop('Temperature')
y_eval = dfeval.pop('Temperature')


CATEGORICAL_COLUMNS = ['variable']
NUMERIC_COLUMNS = ['sea_pressure','Precipitation','Cloud','Sunshine','Radiation','Wind','WDirection']
feature_column = []
dftrain.fillna('', inplace=True)
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() 
    feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_column.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  
    if shuffle:
      ds = ds.shuffle(1000)  
    ds = ds.batch(batch_size).repeat(num_epochs)  
    return ds 
  return input_function  

train_input_fn = make_input_fn(dftrain, y_train)  
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_column)
linear_est.train(train_input_fn)  
result = linear_est.evaluate(eval_input_fn)  
print(result)
print("<><><><><><><><><><><>")
clear_output() 
 
pred_dicts = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0])

a=0
for i in range(0,len(pred_dicts)):
    print(pred_dicts[i]['predictions']) 
    print(y_eval.loc[i])
    print(pred_dicts[i]['predictions'][0]-y_eval.loc[i])
    print("--------------------------")
    if(pred_dicts[i]['predictions'][0]-y_eval.loc[i]<0.6 and pred_dicts[i]['predictions'][0]-y_eval.loc[i]>-0.6):
        a=a+1
  
print((a/len(pred_dicts))*100)
