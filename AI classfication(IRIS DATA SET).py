#!/usr/bin/env python
# coding: utf-8

# In[17]:


#IRIS DATA SETS / Classfication .....

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc

coL_nam=["SepalLength" ,"SepalWidth" , "PetalLength","PetalWidth" ,"Species" ]
SPECIES=["Setosa" ,"Versicolor",'Virginica']


# In[18]:


trn_path=tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")

test_path=tf.keras.utils.get_file('iris_test.csv','https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')


# In[19]:


train=pd.read_csv(trn_path,names=coL_nam , header=0)
test=pd.read_csv(test_path,names=coL_nam,header=0)


# In[20]:


train.head()


# In[21]:


train.shape


# In[22]:


test.shape


# In[23]:


test.head()


# In[24]:


train_y=train.pop('Species')
test_y=test.pop('Species')


# In[25]:


train.head()


# In[26]:


#no epochs here......

def input_function(features,dflables, Training=True, batch_size=256):
        ds=tf.data.Dataset.from_tensor_slices((dict(features),dflables))
        if Training:
            ds=ds.shuffle(1000).repeat()
        return ds.batch(batch_size)


# In[27]:


Feat_col=[]
for key in train.keys():
    Feat_col.append(tf.feature_column.numeric_column(key=key))


# In[28]:


classifeir=tf.estimator.DNNClassifier(
feature_columns=Feat_col,
    hidden_units=[30,10],
    n_classes=3
)
classifeir.train(
input_fn=lambda: input_function(train,train_y,Training=True),steps=5000
)


# In[29]:


eval_res=classifeir.evaluate(input_fn=lambda:input_function(test,test_y,Training=False))
print('\nTest set accuracy:{accuracy:0.3f}\n'.format(**eval_res))


# In[35]:


def input_function(features,b_s=256):
    return tf.data.Dataset.from_tensor_slices((dict(features))).batch(b_s)


# In[40]:



feat = ['SepalLength', 'SepalWidth', 'PetalLength','PetalWidth']
                                    
predict={}
print("ENTER data of flower:")
for feature in feat:
                                        
    valid=True
    while valid:                                          
      val=input(feature + ":") 
      if not val.isdigit():valid=False                 
    predict[feature]=[float(val)]                                 
predictions=classifeir.predict(input_fn=lambda: input_function(predict))
for pred_dict in predictions:
    print(pred_dict)
    class_id=pred_dict['class_ids'][0]
    probability=pred_dict['probabilities'][class_id] 
    print('Prediciton is "{}"({:.1f}%)'.format(SPECIES[class_id],100*probability))


# In[ ]:




