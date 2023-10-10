#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation

import matplotlib.pyplot as plt 
import seaborn as sns
matplotlib inline


# In[5]:


houseDF= pd.read_csv('Housing.csv')


# In[7]:


houseDF.head()


# In[8]:


houseDF.info()


# In[9]:


houseDF.describe()


# In[10]:


houseDF.columns


# In[11]:


sns.pairplot(houseDF)


# In[12]:


sns.heatmap(houseDF.corr(),annot=True)


# In[54]:


X= houseDF[['price', 'area', 'bedrooms', 'bathrooms', 'stories',  
           
       'parking',    ]]
y = houseDF['price' ]


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.40, random_state=101)


# In[57]:


from sklearn.linear_model import LinearRegression


# In[58]:


lm = LinearRegression()


# In[59]:


lm.fit(X_train, y_train)


# In[65]:


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])


# In[66]:


coeff_df


# In[69]:


predictions = lm.predict(X_test)


# In[70]:


plt.scatter(y_test,predictions)


# In[75]:


sns.displot((y_test-predictions),bins=50);


# In[ ]:




