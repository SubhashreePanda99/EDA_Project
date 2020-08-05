#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats


# In[10]:


data=sns.load_dataset('iris')


# In[11]:


data.head()


# In[35]:


data.size


# In[6]:


data['sepal_length'].mean()


# In[10]:


data['sepal_length'].median()


# In[11]:


data['sepal_length'].mode()


# In[7]:


data['sepal_width'].mean()


# In[8]:


data['petal_length'].mean()


# In[9]:


data['petal_width'].mean()


# In[13]:


range_=max(data['sepal_length'])-min(data['sepal_length'])
range_


# In[14]:


q2=data['sepal_length'].quantile(0.50)
q2


# In[18]:


q1=data['sepal_length'].quantile(0.25)
q1


# In[19]:


q3=data['sepal_length'].quantile(0.75)
q3


# In[27]:


st=data['sepal_length'].std()
st


# In[22]:


si=data['sepal_length'].var()
si


# In[23]:


mn=data['sepal_length'].mean()


# In[24]:


mn


# In[30]:


coefficient_of_variation=st/mn*100


# In[31]:


coefficient_of_variation


# In[41]:


co=data.corr()
co


# In[32]:


from scipy.stats import zscore


# In[33]:


zscore(data['sepal_length'])


# In[34]:


sns.pairplot(data)
plt.show()


# In[36]:


sns.countplot(data['sepal_length'])
plt.show()


# In[38]:


sns.boxplot(data['sepal_length'])
plt.plot()


# In[39]:


plt.scatter(data['sepal_length'],data['sepal_width'])
plt.show()


# In[42]:


sns.heatmap(co)
plt.show()


# In[44]:


data.isnull().sum()


# In[49]:


sns.violinplot(data['sepal_length'])
plt.show()


# In[6]:


plt.hist(data['sepal_length'],bins=4)
plt.show()


# In[8]:


data.hist()


# In[52]:


sns.violinplot(x='petal_length',y='petal_length',hue='species',data=data)


# In[4]:


dt=pd.read_csv('titanictrain.csv')
dt


# In[6]:


dt.size


# In[8]:


a=pd.get_dummies(dt['Sex'])
a.head()#we have to add the columns


# In[13]:


pd.get_dummies(dt,columns=['Sex']).head()# it direct add the columns to the dataframe skipping the original one


# In[74]:


pd.merge(dt,a,on=dt.index)# adding the columns by merge method


# In[14]:


import sklearn
from sklearn.preprocessing import OneHotEncoder
hotencoder=OneHotEncoder()
encoded=hotencoder.fit_transform(dt.Sex.values.reshape(-1,1)).toarray()
encoded


# In[69]:


aa=pd.get_dummies(data,prefix='park',columns=['species'])
aa.head()


# In[75]:


from sklearn.preprocessing import StandardScaler


# In[76]:


std_scale=StandardScaler()
std_scale


# In[81]:


dt['age_stdscl']=std_scale.fit_transform(dt[['Age']])# it takes


# In[82]:


dt['age_stdscl']


# In[86]:


from sklearn.preprocessing import MinMaxScaler


# In[88]:


minmax_scale=MinMaxScaler()
minmax_scale


# In[89]:


dt['t']=minmax_scale.fit_transform(dt[['Age']])# it takes only numerical value


# In[90]:


dt['t']


# In[ ]:




