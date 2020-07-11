#!/usr/bin/env python
# coding: utf-8

# In[123]:


import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt 
import seaborn as sn                   # For plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")


# In[124]:


# Loading the train and test data 
train = pd.read_csv("/home/neo/Task_Intershala/Bank_task/train.csv")
test = pd.read_csv("/home/neo/Task_Intershala/Bank_task/test.csv")


# In[125]:


train.shape , test.shape


# In[126]:


train.columns


# In[127]:


train.head()


# In[128]:


train.info()


# # Univarient analysys

# In[129]:


# checking the distribution of thr target variable
train['subscribed'].value_counts()


# In[130]:


train['subscribed'].value_counts(normalize=True)


# In[131]:


train['subscribed'].value_counts().plot.bar()


# In[132]:


# checking the distribution of age
train['age'].plot.hist()


# In[133]:


sn.distplot(train["age"])


# In[134]:


# checking the distribution of job
train['job'].value_counts().plot.bar()


# In[135]:


train['default'].value_counts().plot.bar()


# In[136]:


train['education'].value_counts().plot.bar()


# In[137]:


train['housing'].value_counts().plot.bar()


# # Bivarient Analysys

# In[138]:


# we will explore the relation of the dependent variable with other independent variables
print(pd.crosstab(train['job'],train['subscribed']))


# In[139]:


job = pd.crosstab(train['job'],train['subscribed'])
job.div(job.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('Job')
plt.ylabel('Percentage')


# In[140]:


print(pd.crosstab(train['education'],train['subscribed']))
job = pd.crosstab(train['education'],train['subscribed'])
job.div(job.sum(1).astype(float),axis = 0).plot(kind="bar", stacked=True, figsize=(10,10))
plt.xlabel('education')
plt.ylabel('Percentage')


# In[141]:


print(pd.crosstab(train['marital'],train['subscribed']))
job = pd.crosstab(train['marital'],train['subscribed'])
job.div(job.sum(1).astype(float),axis = 0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('marital')
plt.ylabel('Percentage')


# In[142]:


print(pd.crosstab(train['default'],train['subscribed']))
job = pd.crosstab(train['default'],train['subscribed'])
job.div(job.sum(1).astype(float),axis = 0).plot(kind="bar", stacked=True, figsize=(8,8))
plt.xlabel('default')
plt.ylabel('Percentage')


# In[143]:


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# In[144]:


train.corr()


# In[145]:


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# # Model

# In[146]:


target = train['subscribed']
train = train.drop('subscribed',1)


# In[147]:


train.head()


# In[148]:


# to apply logistic model we have to convert all catagorical values to numbers
train = pd.get_dummies(train)


# In[149]:


train.head()


# In[150]:


train.columns


# In[151]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[152]:


X_train,X_val,y_train,y_val = train_test_split(train,target,test_size = 0.2, random_state=12)


# In[153]:


log_reg = LogisticRegression()


# In[154]:


log_reg.fit(X_train,y_train)


# In[155]:


log_pred = log_reg.predict(X_val)


# In[156]:


# for logistic regression we use accuracy 
from sklearn.metrics import accuracy_score


# In[157]:


accuracy_score(y_val,log_pred)


# # Check the score with Dicision tree

# In[158]:


from sklearn.tree import DecisionTreeClassifier


# In[159]:


tree = DecisionTreeClassifier(max_depth=5, random_state=0)


# In[160]:


tree.fit(X_train,y_train)


# In[161]:


tree_pred = tree.predict(X_val)


# In[162]:


accuracy_score(y_val,tree_pred)


# We can see the Dicision tree has greater accuracy hence we can use this for our test data

# In[163]:


test = pd.get_dummies(test)


# In[164]:


test_pred = tree.predict(test)


# In[168]:


test_pred


# In[172]:


submission_final = pd.DataFrame()
submission_final['ID'] = test['ID']
submission_final['subscribed'] = test_pred


# In[174]:


# all the result is in 0 & 1 hence convert them to yes and no as initial format
submission_final['subscribed'].replace(0,'no',inplace=True)
submission_final['subscribed'].replace(1,'yes',inplace=True)


# In[175]:



submission_final.to_csv('submission_final.csv', header=True, index=False)


# In[ ]:




