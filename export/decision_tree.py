
# coding: utf-8

# # Performance validation of Electrical Impedance Tomography Images using Decision Tree Classifier - Machine Learning
# 
# ## Copyright (c) 2018, Faststream Technologies
# ## Author: Sudhanva Narayana

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ### CURR and PARENT directory constants

# In[2]:


CURR_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))


# ### Import dataset ignoring headers

# In[3]:


df = pd.read_csv(PARENT_DIR + '\\assets\\datasets\\eit_data.csv', index_col=[0], header = [0], skiprows= [1] ,skipinitialspace=True)


# ### Dataset

# In[4]:


df.head()


# ### Visualise the higher intensities

# In[5]:


plt.scatter(df['brown'], df['orange'], c=['brown', 'orange'])
plt.xlabel("Brown")
plt.ylabel("Orange")
plt.show()


# ### Importing dataset

# In[6]:


X = df.loc[:, ['gray', 'violet', 'blue', 'green', 'yellow', 'orange', 'red', 'brown']].values.astype(float)
y = df.loc[:, ['target']].values


# ### Splitting the dataset into the Training set and Test set (75%, 25%)

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[8]:


y_train = y_train.ravel()


# ### Feature Scaling

# In[9]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ### Fitting classifier to the Training set

# In[10]:


classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)


# ### Predicting the Test set results

# In[11]:


y_pred = classifier.predict(X_test)

print(classifier.score(X_test, y_test))


# ### The Confusion Matrix

# In[12]:


print(confusion_matrix(y_test, y_pred))


# In[13]:


print(classification_report(y_test, y_pred))

