
# coding: utf-8

# # Performance validation of Electrical Impedance Tomography Images using K-Nearest Neighbors Classifier - Machine Learning
# 
# ## Copyright (c) 2018, Faststream Technologies
# ## Author: Sudhanva Narayana

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
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
y = y.ravel()


# ### Splitting the dataset into the Training set and Test set (75%, 25%)

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# ### Feature Scaling

# In[8]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ### Fitting classifier to the Training set

# In[9]:


classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier.fit(X_train, y_train)


# ### Predicting the Test set results

# In[10]:


y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test))


# ### The Confusion Matrix

# In[11]:


print(confusion_matrix(y_test, y_pred))


# In[12]:


print(classification_report(y_test, y_pred))


# ### Visualising different values of K for Neighbours and Overfitting/Underfitting

# In[13]:


# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

