#!/usr/bin/env python
# coding: utf-8

# # Environment conda3--python3
# ## Coding UTF-8
# ### Import Libraries

# In[1]:


# turn off feature warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# show library versions for documentation reference
import sys
print("Python: {}".format(sys.version))
print("pandas: {}".format(pd.__version__))
print("numpy: {}".format(np.__version__))


# In[2]:


df_land = pd.read_csv(r'E:\Dataset\Review_4_Ajarn\land_data_V.3_withtarget.txt', delimiter='\t')


# ### Clean Data --Land_Dataset

# In[3]:


# show example data from loaded file
df_land


# #### Drop Columns with missing Values

# In[4]:


df_land_droped = df_land.dropna(axis='columns')
df_land_droped


# #### Change String to Numeric Value

# In[5]:


from sklearn.preprocessing import LabelEncoder

# encoding string values into numeric values
le_colorTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_userTy = LabelEncoder()
le_roadTy = LabelEncoder()
le_groundLev = LabelEncoder()


# In[6]:


# create new columns containing numeric code of former column
df_land_droped['ColorType_n'] = le_colorTy.fit_transform(df_land_droped['ColorType'])
df_land_droped['AsseStatus_n'] = le_asseSt.fit_transform(df_land_droped['AsseStatus'])
df_land_droped['UserType_n'] = le_userTy.fit_transform(df_land_droped['UserType'])
df_land_droped['RoadType_n'] = le_roadTy.fit_transform(df_land_droped['RoadType'])
df_land_droped['GroundLevel_n'] = le_groundLev.fit_transform(df_land_droped['GroundLevel'])
df_land_droped


# ### Select Columns that are used as Variables in Model

# In[7]:


df_land_select = df_land_droped[['ColorType_n', 'CostestimateB', 'SellPrice', 'MarketPrice',  'RoadType_n', 'AsseStatus_n', 'UserType_n']]
df_land_select


# #### Export Cleaned Dataset

# In[8]:


df_export = df_land_select.to_csv(r'E:\Dataset\df_land_cleaned_export.txt', index=False)


# ### Prepare Data & Target Value for Model Training 

# In[9]:


# seperate train data and target data
df_train_land = df_land_select.drop('UserType_n', axis='columns')
target_land = df_land_select['UserType_n']


# In[10]:


df_train_land


# In[11]:


target_land


# ### Decision Tree Clasifier (Land)

# In[12]:


from sklearn.tree import DecisionTreeClassifier


# #### Data Slicing
# ##### Split Dataset int Train and Test

# In[13]:


from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
XL = df_train_land
YL = target_land
XL_train, XL_test, YL_train, YL_test = train_test_split(XL, YL, test_size=0.3)


# In[14]:


XL_train.shape, YL_train.shape


# In[15]:


XL_test.shape, YL_test.shape


# #### Train Dataset
# ##### Gini Index = a metric to measure how often a randomly chosen element would be incorrectly identified. It means an attribute with lower gini index should be preferred.

# In[16]:


# Decision Tree with Gini Index
clf_giniLand = DecisionTreeClassifier(min_samples_leaf=3) 


# In[17]:


# Perform Training 
clf_giniLand.fit(XL_train, YL_train) 
clf_giniLand


# ##### View Decision Tree Model

# In[18]:


from sklearn import tree


# In[19]:


view_model_tree = tree.plot_tree(clf_giniLand.fit(XL_train, YL_train))


# #### Show & Export Graph

# In[20]:


import graphviz 
dot_data = tree.export_graphviz(clf_giniLand, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("land")


# In[21]:


dot_data = tree.export_graphviz(clf_giniLand, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph


# #### Prediction & Accuracy
# ##### Prediction using Gini or Entropy

# In[22]:


# Predicton on test with giniIndex 
YLgini_pred = clf_giniLand.predict(XL_test) 
print("Predicted values:") 
YLgini_pred


# #### Calculate Accuracy

# In[23]:


from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# ###### Confusion Matrix = a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. It shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.
# ###### Accuracy = (TP + TN) / (TP + TN + FP + FN); True Positive (TP) : Observation is positive, and is predicted to be positive. False Negative (FN) : Observation is positive, but is predicted negative. True Negative (TN) : Observation is negative, and is predicted to be negative. False Positive (FP) : Observation is negative, but is predicted positive.
# ###### Precision = TP / (TP + TN) High Precision indicates an example labelled as positive is indeed positive (a small number of FP). Recall = TP / (TP + FN)  High Recall indicates the class is correctly recognized (a small number of FN). 
# ###### f1-score = (2*Recall*Precision) / (Recall + Precision) Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. We calculate an F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more. The F-Measure will always be nearer to the smaller value of Precision or Recall.
# ###### High recall, low precision: This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives. 
# ###### Low recall, high precision: This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP) 
# ###### Support = amount of elements/target (in this case the amount of UserType)

# ### Gini Accuracy

# In[24]:


print("Confusion Matrix: ", confusion_matrix(YL_test, YLgini_pred)) 
print ("Accuracy : ", accuracy_score(YL_test, YLgini_pred)*100) 
print("Report : ", classification_report(YL_test, YLgini_pred)) 


# #### Export Model with Pickle
# ##### Land Model

# In[25]:


import pickle 


# In[26]:


#with open(r'E:\Model\Model_Pickle_giniLand_V02', 'wb') as mlA:
#    pickle.dump(clf_giniLand,mlA)


# ## Analysis

# In[27]:


#class distribution
print(df_land_droped.groupby('UserType').size())


# In[28]:


# descriptions
print(df_land_select.describe())


# In[ ]:




