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


# ### Load House Dataset

# In[2]:


df_house = pd.read_csv(r'E:\Dataset\Review_4_Ajarn\house_data_V.3_setTarget.txt', delimiter='\t')


# ### Clean Data --House_Dataset

# In[3]:


df_house


# #### Drop Columns with Missing Values

# In[4]:


df_house_droped = df_house.dropna(axis='columns')
df_house_droped


# #### Change String to Numeric Values

# In[5]:


from sklearn.preprocessing import LabelEncoder

# encoding string values into numeric values
le_propTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_userTy = LabelEncoder()
le_homeCon = LabelEncoder()
le_roadTy = LabelEncoder()


# In[6]:


# create new columns containing numeric code of former column
df_house_droped['PropertyType_n'] = le_propTy.fit_transform(df_house_droped['PropertyType'])
df_house_droped['AsseStatus_n'] = le_asseSt.fit_transform(df_house_droped['AsseStatus'])
df_house_droped['UserType_n'] = le_userTy.fit_transform(df_house_droped['UserType'])
df_house_droped['HomeCondition_n'] = le_homeCon.fit_transform(df_house_droped['HomeCondition'])
df_house_droped['RoadType_n'] = le_roadTy.fit_transform(df_house_droped['RoadType'])
df_house_droped


# ### Select Columns that are used as Variables in Model

# In[7]:


df_house_select = df_house_droped[['PropertyType_n', 'SellPrice', 'CostestimateB','MarketPrice', 'HouseArea', 'Floor', 'HomeCondition_n', 'BuildingAge','RoadType_n','AsseStatus_n', 'UserType_n']]
df_house_select


# #### Export Cleaned Dataset

# In[8]:


df_export = df_house_select.to_csv(r'E:\Dataset\df_house_cleaned_export.txt', index=False)


# #### Prepare Data & Target Value for Model Training 

# In[9]:


# seperate train data and target data
df_train_house = df_house_select.drop('UserType_n', axis='columns')
target_house = df_house_select['UserType_n']


# In[10]:


df_train_house


# In[11]:


target_house


# ### Decision Tree Clasifier (House)

# In[12]:


from sklearn.tree import DecisionTreeClassifier


# #### Data Slicing
# ##### Split Dataset into Train and Test

# In[13]:


from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
XH = df_train_house
YH = target_house
XH_train, XH_test, YH_train, YH_test = train_test_split(XH, YH, test_size=0.3)


# In[14]:


XH_train.shape, YH_train.shape


# In[15]:


XH_test.shape, YH_test.shape


# #### Train Dataset
# ##### Gini Index = a metric to measure how often a randomly chosen element would be incorrectly identified. It means an attribute with lower gini index should be preferred.

# In[16]:


# Classifier Object
clf_giniHouse = DecisionTreeClassifier(max_features="auto", max_leaf_nodes=10)
clf_giniHouse = DecisionTreeClassifier(max_features="auto", max_leaf_nodes=11)
clf_giniHouse = DecisionTreeClassifier(max_leaf_nodes=12)


# In[17]:


clf_giniHouse = DecisionTreeClassifier(max_features="auto", max_leaf_nodes=10)
clf_giniHouse = DecisionTreeClassifier(max_features="auto", max_leaf_nodes=11)
clf_giniHouse = DecisionTreeClassifier(max_leaf_nodes=12)


# In[18]:


# Perform Training 
clf_giniHouse.fit(XH_train, YH_train)
clf_giniHouse


# ##### View Decision Tree Model

# In[19]:


from sklearn import tree


# In[20]:


view_model_tree = tree.plot_tree(clf_giniHouse.fit(XH_train, YH_train))


# #### Show & Export Graph

# In[21]:


import graphviz 
dot_data = tree.export_graphviz(clf_giniHouse, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("house")


# In[22]:


dot_data = tree.export_graphviz(clf_giniHouse, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph


# #### Prediction & Accuracy
# ##### Prediction using Gini

# In[23]:


# Predicton on test with giniIndex 
YHgini_pred = clf_giniHouse.predict(XH_test) 
print("Predicted values:") 
YHgini_pred


# #### Calculate Accuracy

# In[24]:


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

# In[25]:


print("Confusion Matrix: ", confusion_matrix(YH_test, YHgini_pred))
# Confusion Matrix = a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. 
#                           It shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.
#                                                        Predicted: No    Predicted: Yes
#                               Actual: No         [      TN                  FP           ]
#                               Actual: Yes        [      FN                  TP           ]

print ("Accuracy : ", accuracy_score(YH_test, YHgini_pred)*100)
# Accuracy = (TP + TN) / (TP + TN + FP + FN); 
# True Positive (TP) : Observation is positive, and is predicted to be positive. 
# False Negative (FN) : Observation is positive, but is predicted negative. 
# True Negative (TN) : Observation is negative, and is predicted to be negative. 
# False Positive (FP) : Observation is negative, but is predicted positive.

print("Report : ", classification_report(YH_test, YHgini_pred))
# Precision = TP/ (TP + TN) High Precision indicates an example labelled as positive is indeed positive (a small number of FP). 
# Recall = TP / (TP + FN)  High Recall indicates the class is correctly recognized (a small number of FN). 
## f1-score = (2*Recall*Precision) / (Recall + Precision) Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. 
##                We calculate an F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more. The F-Measure will always be nearer to the smaller value of Precision or Recall.
## High recall, low precision: This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives. 
## Low recall, high precision: This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP) 
## Support = amount of elements/target (in this case the amount of UserType)


# #### Export Model with Pickle
# ##### House Model

# In[26]:


import pickle 


# In[27]:


#with open(r'E:\Model\Model_Pickle_giniHouse_V02', 'wb') as mlA:
#    pickle.dump(clf_giniHouse,mlA)


# ## Analysis

# In[28]:


#class distribution
print(df_house_droped.groupby('UserType').size())


# In[29]:


# descriptions
print(df_house_select.describe())

