#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


file = pd.read_csv('weatherAUS.csv')
file


# In[3]:


def alda_regression_linear(X_train, X_test, y_train):

    # Perform linear regression
    # Inputs:
    # X_train: trainning data frame (19 variables, x1-x19)
    # X_test: test data frame(19 variables, x1-x19)
    # y_train: dependent variable, training data (vector, continous type)
  
    # allowed packages: sklearn.linear_model
  
    # Function hints: Read the documentation for the functions LinearRegression (link above)
    
    # write code for building a linear regression model using X_train, y_train
    # YOUR CODE HERE
    a = []
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    a.append(reg)
    a.append(reg.predict(X_test))
    return a


# In[4]:


def alda_regression_lasso(X_train, X_test, y_train, random_state=0):
    # Perform lasso regression
    # Inputs:
    # X_train: training data frame(19 variables, x1-x19)
    # X_test: test data frame(19 variables, x1-x19)
    # y_train: dependent variable, training data (vector, continous type)
    # random_state: a random state to use in CV model training
    # General Information:
        # use 10-fold cross validation to determine the best model hyperparameters
    
    # Output:
    # A tuple containing:
    # - The regression model and 
    # - The list of predictions on test data (X_test) (vector) 
  
    # allowed packages: sklearn.linear_model
  
    # Function hints: Read the documentation for the functions LassoCV (link above)
    
    # write code for lasso regression here
    # 10 fold cross validation
    # set up the random_state as 0 in order for reproducibility
    # YOUR CODE HERE
    a = []
    reg = LassoCV(cv = 10, random_state = 0)
    reg.fit(X_train, y_train)
    a.append(reg)
    a.append(reg.predict(X_test))
    return a


# In[5]:


def evaluation_measures(y_true, y_pred):
    """
    Write a single function to calculate accuracy and a confusion matrix for the given data
    Input:
        y_true: A numpy array containing the actual ground truth labels
        y_pred: A numpy array containing the predicted labels from a model (such as a decision tree or knn classifier)
                y_pred has the same dimensions as y_true
                
    Output:
        A list in the following order:
        overall accuracy score, confusion matrix
    
    Allowed Libraries: sklearn.metrics.confusion_matrix and np methods *only* (no other sklearn metrics)
    
    Hint: Take a look at confusion_matrix method from sklearn.metrics
    """
    # YOUR CODE HERE
    array = confusion_matrix(y_true, y_pred)
    accuracy = 0
    for i in range(0, y_true.size):
        if y_true[i] != y_pred[i]:
            accuracy = accuracy + 1
    accuracy = (y_true.size-accuracy) / y_true.size
    a=[]
    a.append(accuracy)
    a.append(array)
    return a


# In[ ]:





# In[ ]:





# In[6]:


uluru = file.loc[file['Location'] == 'Uluru' ]
brisbane = file.loc[file['Location'] == 'Brisbane' ]
norfolkIsland = file.loc[file['Location'] == 'NorfolkIsland' ]


# In[7]:


uluru


# In[8]:


uluru_total_number = uluru.shape[0]
brisbane_total_number = brisbane.shape[0]
norfolkIsland_total_number = norfolkIsland.shape[0]

print("Uluru: ", uluru_total_number, " samples")
print("Brisbane: ", brisbane_total_number, " samples")
print("Norfolk Island: ", norfolkIsland_total_number, " samples")


# In[ ]:





# In[9]:


uluru.info()


# In[10]:


brisbane.info()


# In[11]:


norfolkIsland.info()


# In[12]:


## Uluru has no data for evaporations and shunshine, therefore, we have to remove them
## We can not handle the direction of wind speed properly
file.drop(['Evaporation', 'Sunshine', 'WindGustDir', 'WindDir9am','WindDir3pm'], inplace = True, axis = 1)
# Drop all data that contains null values
file= file.dropna(axis=0,how="any")


# In[13]:


## Get new values for all them
uluru = file.loc[file['Location'] == 'Uluru' ]
brisbane = file.loc[file['Location'] == 'Brisbane' ]
norfolkIsland = file.loc[file['Location'] == 'NorfolkIsland' ]
total = uluru.append(brisbane).append(norfolkIsland)

uluru_total_number = uluru.shape[0]
brisbane_total_number = brisbane.shape[0]
norfolkIsland_total_number = norfolkIsland.shape[0]

print("Uluru: ", uluru_total_number, " samples")
print("Brisbane: ", brisbane_total_number, " samples")
print("Norfolk Island: ", norfolkIsland_total_number, " samples")

uluru


# In[14]:


## Random selecting training data and testing data, 30% of the data are used to train
arr = uluru.to_numpy()
np.random.shuffle(arr)
uluru_x_train = arr[0:uluru_total_number//3,2:-1]
uluru_test = arr[uluru_total_number//3:len(arr),2:-1]
uluru_y_train = arr[0:uluru_total_number//3,-1].reshape(uluru_total_number//3,1)

arr = brisbane.to_numpy()
np.random.shuffle(arr)
brisbane_x_train = arr[0:brisbane_total_number//3,2:-1]
brisbane_test = arr[brisbane_total_number//3:len(arr),2:-1]
brisbane_y_train = arr[0:brisbane_total_number//3,-1].reshape(brisbane_total_number//3,1)

arr = norfolkIsland.to_numpy()
np.random.shuffle(arr)
norfolk_x_train = arr[0:norfolkIsland_total_number//3,2:-1]
norfolk_test = arr[norfolkIsland_total_number//3:len(arr),2:-1]
norfolk_y_train = arr[0:norfolkIsland_total_number//3,-1].reshape(norfolkIsland_total_number//3,1)


# In[15]:


uluru_lasso_model, uluru_lasso_regression_result = alda_regression_lasso(uluru_x_train, uluru_test, uluru_y_train)
print(f'Lasso Regression Model coefficients for Uluru:\n{uluru_lasso_model.coef_}')

brisbane_lasso_model, brisbane_lasso_regression_result = alda_regression_lasso(brisbane_x_train, brisbane_test, brisbane_y_train)
print(f'Lasso Regression Model coefficients for Brisbane:\n{brisbane_lasso_model.coef_}')

norfolk_lasso_model, norfolk_lasso_regression_result = alda_regression_lasso(norfolk_x_train, norfolk_test, norfolk_y_train)
print(f'Lasso Regression Model coefficients for Norfolk Island:\n{norfolk_lasso_model.coef_}')


# In[16]:


## Lasso eliminates too much componenets, therefore, we tried to do linear regression to select features
uluru_linear_model, uluru_linear_regression_result = alda_regression_linear(uluru_x_train, uluru_test, uluru_y_train)
print(f'Linear Regression Model coefficients for Uluru:\n{uluru_linear_model.coef_}')

brisbane_linear_model, brisbane_linear_regression_result = alda_regression_linear(brisbane_x_train, brisbane_test, brisbane_y_train)
print(f'Model coefficients for Brisbane:\n{brisbane_linear_model.coef_}')

norfolk_linear_model, norfolk_linear_regression_result = alda_regression_linear(norfolk_x_train, norfolk_test, norfolk_y_train)
print(f'Model coefficients for Norfolk Island:\n{norfolk_linear_model.coef_}')


# In[ ]:





# In[17]:


features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm','RainToday']
features_data = total.loc[:,features].values
actual_data = total.loc[:,['RainTomorrow']].values

uluru_features = uluru.loc[:,features].values
uluru_actual = uluru.loc[:,['RainTomorrow']].values

brisbane_features = brisbane.loc[:,features].values
brisbane_actual = brisbane.loc[:,['RainTomorrow']].values

norfolk_features = norfolkIsland.loc[:,features].values
norfolk_actual = norfolkIsland.loc[:,['RainTomorrow']].values
## Standardlize the data
features_data = StandardScaler().fit_transform(features_data)


# In[18]:


pca = PCA(n_components = 15)
pc = pca.fit_transform(features_data)

eigenvalue = pca.explained_variance_ratio_
eigenvalue


# In[19]:


plt.scatter(range(1,16),eigenvalue)
plt.plot(range(1,16),eigenvalue)
plt.xlabel('Features')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[20]:


## From the plot that we know, the fourth feature has a variance lower than 10%. Therefore, we are going to select three features to make the prediction.
pca = PCA(n_components=3)
pc = pca.fit_transform(features_data)


# In[21]:


uluru_fit = RFE(uluru_linear_model, 3).fit(uluru_features,uluru_actual)
print("Num Features: %s" % (uluru_fit.n_features_))
print("Selected Features: %s" % (uluru_fit.support_))
print("Feature Ranking: %s" % (uluru_fit.ranking_))


# In[22]:


brisbane_fit = RFE(brisbane_linear_model, 3).fit(brisbane_features,brisbane_actual)
print("Num Features: %s" % (brisbane_fit.n_features_))
print("Selected Features: %s" % (brisbane_fit.support_))
print("Feature Ranking: %s" % (brisbane_fit.ranking_))


# In[23]:


norfolk_fit = RFE(norfolk_linear_model, 3).fit(norfolk_features,norfolk_actual)
print("Num Features: %s" % (norfolk_fit.n_features_))
print("Selected Features: %s" % (norfolk_fit.support_))
print("Feature Ranking: %s" % (norfolk_fit.ranking_))


# In[24]:


#Therefore, from the feature ranking that we knew, three of the top features that affect the prediction the most are:
#Uluru: Pressure3pm, Temp3pm, and RainToday
#Brisbane: Humidity3pm, Cloud3pm, and RainToday
#Norfolk Island: Pressure9am, Pressure3pm, and RainToday
#Create new dataframe for these three attributes of different location
uluru_features_value=['Pressure3pm', 'Temp3pm', 'RainToday']
brisbane_features_value = ['Humidity3pm', 'Cloud3pm', 'RainToday']
norfolk_features_value = ['Pressure9am', 'Pressure3pm', 'RainToday']

uluru_features = uluru.loc[:,uluru_features_value].values
uluru_actual = uluru.loc[:,['RainTomorrow']].values

brisbane_features = brisbane.loc[:,brisbane_features_value].values
brisbane_actual = brisbane.loc[:,['RainTomorrow']].values

norfolk_features = norfolkIsland.loc[:,norfolk_features_value].values
norfolk_actual = norfolkIsland.loc[:,['RainTomorrow']].values

uluru_features=np.asarray(uluru_features,dtype=np.float64)
uluru_actual=np.asarray(uluru_actual,dtype=np.float64)

brisbane_features=np.asarray(uluru_features,dtype=np.float64)
brisbane_actual=np.asarray(uluru_actual,dtype=np.float64)

norfolk_features=np.asarray(uluru_features,dtype=np.float64)
norfolk_actual=np.asarray(uluru_actual,dtype=np.float64)


# In[25]:


# split into train and test in a stratified manner
uluru_train, uluru_test, uluru_labels_train, uluru_labels_test = train_test_split(uluru_features,uluru_actual,test_size = 0.33)

# scale the train and test datasets
uluru_scaler = MinMaxScaler().fit(uluru_train)
uluru_train = uluru_scaler.transform(uluru_train)
uluru_test = uluru_scaler.transform(uluru_test)

# apply DecisionTreeClassifier on the dataset
# in this case, we're using the gini index to split the data
# We only have 3 attributes, therefore, the max-depth of the decision tree should be 3
uluru_decision_tree = DecisionTreeClassifier(criterion='gini', max_depth = 3)
# now, train the model
uluru_decision_tree.fit(X=uluru_train, y=uluru_labels_train)
# predict on the test set
uluru_predictions = uluru_decision_tree.predict(X=uluru_test)

# evaluate your model and print the accuracy 
uluru_evaluations = evaluation_measures(y_true=uluru_labels_test, y_pred=uluru_predictions)

# print the evaluation of the model
print(f'Accuracy of your model is {uluru_evaluations[0]},\nConfusion matrix is:\n {uluru_evaluations[1]}')

# print the tree itself, using the sklearn.tree.plot_tree function
plt.figure(figsize=(20,20))
plot_tree(uluru_decision_tree)


# In[26]:


brisbane_train, brisbane_test, brisbane_labels_train, brisbane_labels_test = train_test_split(brisbane_features,brisbane_actual,test_size = 0.33)


brisbane_scaler = MinMaxScaler().fit(brisbane_train)
brisbane_train = brisbane_scaler.transform(brisbane_train)
brisbane_test = brisbane_scaler.transform(brisbane_test)

brisbane_decision_tree = DecisionTreeClassifier(criterion='gini', max_depth = 3)
brisbane_decision_tree.fit(X=brisbane_train, y=brisbane_labels_train)
brisbane_predictions = brisbane_decision_tree.predict(X=brisbane_test)
brisbane_evaluations = evaluation_measures(y_true=brisbane_labels_test, y_pred=brisbane_predictions)
print(f'Accuracy of your model is {brisbane_evaluations[0]},\nConfusion matrix is:\n {brisbane_evaluations[1]}')
plt.figure(figsize=(20,20))
plot_tree(brisbane_decision_tree)


# In[27]:


norfolk_train, norfolk_test, norfolk_labels_train, norfolk_labels_test = train_test_split(norfolk_features,norfolk_actual,test_size = 0.33)


norfolk_scaler = MinMaxScaler().fit(norfolk_train)
norfolk_train = norfolk_scaler.transform(norfolk_train)
norfolk_test = norfolk_scaler.transform(norfolk_test)

norfolk_decision_tree = DecisionTreeClassifier(criterion='gini', max_depth = 3)
norfolk_decision_tree.fit(X=norfolk_train, y=norfolk_labels_train)
norfolk_predictions = norfolk_decision_tree.predict(X=norfolk_test)
norfolk_evaluations = evaluation_measures(y_true=norfolk_labels_test, y_pred=norfolk_predictions)
print(f'Accuracy of your model is {norfolk_evaluations[0]},\nConfusion matrix is:\n {norfolk_evaluations[1]}')
plt.figure(figsize=(20,20))
plot_tree(norfolk_decision_tree)


# In[28]:


decision_tree_average = (norfolk_evaluations[0]+brisbane_evaluations[0]+uluru_evaluations[0])/3
print(f'In conclusion, the overall accuracy of using a decision tree model is {decision_tree_average}')


# In[29]:


# Now, let's try with K-NN approaching
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(uluru_train,uluru_labels_train)
Y_pred2=knn.predict(uluru_test)
uluru_accuracy = accuracy_score(uluru_labels_test,Y_pred2)
uluru_accuracy


# In[30]:


knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(brisbane_train,brisbane_labels_train)
Y_pred2=knn.predict(brisbane_test)
brisbane_accuracy = accuracy_score(brisbane_labels_test,Y_pred2)
brisbane_accuracy


# In[31]:


knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(norfolk_train,norfolk_labels_train)
Y_pred2=knn.predict(norfolk_test)
norfolk_accuracy = accuracy_score(norfolk_labels_test,Y_pred2)
norfolk_accuracy


# In[32]:


average = (uluru_accuracy+brisbane_accuracy+norfolk_accuracy)/3
print(f'In conclusion, the overall accuracy of using a KNN model is {average}')


# In[ ]:





# In[ ]:




