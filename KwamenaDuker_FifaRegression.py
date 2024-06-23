#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('/Users/kwamenaduker/AI/male_players (legacy).csv', na_values = ['-',' ','']) # to remove the dashes and empty cells
df


# In[3]:


df.info()


# In[4]:


df.columns.tolist()


# In[5]:


# Question 1
# Demonstrate the data preparation & feature extraction process 


# In[6]:


# Removing useless varaiables
# we want to predict predict a player's overall rating
# for this we drop the features we don't need by eye test
df.drop(columns=['player_url','real_face','fifa_version','fifa_update','fifa_update_date','player_face_url'], inplace = True)


# In[7]:


df


# In[8]:


# We don't use any feature that has 30% or more missing values
L = []
L_less = []
for i in df.columns:
    if((df[i].isnull().sum())<(0.4*(df.shape[0]))):
        L.append(i)
    else:
        L_less.append(i)


# In[9]:


L


# In[10]:


L_less


# In[11]:


df = df[L]


# In[12]:


df


# In[13]:


df.dtypes.tolist() # to see the different datatypes in the dataframe


# In[14]:


# Separating the categorical and quantitative variables
numeric_data = df.select_dtypes(include = np.number)
non_numericdata = df.select_dtypes(include = ['object'])


# In[15]:


numeric_data


# In[16]:


numeric_data.isnull().sum().tolist() # checking to see if there are any null values for numeric data


# In[17]:


non_numericdata


# In[18]:


non_numericdata.isnull().sum().tolist() # checking to see if there are any null of non numeric data 


# In[19]:


# Imputing Numeric Data
# Using multivariate imputation

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=5, random_state=0, verbose=2)
numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns=numeric_data.columns)
# here we impute the missing values, round it up, and put it back in the dataframe


# In[20]:


numeric_data


# In[21]:


numeric_data.isnull().sum().tolist() # checking to see if there are any null values


# In[22]:


non_numericdata


# In[23]:


# Imputation for non_numeric features using Forward Filling
non_numericdata = non_numericdata.ffill()
non_numericdata_df = pd.DataFrame(non_numericdata, columns=non_numericdata.columns)
non_numericdata_df


# In[24]:


non_numericdata_df.isnull().sum().tolist() # checking to see if there are any null values


# In[25]:


# One Hot Encoding
non_numericdata_df_encoded = pd.get_dummies(non_numericdata_df)
non_numericdata_df_encoded


# In[26]:


# Creating a dataframe that contains both the numeric and non numeric variables
new_df = pd.concat([numeric_data, non_numericdata_df_encoded], axis=1)
new_df


# In[27]:


# Separating the independent and dependent variables
y = new_df['overall']
X = new_df.drop('overall', axis = 1)


# In[28]:


y


# In[29]:


# Creating feature subsets that show maximum correlation with the dependent variable(overall rating)


# In[30]:


X_df =  pd.DataFrame(X)
correlations = X_df.corrwith(y)
correlations


# In[31]:


correlations.sort_values(ascending=False)


# In[32]:


# Correlation of 0.45 and above represents a moderate to strong correlation

selected_features = correlations[(correlations >= 0.45)] 
# selecting features with correlations greater than or equal to 0.45

selected_features.sort_values(ascending=False)


# In[33]:


# New dataframe with only the selected features
feature_subset = X_df[selected_features.index]
X = feature_subset # putting the selected features in X
X


# In[34]:


X.info()


# In[35]:


# Preprocessing 
# Scaling the independent variables

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)


# In[36]:


X.tolist()


# In[37]:


pd.Series(y).value_counts()


# In[38]:


# from the count we see there is bias, so we need to stratify when training
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
Xtrain.shape


# In[39]:


Xtest.shape


# In[40]:


# Cross Validation for the Random Forest Regressor

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

cv = KFold(n_splits=3)

RFM_PARAMETERS = {"max_depth":[5,10],
                  "min_samples_split": [2, 10],
                  "n_estimators":[50,100,200]}

full = RandomForestRegressor(n_jobs= -1)
RFM_model_gs = GridSearchCV (full, param_grid= RFM_PARAMETERS, cv=cv, scoring='neg_mean_squared_error')
RFM_model_gs.fit ( Xtrain, Ytrain)
RFMy_pred= RFM_model_gs.predict(Xtest)


# In[41]:


# Cross Validation for the XGBoost Regressor

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
cv = KFold(n_splits=3)
PARAMETERS = {"subsample":[0.5, 0.75, 1],
              "max_depth":[5, 10],
              "learning_rate":[0.1, 0.2, 0.3],
              "n_estimators":[50,100,200]}

XGBModel = xgb.XGBRegressor(n_jobs= -1)

XGBmodel_gs = GridSearchCV(XGBModel, param_grid=PARAMETERS, cv=cv, scoring='neg_mean_squared_error')
XGBmodel_gs.fit(Xtrain,Ytrain)
XGBy_pred= XGBmodel_gs.predict(Xtest)


# In[42]:


# Cross Validation for the Gradient Boosting Regressor - Ensemble

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


model = GradientBoostingRegressor()

param_grid = {
    'n_estimators': [50, 100,200],
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5]}

GBgrid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
GBgrid_search.fit(Xtrain, Ytrain)
GBy_pred = GBgrid_search.predict(Xtest)


# In[43]:


# Question 4
# Measure the model's performance and fine-tune it as a process of optimization

from sklearn.metrics import mean_absolute_error, r2_score
# Calculating the MAE of the RandomForestRegressor

mae_RFM = mean_absolute_error(Ytest, RFMy_pred )
r2Score_RFM = r2_score(Ytest,RFMy_pred)

print(f"Mean Absolute Error (MAE): {mae_RFM}")
print(f"R2 Score: {r2Score_RFM}")


# In[44]:


# Calculating the MAE of the XGBoostRegressor

mae_XGB= mean_absolute_error(Ytest, XGBy_pred )
r2Score_XGB = r2_score(Ytest,XGBy_pred)

print(f"Mean Absolute Error (MAE): {mae_XGB}")
print(f"R2 Score: {r2Score_XGB}")


# In[45]:


# Calculating the MAE of the GradientBoostingRegressor

mae_GB= mean_absolute_error(Ytest, GBy_pred )
r2Score_GB = r2_score(Ytest,GBy_pred)

print(f"Mean Absolute Error (MAE): {mae_GB}")
print(f"R2 Score: {r2Score_GB}")


# In[46]:


# Question 5
# Use the data from players_22 test how good is the model.
df2 = pd.read_csv('/Users/kwamenaduker/AI/players_22-1.csv', na_values = ['-',' ','']) # to remove the dashes and empty cells
df2


# In[47]:


# Separating the categorical and quantitative variables
numeric_data_test = df.select_dtypes(include = np.number)
non_numericdata_test = df.select_dtypes(include = ['object'])


# In[48]:


# Imputing Numeric Data
# Using multivariate imputation

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=5, random_state=0, verbose=2)
numeric_data_test = pd.DataFrame(np.round(imp.fit_transform(numeric_data_test)), columns=numeric_data_test.columns)
# here we impute the missing values, round it up, and put it back in the dataframe


# In[49]:


# From the features selected, there are no non numeric ones so we focus only on numeric data
# Creating a dataframe that contains both the numeric and non numeric variables
test_df = pd.DataFrame(numeric_data_test, columns=numeric_data_test.columns)
test_df


# In[50]:


# Separating the independent and dependent variables in the 2022 dataset
y1 = test_df['overall']
X1 = test_df.drop('overall', axis = 1)


# In[51]:


# Extracting the features with high correlation from the 2022 dataset based on observations in the 2021 dataset.
testData =test_df[selected_features.index] 


# In[52]:


testData


# In[53]:


# Scaling the independent variables of the 2022 dataset**"""
from sklearn.preprocessing import StandardScaler

X1=scaler.fit_transform(testData)


# In[54]:


# Testing the choosen model using the selected features from the 2022 dataset

XGBy_pred1= XGBmodel_gs.predict(testData)


# In[55]:


# Calulating the Mean Absolute Error and R2score

mae_XGB= mean_absolute_error(y1, XGBy_pred1 )
r2Score_XGB = r2_score(y1,XGBy_pred1)

print(f"Mean Absolute Error (MAE): {mae_XGB}")
print(f"R2 Score: {r2Score_XGB}")


# In[58]:


# Saving the model in a pickle file

import pickle
XGBmodel_gs_model = XGBmodel_gs

# Specify the filename for the pickle file
pickle_filename = 'XGBmodel_gs_model.pkl'

# Save the model to a pickle file
with open(pickle_filename, 'wb') as file:
    pickle.dump(XGBmodel_gs_model, file)

