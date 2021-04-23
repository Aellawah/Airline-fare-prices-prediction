#!/usr/bin/env python
# coding: utf-8

# # Airline fare prices prediction

# ## Table of contents

# * [Introduction](#Introduction)
# * [Data_wrangling](#Data_wrangling)
# * [Model_building](#Model_building)
# * [Conclusions](#Conclusion)

# ## Introduction

# ### About Dataset

# In[1]:


# Import libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[2]:


# import data

df = pd.read_excel('Data_Train.xlsx')


# ## Data_wrangling

# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


#checking airlines names

df['Airline'].unique()


# In[7]:


# checking for null values

df.isnull().sum()


# In[8]:


# checking null record

df[df['Route'].isnull()]


# In[9]:


# checking null record

df[df['Total_Stops'].isnull()]


# In[10]:


#removing null values

df.dropna(inplace=True)


# In[11]:


df.head(1)


# It seems that the Date_of_Journey consists of days months and years which can be extracted into three columns seperatley to inlcude it in the model after changing the column type from object to datetype.

# In[12]:


#changing date of journey column type into datetime type to be able to extract day and month and year columns seperatley from it

df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'])
df['Date_of_Journey_day']=df['Date_of_Journey'].dt.day
df['Date_of_Journey_month']=df['Date_of_Journey'].dt.month
df['Date_of_Journey_year']=df['Date_of_Journey'].dt.year


# In[13]:


#Dropping date of journey column after transformation

df.drop(['Date_of_Journey'],axis=1,inplace=True)


# In[14]:


df.head(1)


# It seems that the Dep-time consists of hours and minutes which can be extracted into two columns seperatley to inlcude it in the model after changing the column type from object to datetype

# In[15]:


#changing departure time column type into datetime type to be able to extract hour , minute columns seperatley from it.

df['Dep_Time']=pd.to_datetime(df['Dep_Time'])
df['dep_hour']=df['Dep_Time'].dt.hour
df['dep_minute']=df['Dep_Time'].dt.minute


# In[16]:


#Dropping Departure time column after transformation.

df.drop(['Dep_Time'],inplace=True,axis=1)


# It seems that the Arr-time consists of hours,minutes,months which can be extracted into three columns seperatley to inlcude it in the model after changing the column type from object to datetype

# In[17]:


#changing Arrival time column type into datetime type to be able to extract minute, hour ,month columns seperatley from it.

df['Arrival_Time']=pd.to_datetime(df['Arrival_Time'])
df['Arr_hour']=df['Arrival_Time'].dt.hour
df['Arr_minute']=df['Arrival_Time'].dt.minute
df['Arr_month']=df['Arrival_Time'].dt.month


# In[18]:


#Dropping Arrival time column after transformation.

df.drop(['Arrival_Time'],inplace=True,axis=1)


# In[19]:


#checking final results after transformation.

df.head(1)


# By looking at our categorical data, it seems that we have two kinds of cateorical data nominal categorical columns(Airline,Source,Destination) which means that they do not represent any order or rank in them and Total stops which is ordinal categorical data that represents rank in them.
# 
# I transform categorical columns into numerical data using one hot coding for nominal data and label encoding for ordinal data

# Airlines column

# In[20]:


#Checking airlines relative to their prices

colors=['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(10,5));
df.groupby(['Airline'])['Price'].sum().sort_values(ascending=False).plot.bar(color=colors);


# I will convert the column using one hot coding scince it is a nominal column

# In[21]:


#transforming Airline column from categorical into numerical using one hot coding

airline=df[['Airline']]
airline=pd.get_dummies(airline,drop_first=True)
airline.head()


# Source column

# In[22]:


#Checking how many flights for each source

df['Source'].value_counts()


# In[23]:


#Checking Source relative to their prices

plt.figure(figsize=(10,5));
df.groupby(['Source'])['Price'].sum().sort_values(ascending=False).plot.bar(color=colors);


# In[24]:


#transforming source column from categorical into numerical using one hot coding

source=df[['Source']]
source=pd.get_dummies(source,drop_first=True)
source.head()


# Destinations column

# In[25]:


#checking how many flights are going to available destinations in our dataset

df['Destination'].value_counts()


# In[26]:


#checking destinations relative to their prices

plt.figure(figsize=(10,5));
df.groupby(['Destination'])['Price'].sum().sort_values(ascending=False).plot.bar(color=colors);


# In[27]:


#transforming destination column from categorical into numerical using one hot coding

destination=df[['Destination']]
destination=pd.get_dummies(source,drop_first=True)
destination.head()


# In[28]:


# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[29]:


# Adding duration_hours and duration_mins list to train_data dataframe

df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins


# Transforming ordinal column using label_encoder

# In[30]:


#Checking how many flights for every stop

df['Total_Stops'].value_counts()


# In[31]:


#Transforming using label_encoder

from sklearn.preprocessing import LabelEncoder
df['Total_Stops']=LabelEncoder().fit_transform(df['Total_Stops'])


# After finishing transformations will drop the categorical from the dataframe and concatenate the transformed columns into it

# I will dropthe route and the duration columns from the dataframe since i will not include them into my model

# In[32]:


#dropping columns

df.drop(['Route','Airline','Source','Destination','Duration','Additional_Info','Duration'],axis=1,inplace=True)


# In[33]:


#Concatenating transformed airline source destination into the dataframe

df=pd.concat([airline,source,destination,df],axis=1)


# In[34]:


#cheching final results

df.head()


# ### Model_building

# In[35]:


df.shape


# In[36]:


tdf.shape


# In[ ]:


#selecting train independant features to include in my model

x=df.drop('Price',axis=1)


# In[ ]:


#selecting dependable variable i am trying to predict

y=df['Price']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[ ]:


#fit the model on x_Train and y_Train variables

from sklearn.ensemble import RandomForestRegressor
RF= RandomForestRegressor()
RF.fit(X_train,y_train)


# In[ ]:


# Trying to predict on the test data set

predictions=RF.predict(X_test)


# In[ ]:


predictions


# In[ ]:


sns.distplot(y_test-predictions);


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[ ]:


metrics.r2_score(y_test,predictions)


# ## Conclusions

# After transforming the data by removing nulls and changing the column types and transforming the categorical data columns using one hot coding and label encoder we included them into the Random forest regressor model whch gave us an MAE of 1194 and MSE of 453 and RMSE of 2130 and R2 79% which means that the model represents 79% percentage of the model variance.
