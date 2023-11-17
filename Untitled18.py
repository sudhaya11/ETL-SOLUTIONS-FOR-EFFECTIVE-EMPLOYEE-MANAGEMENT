#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install mysql-connector-python')
import mysql.connector as connection


# In[3]:


import pandas as pd
import mysql.connector as sql

mydb=sql.connect(host='localhost',
                 user='root',
                 password='root@123',
                 database='pozent')

mycursor=mydb.cursor()

query="select * from employee"
dd=pd.read_sql(query,mydb)


dd


# In[4]:


target_data=dd[dd['salary']>=2000000]


# In[5]:


target_data


# In[6]:


target_data.to_csv('output_data.csv')


# In[7]:


data=pd.read_csv(r'output_data.csv')
data


# In[8]:


dd['transformed_column'] = dd['salary'] + 1000
dd


# In[9]:


dd.shape


# In[10]:


dd.size


# In[11]:


#DATA BALANCE


# In[12]:


# Assuming you have a DataFrame named df
class_balance = dd['job_desc'].value_counts()

print(class_balance)


# In[13]:


print("class_balance:")
print(class_balance)


# In[14]:


class_balance_percentage=dd['job_desc'].value_counts(normalize=True)


# In[15]:


print(class_balance_percentage)


# In[16]:


#DATA EMPTINESS


# In[17]:


# Count the missing values in each column
missing_values = dd.isnull().sum()

print(missing_values)


# In[ ]:





# In[18]:


# Check for rows with all missing values
empty_rows = dd[dd.isnull().all(axis=1)]

print(empty_rows)


# In[19]:


# Calculate the percentage of missing values in the entire DataFrame
sparsity = dd.isnull().mean().mean() * 100

print(f"Dataset sparsity: {sparsity:.2f}%")


# In[20]:


#IMPUTE


# In[21]:


dd_imputed = dd.fillna(dd.mean())


# In[ ]:


dd_imputed = dd.fillna(dd.mean()) 


# In[ ]:


dd


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame and 'column_name' is the column with numeric data
sns.boxplot(x=dd['salary'])
plt.show()


# In[ ]:


# Remove outliers from 'column_name' using a z-score threshold
from scipy.stats import zscore

z_scores = zscore(dd['salary'])
dd_no_outliers = dd[(z_scores < 3) & (z_scores > -3)]


# In[ ]:


get_ipython().system('pip install Numpy')
import numpy as np


# Log transformation to reduce the impact of outliers
dd['salary_log'] = np.log1p(dd['salary'])


# In[ ]:


from scipy.stats.mstats import winsorize

# Winsorize 'column_name' at the 5th and 95th percentiles
dd['salary_winsorized'] = winsorize(dd['salary'], limits=[0.05, 0.05])


# In[ ]:


import pandas as pd
from scipy.stats import zscore

# Assuming df is your DataFrame and 'column_name' is the column with numeric data
# Calculate Z-scores
z_scores = zscore(dd['salary'])

# Define a threshold for considering values as outliers (e.g., 3 standard deviations)
threshold = 3

# Create a mask for outliers
outlier_mask = (abs(z_scores) > threshold)

# Remove outliers from the DataFrame
dd_no_outliers = dd[~outlier_mask]


# In[ ]:


# Calculate the IQR
Q1 = dd['salary'].quantile(0.25)
Q3 = dd['salary'].quantile(0.75)
IQR = Q3 - Q1

# Create a mask for outliers using IQR
outlier_mask_iqr = (dd['salary'] < Q1 - 1.5 * IQR) | (dd['salary'] > Q3 + 1.5 * IQR)

# Remove outliers using the IQR method
dd_no_outliers_iqr = dd[~outlier_mask_iqr]


# In[ ]:


print(dd_no_outliers_iqr)


# In[30]:


correlation_matrix = dd.corr()
outcome_correlations = correlation_matrix['transformed_column'].sort_values(ascending=False)
print(outcome_correlations)


# In[32]:


from scipy.stats import ks_2samp

# Assuming df_train is your training dataset and df_new is your incoming dataset
ks_statistic, ks_p_value = ks_2samp(dd_train['target_data'], dd_new['output_data'])

if ks_p_value < 0.05:
    print("Data drift detected!")
else:
    print("No significant data drift.")


# In[ ]:




