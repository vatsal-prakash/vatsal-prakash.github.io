#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import csv
import json
import requests
from datetime import datetime


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
focus = df.copy().drop(['Lat','Long'], axis=1).set_index(['Country/Region','Province/State'])
confirm = focus.groupby('Country/Region').sum().reset_index()


# In[3]:


do_not_include = ['Antigua and Barbuda', 'Angola', 'Benin', 'Botswana', 
                   'Burundi', 'Cabo Verde', 'Chad', 'Comoros', 
                  'Congo (Brazzaville)', 'Congo (Kinshasa)',"Cote d'Ivoire", 'Central African Republic',
                  'Diamond Princess', 'Equatorial Guinea',
                  'Eritrea', 'Ecuador', 'Eswatini',   'Gabon', 
                  'Gambia', 'Ghana', 'Grenada', 'Guinea', 'Guinea-Bissau',
                  'Guyana', 'Laos', 'Lesotho', 'Liberia', 'Libya', 'Madagascar',
                  'Malawi', 'Maldives', 'Mauritania', 'Mozambique',
                  'MS Zaandam', 'Namibia', 'Nicaragua', 'Papua New Guinea',
                  'Rwanda',   'Saint Lucia', 
                  'Saint Vincent and the Grenadines', 'Sao Tome and Principe',
                  'Seychelles', 'Sierra Leone', 'South Sudan', 'Suriname', 'Syria', 
                  'Tanzania',   'Togo', 'Uganda', 'West Bank and Gaza',
                  'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe']


# In[4]:


focus


# In[5]:


# replacing 0 total cases with nan
confirm.replace(0, np.nan, inplace=True)


# In[6]:


confirm


# In[7]:


# convert "pivoted" data to "long form"
data = pd.melt(confirm, id_vars=['Country/Region'], var_name='date', value_name='cases')

data = data.rename(columns = {'Country/Region':'country'})

# convert date column
data['date'] = pd.to_datetime(data['date'], format= '%m/%d/%y')


# In[8]:


data


# In[9]:


# pivot data with countries as columns
pivot_cases = pd.pivot_table(data, index = "date", columns = "country", values= "cases")

# drop countries listed above
pivot_cases = pivot_cases.drop(columns=do_not_include)


# In[10]:


pivot_cases


# In[11]:


# new dataframe to store "daily new cases"
pivot_newcases = pivot_cases.copy()

# calculate "daily new cases"
for column in pivot_newcases.columns[0:]:
    #DailyNewCases = column
    pivot_newcases[column] = pivot_newcases[column].diff()


# In[12]:


# fill NaN in pivot_newcases (first row) with values from pivot_cases
pivot_newcases.fillna(pivot_cases, inplace=True)


# In[13]:


pivot_newcases


# In[14]:


pivot_newcases


# In[15]:


# replace negative daily values by setting 0 as the lowest value
pivot_newcases = pivot_newcases.clip(lower=0)


# In[16]:


# new dataframe to store "avg new cases"
pivot_avgnewcases = pivot_newcases.copy()

# calculate 7-day averages of new cases
for column in pivot_avgnewcases.columns[0:]:
    DaySeven = column
    pivot_avgnewcases[DaySeven] = pivot_avgnewcases[column].rolling(window=7, center=False).mean()


# In[17]:


# fill NaN in pivot_avgnewcases (first 6 rows) with values from pivot_newcases
pivot_recentnew = pivot_avgnewcases.fillna(pivot_newcases)


# In[18]:


pivot_recentnew


# In[19]:


# new dataframe to store "avg new cases" with centered average
pivot_avgnewcases_center = pivot_newcases.copy()

# calculate 7-day averages of new cases with centered average
for column in pivot_avgnewcases_center.columns[0:]:
    DaySeven = column
    pivot_avgnewcases_center[DaySeven] = pivot_avgnewcases_center[column].rolling(window=7, min_periods=4, center=True).mean()


# In[20]:


pivot_avgnewcases_center


# In[21]:


# reset indexes of "pivoted" data
pivot_cases = pivot_cases.reset_index()
pivot_newcases = pivot_newcases.reset_index()
pivot_recentnew = pivot_recentnew.reset_index()
pivot_avgnewcases_center = pivot_avgnewcases_center.reset_index()


# In[22]:


# convert "pivot" of total cases to "long form"
country_cases = pd.melt(pivot_cases, id_vars=['date'], var_name='country', value_name='cases')


# In[23]:


country_cases


# In[24]:


# convert "pivot" of daily new cases to "long form"
country_newcases = pd.melt(pivot_newcases, id_vars=['date'], var_name='country', value_name='new_cases')


# In[25]:


country_newcases


# In[26]:


# convert "pivot" of recent new cases to "long form" (7-day avg w first 6 days from "new cases")
country_recentnew = pd.melt(pivot_recentnew, id_vars=['date'], var_name='country', value_name='recent_new')


# In[27]:


country_recentnew


# In[28]:


# convert "pivot" of centered average new cases to "long form"
country_avgnewcases_center = pd.melt(pivot_avgnewcases_center, id_vars=['date'], var_name='country', value_name='avg_cases')


# In[29]:


country_avgnewcases_center


# In[30]:


# merge the 4 "long form" dataframes based on index
country_merge = pd.concat([country_cases, country_newcases, country_avgnewcases_center, country_recentnew], axis=1)


# In[31]:


country_merge


# In[32]:


# remove duplicate columns
country_merge = country_merge.loc[:,~country_merge.columns.duplicated()]


# In[33]:


# dataframe with only the most recent date for each country
# https://stackoverflow.com/questions/23767883/pandas-create-new-dataframe-choosing-max-value-from-multiple-observations
country_latest = country_merge.loc[country_merge.groupby('country').date.idxmax().values]


# In[34]:


country_latest


# In[35]:


# dataframe with peak average cases for each country
peak_avg_cases = country_merge.groupby('country')['avg_cases'].agg(['max']).reset_index()
peak_avg_cases = peak_avg_cases.rename(columns = {'max':'peak_avg_cases'})


# In[36]:


# merging total cases onto the merged dataframe
country_color_test = country_latest.merge(peak_avg_cases, on='country', how='left')


# In[37]:


country_color_test


# In[38]:


#choosing colors
n_0 = 20
f_0 = 0.5
f_1 = 0.2

# https://stackoverflow.com/questions/49586471/add-new-column-to-python-pandas-dataframe-based-on-multiple-conditions/49586787
def conditions(country_color_test):
    if country_color_test['avg_cases'] <= n_0*f_0 or country_color_test['avg_cases'] <= n_0 and country_color_test['avg_cases'] <= f_0*country_color_test['peak_avg_cases']:
        return 'green'
    elif country_color_test['avg_cases'] <= 1.5*n_0 and country_color_test['avg_cases'] <= f_0*country_color_test['peak_avg_cases'] or country_color_test['avg_cases'] <= country_color_test['peak_avg_cases']*f_1:
        return 'yellow'
    else:
        return 'red'

country_color_test['color'] = country_color_test.apply(conditions, axis=1)


# In[39]:


country_color_test


# In[40]:


# dataframe with just country, total cases, and color
country_total_color = country_color_test[['country','cases','color']]

# rename cases to total_cases for the purpose of merging
country_total_color = country_total_color.rename(columns = {'cases':'total_cases'})


# In[41]:


# merging total cases onto the merged dataframe
country_final = country_merge.merge(country_total_color, on='country', how='left')


# In[42]:


country_final = country_final[['country','date','cases','new_cases','avg_cases','total_cases','recent_new','color']]


# In[43]:


country_final


# In[44]:


# drop rows where cumulative cases is NaN (dates before reported cases)
country_final = country_final.dropna(subset=['cases']) 


# In[45]:


country_final


# In[46]:


country_final.drop(columns ='cases', inplace = True)


# In[47]:


country_final.head()


# In[48]:


country_final.to_csv('result.csv')

