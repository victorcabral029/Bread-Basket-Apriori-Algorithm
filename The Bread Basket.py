#!/usr/bin/env python
# coding: utf-8

# ### Bread Basket Association Rules with Apriori Algortithm
# 
# Using the Apriori Algorithm, I will try to find some associations between products in orders of a bakery in Edinburgh, Scotland. The dataset has 20507 entries, over 9000 transactions, and 5 columns.
# 
# * Transaction - Id of the transaction
# * Item - Product of that transaction
# * date_time - Day and hour
# * period_day - Period of the Day (morning, afteernoon, evening, night) 
# * weekday_weekend - whether it was weekday or weekend

# ### Importing The Libraries

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing The Dataset

# In[56]:


basket = pd.read_csv('bread basket.csv')


# ### Dataset Description

# In[57]:


basket.head()


# In[58]:


basket.info()


# ### Visualizing Null Values

# In[59]:


plt.subplots(figsize=(10,6))
sns.heatmap(basket.isnull())


# Non null values in this dataset

# ## Data Cleaning

# ### Cleaning item column

# In[60]:


basket['Item'] = basket['Item'].str.lower()
basket['Item'] = basket['Item'].str.strip()


# ### Spliting date time column

# In[61]:


basket['Datetime'] = pd.to_datetime(basket['date_time'])


# In[62]:


basket['date'] = basket['Datetime'].dt.date
basket['month'] = basket['Datetime'].dt.month
basket['day'] = basket['Datetime'].dt.weekday
basket['hour'] = basket['Datetime'].dt.hour

basket['day'] = basket['day'].replace((0,1,2,3,4,5,6), 
('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))


# In[63]:


basket.drop(columns='date_time',inplace=True)


# In[64]:


basket.head(3)


# ## Data Visualization

# ### Transactions per hour of the day

# In[65]:


countByHour = basket.groupby('hour')['Transaction'].count().reset_index()
countByHour.sort_values('hour',inplace=True)


# In[66]:


colors = sns.color_palette("YlOrRd", 18)
fig = plt.figure(figsize=(12,5))
sns.barplot(x=countByHour['hour'], y=countByHour['Transaction'], palette = colors)


# ### Transactions by Day Period

# In[13]:


countByDayPeriod = basket.groupby('period_day')['Transaction'].count().reset_index()
countByDayPeriod.loc[:,"orderOfDayPeriod"] = [1,2,0,3]
countByDayPeriod.sort_values('orderOfDayPeriod',inplace=True)


# In[14]:


colors = sns.color_palette("YlOrRd", 4)
fig = plt.figure(figsize=(12,5))
sns.barplot(x=countByDayPeriod['period_day'], y=countByDayPeriod['Transaction'], palette = colors)


# Most transactions in the morning and afternoon period

# ### Transactions per day of the week

# In[67]:


countByDay = basket.groupby('day')['Transaction'].count().reset_index()
countByDay.loc[:,"orderOfDays"] = [4,0,5,6,3,1,2]
countByDay.sort_values("orderOfDays",inplace=True)


# In[68]:


colors = sns.color_palette("YlOrRd", 7)
fig = plt.figure(figsize=(12,5))
sns.barplot(x=countByDay['day'], y=countByDay['Transaction'], palette = colors)


# ### Transactions by Week Period

# In[69]:


countByWeekPeriod = basket.groupby('weekday_weekend')['Transaction'].count().reset_index()
countByWeekPeriod.sort_values('weekday_weekend',inplace=True)


# In[70]:


colors = sns.color_palette("YlOrRd", 2)
fig = plt.figure(figsize=(12,5))
sns.barplot(x=countByWeekPeriod['weekday_weekend'], y=countByWeekPeriod['Transaction'], palette = colors)


# ### Transactions By Month

# In[19]:


countByMonth = basket.groupby('month')['Transaction'].count().reset_index()
countByMonth.sort_values('month',inplace=True)


# In[20]:


colors = sns.color_palette("YlOrRd", 12)
fig = plt.figure(figsize=(12,5))
sns.barplot(x=countByMonth['month'], y=countByMonth['Transaction'], palette = colors)


# Most transactions in fall and winter

# ### Top 25 sold items

# In[21]:


fig = plt.figure(figsize=(15,5))
colors = sns.color_palette("YlOrRd", 25)
names = basket.Item.value_counts().head(25).index
values = basket.Item.value_counts().head(25)
sns.barplot(x = names, y = values, palette = colors)
plt.xticks(rotation=45)


# Coffee and bread are clearly the best selling products

# ### Top items sold by day period

# In[71]:


items = basket.groupby(['Item','period_day'])['Transaction'].count().reset_index().sort_values(['period_day','Transaction'],ascending=False)


# In[72]:


colors = sns.color_palette("YlOrRd", 10)
fig = plt.subplots(figsize=(17,9))
plt.subplots_adjust(hspace = 0.6)

plt.subplot(2,2,1)
plt.xticks(rotation=45)
plt.title('Top 10 orders in the morning')
dfMorning = items[items['period_day']=='morning'].head(10) 
sns.barplot(x = dfMorning.Item, y = dfMorning.Transaction, palette = colors)

plt.subplot(2,2,2)
plt.xticks(rotation=45)
plt.title('Top 10 orders in the afternoon')
dfAfternoon = items[items['period_day']=='afternoon'].head(10) 
sns.barplot(x = dfAfternoon.Item, y = dfAfternoon.Transaction, palette = colors)

plt.subplot(2,2,3)
plt.xticks(rotation=45)
plt.title('Top 10 orders in the evening')
dfEvening = items[items['period_day']=='evening'].head(10) 
sns.barplot(x = dfEvening.Item, y = dfEvening.Transaction, palette = colors)

plt.subplot(2,2,4)
plt.xticks(rotation=45)
plt.title('Top 10 orders in the night')
dfNight = items[items['period_day']=='night'].head(10) 
sns.barplot(x = dfNight.Item, y = dfNight.Transaction, palette = colors)


# ### Top items sold by Weekend or WeekDay

# In[73]:


items2 = basket.groupby(['Item','weekday_weekend'])['Transaction'].count().reset_index().sort_values(['weekday_weekend','Transaction'],ascending=False)


# In[74]:


colors = sns.color_palette("YlOrRd", 10)
fig = plt.subplots(figsize=(15,5))

plt.subplot(1,2,1)
plt.xticks(rotation=45)
plt.title('Top 10 orders in weekdays')
dfWeekday = items2[items2['weekday_weekend']=='weekday'].head(10) 
sns.barplot(x = dfWeekday.Item, y = dfWeekday.Transaction, palette = colors)

plt.subplot(1,2,2)
plt.xticks(rotation=45)
plt.title('Top 10 orders in weekends')
dfWeekend = items2[items2['weekday_weekend']=='weekend'].head(10) 
sns.barplot(x = dfWeekend.Item, y = dfWeekend.Transaction, palette = colors)


# ## Apriori Algorithm

# In[75]:


transactions = basket.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Number of Items')
table = transactions.pivot_table(index='Transaction', columns='Item', values='Number of Items', aggfunc='sum').fillna(0)


# In[76]:


table.head()


# In[77]:


def hot_encode(x): 
    if(x==0): 
        return False
    if(x>0): 
        return True


# In[78]:


final_table = table.applymap(hot_encode) 


# In[79]:


final_table.head()


# In[80]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[81]:


frequence = apriori(final_table, min_support=0.015, use_colnames=True)
rules = association_rules(frequence, metric="lift", min_threshold=1)


# In[82]:


rules.head()


# In[85]:


rules.sort_values('confidence', ascending = False, inplace=True)
rules


# #### We can clearly see that coffee is an item that is very associated with other purchases in the bakery, such as toast, medialuna and pastry

# In[ ]:




