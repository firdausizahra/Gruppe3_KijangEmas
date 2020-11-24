#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import csv


# In[11]:


df = pd.read_excel ('kijangemas_quelle.xlsx')
print (df)
print(df.columns.ravel())


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


# In[19]:


timestamp, selling = [], []  
timestamp = pd.DatetimeIndex(df['Date']).date
selling = selling = df["1 oz Sell"]
len(timestamp), len(selling)


# In[20]:


plt.figure(figsize = (15, 5))
plt.plot(selling)
plt.xticks(np.arange(len(timestamp))[::15], timestamp[::15], rotation = '45')
plt.show()


# # Verteilungsanalyse

# In[22]:


plt.figure(figsize = (15, 5))
sns.distplot(df["1 oz Sell"])
plt.show()


# # lagging analysis

# In[29]:


import pandas as pd
df2 = pd.DataFrame({'timestamp': timestamp, 'selling': selling})
df2.head()


# In[49]:


def df_shift(df2, lag = 0, start = 1, skip = 1, rejected_columns = []):
    df2 = df2.copy()
    if not lag:
        return df2
    cols = {}
    for i in range(start, lag + 1, skip):
        for x in list(df2.columns):
            if x not in rejected_columns:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data = None, columns = columns, index = df2.index)
        i = start - 1
        for c in columns:
            dfn[c] = df2[k].shift(periods = i)
            i += skip
        df2 = pd.concat([df2, dfn], axis = 1, join_axes = [df2.index])
    return df2


# # Shifted and moving average are not same.
# 
# moving average: the moving average is commonly used with time series to smooth random short-term variations and to highlight other components such as trend, season or cycle present in the data. The moving average is also knows as rolling mean, is calculated by averaging data of the time series withing k periods of time. rolling(n).mean() for unit k means the sum of (unit_k + unit_k-1 + ... + unit_k-n) / n
# 
# shifted:
# 
# lag:

# # why we lagged or shifted to certain units?
# Virals took some time, impacts took some time, same goes to price lot / unit.
# 
# Now I want to lag for until 12 units, start at 4 units shifted, skip every 2 units.

# In[31]:


df_crosscorrelated = df_shift(
    df2, lag = 12, start = 4, skip = 2, rejected_columns = ['timestamp']
)
df_crosscorrelated['ma7'] = df_crosscorrelated['selling'].rolling(7).mean()
df_crosscorrelated['ma14'] = df_crosscorrelated['selling'].rolling(14).mean()
df_crosscorrelated['ma21'] = df_crosscorrelated['selling'].rolling(21).mean()


# In[32]:


df_crosscorrelated.head(21)


# In[33]:


plt.figure(figsize = (20, 4))
plt.subplot(1, 3, 1)
plt.scatter(df_crosscorrelated['selling'], df_crosscorrelated['selling_4'])
mse = (
    (df_crosscorrelated['selling_4'] - df_crosscorrelated['selling']) ** 2
).mean()
plt.title('close vs shifted 4, average change: %f'%(mse))
plt.subplot(1, 3, 2)
plt.scatter(df_crosscorrelated['selling'], df_crosscorrelated['selling_8'])
mse = (
    (df_crosscorrelated['selling_8'] - df_crosscorrelated['selling']) ** 2
).mean()
plt.title('close vs shifted 8, average change: %f'%(mse))
plt.subplot(1, 3, 3)
plt.scatter(df_crosscorrelated['selling'], df_crosscorrelated['selling_12'])
mse = (
    (df_crosscorrelated['selling_12'] - df_crosscorrelated['selling']) ** 2
).mean()
plt.title('close vs shifted 12, average change: %f'%(mse))
plt.show()


# In[41]:



plt.figure(figsize = (10, 5))
plt.scatter(
    df_crosscorrelated['selling'],
    df_crosscorrelated['selling_4'],
    label = 'close vs shifted 4',
)
plt.scatter(
    df_crosscorrelated['selling'],
    df_crosscorrelated['selling_8'],
    label = 'close vs shifted 8',
)
plt.scatter(
    df_crosscorrelated['selling'],
    df_crosscorrelated['selling_12'],
    label = 'close vs shifted 12',
)
plt.legend()
plt.show()


# In[47]:


fig, ax = plt.subplots(figsize = (15, 5))
df_crosscorrelated.plot(
    x = 'timestamp', y = ['selling', 'ma7', 'ma14', 'ma21'], ax = ax
)


# As you can see, even moving average 7 already not followed sudden trending (blue line), means that, dilation rate required less than 7 days! so fast!
# 
# # How about correlation?
# We want to study linear relationship between, how many days required to give impact to future sold units?

# In[51]:



colormap = plt.cm.RdBu
plt.figure(figsize = (15, 5))
plt.title('cross correlation', y = 1.05, size = 16)

sns.heatmap(
    df_crosscorrelated.iloc[:, 1:].corr(),
    linewidths = 0.1,
    vmax = 1.0,
    cmap = colormap,
    linecolor = 'white',
    annot = True,
)
plt.show()


# Based on this correlation map, look at selling vs selling_X,
# 
# selling_X from 4 to 12 is getting lower, means that, if today is 50 mean, next 4 days should increased by 0.995 * 50 mean, and continue.
# 
# # Outliers
# Simple, we can use Z-score to detect outliers, which timestamps gave very uncertain high and low value.

# In[52]:


std_selling = (selling - np.mean(selling)) / np.std(selling)


# In[53]:


def detect(signal, treshold = 2.0):
    detected = []
    for i in range(len(signal)):
        if np.abs(signal[i]) > treshold:
            detected.append(i)
    return detected


# In[54]:


outliers = detect(std_selling)


# In[55]:


plt.figure(figsize = (15, 7))
plt.plot(selling)
plt.plot(
    np.arange(len(selling)),
    selling,
    'X',
    label = 'outliers',
    markevery = outliers,
    c = 'r',
)
plt.legend()
plt.show()


# # Predictive Modelling
# # Lineare Regression

# In[56]:


from sklearn.linear_model import LinearRegression


# In[60]:


0.8*len(selling)


# In[61]:


train_selling = selling[: int(0.8 * len(selling))]
test_selling = selling[int(0.8 * len(selling)) :]


# In[62]:


future_count = len(test_selling)
future_count


# In[63]:


get_ipython().run_cell_magic('time', '', 'linear_regression = LinearRegression().fit(\n    np.arange(len(train_selling)).reshape((-1, 1)), train_selling\n)\nlinear_future = linear_regression.predict(\n    np.arange(len(train_selling) + future_count).reshape((-1, 1))\n)')


# In[64]:


fig, ax = plt.subplots(figsize = (15, 5))
ax.plot(selling, label = '20% test trend')
ax.plot(train_selling, label = '80% train trend')
ax.plot(linear_future, label = 'forecast linear regression')
plt.xticks(
    np.arange(len(timestamp))[::10],
    np.arange(len(timestamp))[::10],
    rotation = '45',
)
plt.legend()
plt.show()


# In[67]:


from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


# In[68]:


def calculate_accuracy(real, predict):
    r2 = r2_score(real, predict)
    if r2 < 0:
        r2 = 0

    def change_percentage(val): 
    # minmax, we know that correlation is between -1 and 1
        if val > 0:
            return val
        else:
            return val + 1

    pearson = pearsonr(real, predict)[0]
    spearman = spearmanr(real, predict)[0]
    pearson = change_percentage(pearson)
    spearman = change_percentage(spearman)
    return {
        'r2': r2 * 100,
        'pearson': pearson * 100,
        'spearman': spearman * 100,
    }


# In[76]:


def calculate_distance(real, predict):
    mse = ((real - predict) ** 2).mean()
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse}


# In[70]:


linear_cut = linear_future[: len(train_selling)]
#arima_cut = arima_future[: len(train_selling)]
#lstm_cut = lstm_future[: len(train_selling)]


# 80%

# In[75]:


calculate_distance(train_selling, linear_cut)


# In[73]:


calculate_accuracy(train_selling, linear_cut)


# 20%

# In[78]:


linear_cut = linear_future[len(train_selling) :]


# In[79]:


calculate_distance(test_selling, linear_cut)


# In[80]:


calculate_accuracy(test_selling, linear_cut)


# In[ ]:




