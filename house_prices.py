import numpy as np
import pandas as pd


df = pd.read_csv('./data/train.csv')

import matplotlib.pyplot as plt
import seaborn as sn

'''
figsize=(20,10)
df.hist(bins=100, figsize=figsize)
plt.show()
'''

corr = df.corr()
sc = corr['SalePrice'].sort_values()
print(sc)
attr = [k for k, v in sc.items() if abs(v) > 0.6]
print(attr)
num_attr = df.select_dtypes([np.number])
print(num_attr.isna().any())
cat_attr = ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'LandContour']

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
house_cat_1hot = cat_encoder.fit_transform(df[cat_attr])

from pandas.plotting import scatter_matrix
scatter_matrix(df[attr], figsize=(16, 10))
plt.show()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
