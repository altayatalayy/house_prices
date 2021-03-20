import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


from sklearn.model_selection import train_test_split
df = pd.read_csv('./data/train.csv')
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
X_train = train_set.drop('SalePrice', axis=1)
y_train = train_set['SalePrice'].copy()

X_test = test_set.drop('SalePrice', axis=1)
y_test = test_set['SalePrice'].copy()

num_attr = X_train.select_dtypes([np.number])
cat_attr = X_train.select_dtypes([object])

# Preprocess data
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

'''
Pipeline for numeric values:
    handle missing values by replacing them with the median value,
    standardization of the values with scikit-learns stdscaler

'''
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
    ])

'''
Full pipeline to transform both numerical and categorical data:
    use num_pipeline for numeric values
    use one hot encoding for categorical
'''
pipeline = ColumnTransformer([
    ('num', num_pipeline, list(num_attr)),
    ('cat', OneHotEncoder(), list(cat_attr)),
    ])

X_train = pipeline.fit_transform(X_train)
#X_test = pipeline.transform(X_test)


# Training & Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as loss

reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluate on training data
y_pred = reg.predict(X_train)
mse = loss(y_pred, y_train)
print(mse)
