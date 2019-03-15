import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston

pd.set_option('precision', 3)

# https://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
boston_house = load_boston()

# boston data is a dictionary, so we can get an idea about the data structure using keys

# print(boston_house.keys())
# print(boston_house['DESCR'])


features = [feat.lower() for feat in boston_house['feature_names']]
df_boston = pd.DataFrame(data=boston_house['data'], columns=features)
df_boston['price'] = boston_house['target']

#### Get info and description of the dataframe
# print(df_boston.info())
# print(df_boston.describe())

X = df_boston[df_boston.columns[:-1]]
y = df_boston[df_boston.columns[-1]]
# print(X.head(1))
# print(y.head(1))


"""
1- Correlation refers to the relationship between two variables 
and how they may or may not
change together. 
2- The most common method for calculating correlation is Pearson’s Correlation
Coefficient, that assumes a normal distribution of the attributes involved. 
3- A correlation of -1 or 1 shows a full negative or positive correlation respectively. Whereas a value of 0 shows no
correlation at all. 
4- Some machine learning algorithms like linear and logistic regression can suffer
poor performance if there are highly correlated attributes in your dataset. 
"""
correlations = X.corr(method='pearson')


"""
1- Skew refers to a distribution that is assumed Gaussian (normal or bell curve) that is shifted or
squashed in one direction or another. 
2- Many machine learning algorithms assume a Gaussian
distribution. 
3- Knowing that an attribute has a skew may allow you to perform data preparation
to correct the skew and later improve the accuracy of your models. 
4- You can calculate the skew of each attribute using the skew() function on the Pandas DataFrame.
"""
skewness = X.skew()

print(correlations)
print(skewness)
