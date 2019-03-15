import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston

# https://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
boston_house = load_boston()

# boston data is a dictionary, so we can get an idea about the data structure using keys

# print(boston_house.keys())
print(boston_house['DESCR'])

features = [feat.lower() for feat in boston_house['feature_names']]

df_boston = pd.DataFrame(data=boston_house['data'], columns=features)
df_boston['price'] = boston_house['target']
