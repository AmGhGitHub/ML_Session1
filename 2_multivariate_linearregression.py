import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston

# https://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
boston_house = load_boston()
df_bh = pd.DataFrame(data=boston_house.data, columns=boston_house.feature_names)
df_bh['price'] = boston_house.target
