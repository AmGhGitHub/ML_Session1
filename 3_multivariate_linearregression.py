import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_boston

# https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
matplotlib.style.use('seaborn-notebook')
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
scatter_matrix(X)

correlations = X.corr(method='pearson')
# print(correlations)
fig = plt.figure()
ax = fig.add_subplot(111)
# get cmap from https://matplotlib.org/users/colormaps.html
cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='jet')
fig.colorbar(cax)
ticks = np.arange(0, X.shape[1], 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(X.columns)
ax.set_yticklabels(X.columns)

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
# print(skewness)

# X.hist(edgecolor='black', grid=False, color='blue')
# plt.tight_layout()
X.plot(kind='density', subplots=True, layout=(4, 4), sharex=False)
plt.tight_layout()
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f'R^2 for linear model without feature scaling: {r2_score(y_test, lr.predict(X_test)) * 100:.3f}%')
print(f'MSE for linear model without feature scaling: {mean_squared_error(y_test, lr.predict(X_test)):.3f}')
print(f'MAE for linear model without feature scaling: {mean_absolute_error(y_test, lr.predict(X_test)):.3f}')
df_coeff = pd.DataFrame(zip(X.columns, lr.coef_), columns=['feature', 'Coeff'])

min_max_scale = MinMaxScaler()
X_train_norm = min_max_scale.fit_transform(X_train)
X_test_norm = min_max_scale.transform(X_test)

lr_norm = LinearRegression()
lr_norm.fit(X_train_norm, y_train)
print(f'\n\nR^2 for linear model with feature scaling: {r2_score(y_test, lr_norm.predict(X_test_norm)) * 100:.3f}%')
print(f'MSE for linear model wit feature scaling: {mean_squared_error(y_test, lr_norm.predict(X_test_norm)):.3f}')
print(f'MAE for linear model wit feature scaling: {mean_absolute_error(y_test, lr_norm.predict(X_test_norm)):.3f}')
df_coeff_norm = pd.DataFrame(zip(X.columns, lr_norm.coef_), columns=['feature', 'Coeff'])

lr_eq = ""
for i, feature in enumerate(X.columns):
    lr_eq += f'{lr_norm.coef_[i]:.2f}*{feature}'

print(lr_eq)
