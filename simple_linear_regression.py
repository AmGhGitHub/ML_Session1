import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df_capita = pd.read_csv('data/canada_per_capita_income.csv', header=[0])
X = df_capita.iloc[:, 0].values
y = df_capita.iloc[:, 1].values
X = X[:, np.newaxis]
y = y[:, np.newaxis]
poly_dg = 3
scaler = PolynomialFeatures(degree=poly_dg)
X2 = scaler.fit_transform(X)

slr = LinearRegression()
plr = LinearRegression()
slr.fit(X, y)
plr.fit(X2, y)

y_prd_slr = slr.predict(X)
y_prd_plr = plr.predict(X2)

print(slr.score(X, y))
print(plr.score(X2, y))

fig, ax = plt.subplots()

ax.scatter(X, y, label='Data Points', color='black', s=30, marker='o')
ax.plot(X, y_prd_slr, color='blue', linewidth=3, label='Fitted line')
ax.plot(X, y_prd_plr, color='green', linewidth=3, label=f'Polynomial degree {poly_dg}')
#ax.vlines(X, y, slr.predict(X), colors='red', linestyles='--', label='residuals')
ax.grid(True)
ax.legend(loc=0)
plt.show()
