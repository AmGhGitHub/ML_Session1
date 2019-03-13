import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df_capita = pd.read_csv('data/canada_per_capita_income.csv', header=[0])
X = df_capita.iloc[:, 0].values
y = df_capita.iloc[:, 1].values
poly_dg = 5

y_prd_np = np.poly1d(np.polyfit(X, y, poly_dg))(X)

X = X[:, np.newaxis]
y = y[:, np.newaxis]

X_trnf = PolynomialFeatures(degree=poly_dg).fit_transform(X)

simple_lr = LinearRegression()
polynomial_lr = LinearRegression(fit_intercept=False, normalize=False)
simple_lr.fit(X, y)
polynomial_lr.fit(X_trnf, y)

print(simple_lr.coef_)
print(polynomial_lr.coef_)

y_prd_slr = simple_lr.predict(X)
y_prd_plr = polynomial_lr.predict(X_trnf)

print(simple_lr.score(X, y))
print(polynomial_lr.score(X_trnf, y))

fig, ax = plt.subplots()

ax.scatter(X, y, label='Data Points', color='black', s=30, marker='o')
ax.plot(X, y_prd_slr, color='blue', linewidth=3, label='Simple Linear Fit')
ax.plot(X, y_prd_np, color='purple', label=f'NP Poly Fit with n={poly_dg}')
ax.plot(X, y_prd_plr, color='green', linewidth=3, label=f'SK Poly fit with n={poly_dg}')
# ax.vlines(X, y, simple_lr.predict(X), colors='red', linestyles='--', label='residuals')
ax.grid(True)
ax.legend(loc=0)
plt.show()
