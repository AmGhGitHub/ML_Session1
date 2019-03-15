import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df_values = pd.read_csv('data/linear_model_data.csv', header=[0])
df_values_train = df_values.iloc[0::2, ]
df_values_test = df_values.iloc[1::2, ]

X_train = df_values_train.iloc[:, 0].values
y_train = df_values_train.iloc[:, 1].values
X_train = X_train[:, np.newaxis]
y_train = y_train[:, np.newaxis]

X_test = df_values_test.iloc[:, 0].values
y_test = df_values_test.iloc[:, 1].values
X_test = X_test[:, np.newaxis]
y_test = y_test[:, np.newaxis]

poly_dg = 3
poly_transf = PolynomialFeatures(degree=poly_dg)
X_train_transformed = poly_transf.fit_transform(X_train)
X_test_transformed = poly_transf.transform(X_test)

# fit the model
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)
poly_model = LinearRegression()
poly_model.fit(X_train_transformed, y_train)

y_predict_simple = simple_model.predict(X_test)
y_predict_poly = poly_model.predict(X_test_transformed)

print(f'Simple model:           R^2={r2_score(y_test, y_predict_simple) * 100:.2f}% -- MSE={mean_squared_error(y_test, y_predict_simple) :.2f} -- MAE={mean_absolute_error(y_test, y_predict_simple) :.2f}')
print(f'Polynomial model (n={poly_dg}): R^2={r2_score(y_test, y_predict_poly) * 100:.2f}% -- MSE={mean_squared_error(y_test, y_predict_poly):.2f} -- MAE={mean_absolute_error(y_test, y_predict_poly):.2f}')

fig, ax = plt.subplots()

ax.plot(X_train, simple_model.predict(X_train), color='black', ls='--', linewidth=1,
        label='Simple linear fit train data')
ax.plot(X_train, poly_model.predict(X_train_transformed), color='green', ls='--', linewidth=1,
        label=f'Poly fit with n={poly_dg} train data')
ax.scatter(X_test, y_test, label='Test Data', color='blue', s=30, marker='s')
ax.scatter(X_train, y_train, label='Train data', color='black', s=30, marker='o')
ax.plot(X_test, y_predict_simple, color='#ff9114', linewidth=3, label='Simple linear fit test data')
ax.plot(X_test, y_predict_poly, color='green', linewidth=3, label=f'Poly fit with n={poly_dg} test data')

ax.grid(True)
ax.legend(loc=0)
plt.show()
