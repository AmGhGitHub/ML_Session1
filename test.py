import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)


def calc_rmse_and_r2(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def convert_rowmat_to_colmat(x):
    return x[:, np.newaxis]


def generate_response(x):
    noise = np.random.normal(-2, 5, len(x)).reshape(-1, 1)
    return 0.5 * (x ** 3) - 2 * (x ** 2) + x + noise


def generate_datapoints(n_points, test_size):
    x = 2 - 3 * np.random.normal(0, 0.5, n_points)
    select_range = int((1.0 - test_size) * len(x))
    X_train = x[:select_range]
    X_test = x[select_range:]
    X_train = convert_rowmat_to_colmat(np.sort(X_train))
    X_test = convert_rowmat_to_colmat(np.sort(X_test))
    y_train = generate_response(X_train)
    y_test = generate_response(X_test)
    return X_train, X_test, y_train, y_test


# transforming the data to include another axis
X_train, X_test, y_train, y_test = generate_datapoints(50, 0.2)

poly_dg = 5
polynomial_features = PolynomialFeatures(degree=poly_dg)
X_train_trans = polynomial_features.fit_transform(X_train)
X_test_trans = polynomial_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_trans, y_train)

y_test_pred = model.predict(X_test_trans)
y_train_pred = model.predict(X_train_trans)

rmse_train, r2_train = calc_rmse_and_r2(y_train, y_train_pred)
rmse_test, r2_test = calc_rmse_and_r2(y_test, y_test_pred)

print(f'n={poly_dg}')
print(f'RMSE: Train={rmse_train:.2f} & Test={rmse_test:.2f}')
print(f'R2: Train={r2_train:.2f} & Test={r2_test:.2f}')

plt.scatter(X_train, y_train, s=10, c='#59c400', label='Train data')
plt.plot(X_train, model.predict(X_train_trans), color='#316b00', lw=3, label='Train fit')
plt.scatter(X_test, y_test, s=20, c='#0200e8', marker='s', label='Test data')
plt.plot(X_test, model.predict(X_test_trans), color='#01006b', label='Test fit', lw=2, ls='--')
plt.legend(loc=0)
plt.show()
