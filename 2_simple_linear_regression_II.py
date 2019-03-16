import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def f(x):
    # return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2) + 0.025 * np.random.randn(len(x), 1)
    return np.cos(1.5 * np.pi * x) + 0.0 * np.random.randn(len(x), 1)


n_points = 100
x = np.linspace(-2, 2, n_points)
indices = np.random.permutation(x.shape[0])
train_size = int(n_points * 0.6)
training_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = x[training_idx], x[test_idx]
X_train, X_test = np.sort(X_train), np.sort(X_test)

X_train, X_test = X_train[:, np.newaxis], X_test[:, np.newaxis]
y_train, y_test = f(X_train), f(X_test)

poly_dg = 4
poly_transformer = PolynomialFeatures(poly_dg)
X_train_transformed = poly_transformer.fit_transform(X_train)
X_test_transformed = poly_transformer.transform(X_test)

# fit the model
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)

poly_model = LinearRegression()
poly_model.fit(X_train_transformed, y_train)

y_predict_simple = simple_model.predict(X_test)
y_predict_poly = poly_model.predict(X_test_transformed)

print('Simple model: R^2={0:.2f} MSE={1:.2f} MAE={2:.2f}'.format(r2_score(y_test, y_predict_simple) * 100,
                                                                 mean_squared_error(y_test, y_predict_simple),
                                                                 mean_absolute_error(y_test, y_predict_simple)))

print('Polynomial model: R^2={0:.2f} MSE={1:.2f} MAE={2:.2f}'.format(r2_score(y_test, y_predict_poly) * 100,
                                                                     mean_squared_error(y_test, y_predict_poly),
                                                                     mean_absolute_error(y_test, y_predict_poly)))

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, label='Test Data', color='blue', s=30, marker='s', facecolors='white')
ax.scatter(X_train, y_train, label='Train data', color='red', s=30, marker='o')

# ax.plot(X_train, simple_model.predict(X_train), color='black', ls='--', linewidth=1,
#         label='Simple linear fit train data')
ax.plot(X_train, poly_model.predict(X_train_transformed), color='green', ls='-', linewidth=2,
        label=f'Poly fit with n={poly_dg} train data')

# ax.plot(X_test, y_predict_simple, color='#ff9114', linewidth=3, label='Simple linear fit test data')
ax.plot(X_test, y_predict_poly, color='green', linewidth=3, label=f'Poly fit with n={poly_dg} test data')

# ax.grid(True)
ax.legend(loc=0)
plt.show()
