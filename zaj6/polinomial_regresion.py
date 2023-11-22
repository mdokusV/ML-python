import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

data = pd.read_csv("data6.tsv", delimiter="\t", header=None, names=["x", "y"])

x = data[["x"]]
y = data["y"]

linear_model = make_pipeline(PolynomialFeatures(1), LinearRegression())
quadratic_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
fifth_degree_model = make_pipeline(PolynomialFeatures(5), LinearRegression())

linear_model.fit(x, y)
quadratic_model.fit(x, y)
fifth_degree_model.fit(x, y)

X_fit = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)  

y_linear = linear_model.predict(X_fit)
y_quadratic = quadratic_model.predict(X_fit)
y_fifth_degree = fifth_degree_model.predict(X_fit)

plt.scatter(x, y, label="Dane")
plt.plot(X_fit, y_linear, label="Regresja liniowa")
plt.plot(X_fit, y_quadratic, label="Regresja kwadratowa")
plt.plot(X_fit, y_fifth_degree, label="Regresja 5. stopnia")
plt.legend()
plt.show()
