import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Wczytaj dane z pliku
data = pd.read_csv("zaj6/data6.tsv", delimiter="\t", header=None, names=["x", "y"])

# Zakładam, że dane mają dwie kolumny - x i y
X = data[["x"]]
y = data["y"]

# Stwórz modele regresji liniowej i wielomianowej
linear_model = LinearRegression()
quadratic_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
fifth_degree_model = make_pipeline(PolynomialFeatures(5), LinearRegression())

# Dopasuj modele do danych
linear_model.fit(X, y)
quadratic_model.fit(X, y)
fifth_degree_model.fit(X, y)

# Przygotuj dane do wykresu
X_fit = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)

# Predykcje modeli
y_linear = linear_model.predict(X_fit)
y_quadratic = quadratic_model.predict(X_fit)
y_fifth_degree = fifth_degree_model.predict(X_fit)

# Wykres
plt.scatter(X, y, label="Dane")
plt.plot(X_fit, y_linear, label="Regresja liniowa")
plt.plot(X_fit, y_quadratic, label="Regresja kwadratowa")
plt.plot(X_fit, y_fifth_degree, label="Regresja 5. stopnia")
plt.legend()
plt.show()
