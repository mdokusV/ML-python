from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

NUM_FOLDS = 5

all_data = pd.read_csv(
    "communities.data",
    header=None,
    sep=",",
)

# drop columns with "?"
all_data.drop(all_data.columns[all_data.isin(["?"]).any()], axis=1, inplace=True)
all_data.drop(all_data.columns[1], axis=1, inplace=True)

scaler = StandardScaler()

# scale and normalize data
scaled_data = pd.DataFrame(
    data=scaler.fit_transform(all_data), columns=all_data.columns
)
x_param = scaled_data[scaled_data.columns[:-1]]
y_param = scaled_data[scaled_data.columns[-1]]

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)


model_with_ridge = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=40.0))
score_ridge = -cross_val_score(
    model_with_ridge,
    x_param,
    y_param,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    verbose=3,
)

print(f"RMSE with ridge: {score_ridge.mean()} +/- {score_ridge.std()}\n")


model_without_ridge = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
score_no_ridge = -cross_val_score(
    model_without_ridge,
    x_param,
    y_param,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    verbose=3,
)


print(f"RMSE without ridge: {score_no_ridge.mean()} +/- {score_no_ridge.std()}\n")


column_index: int = 0
unscaled = scaler.inverse_transform(scaled_data)
assert isinstance(unscaled, np.ndarray)
unscaled_data = pd.DataFrame(unscaled, columns=all_data.columns)
