import csv
from email import header
import os
import pandas as pd
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)


def join_path(filename: str) -> str:
    return os.path.join(DIR, filename)


data = pd.read_csv(join_path("data2.csv"), header=None)

x = data.iloc[:, 1]
y = data.iloc[:, 3]

plt.plot(x, y, "o")
plt.show()
