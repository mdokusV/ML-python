import numpy as np
import pandas as pd


all_data = pd.read_csv(
    "flats_for_clustering.tsv",
    sep="\t",
)


# change "parter" to 0
all_data.replace("parter", "0", inplace=True)
# replace "poddasze" and "niski parter" with nan
all_data.replace(["poddasze", "niski parter"], np.nan, inplace=True)
# drop rows with nan
all_data.dropna(inplace=True)

