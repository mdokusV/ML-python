from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans

N_CLUSTERS = 5


all_data = pd.read_csv(
    "flats_for_clustering.tsv",
    sep="\t",
)

# preprocess data

# change "parter" to 0
all_data.replace("parter", "0", inplace=True)
# replace "poddasze" and "niski parter" with nan
all_data.replace(["poddasze", "niski parter"], np.nan, inplace=True)
# drop rows with nan
all_data.dropna(inplace=True)


# scale and normalize data

scaler = StandardScaler()
scaled_data = pd.DataFrame(
    data=scaler.fit_transform(all_data), columns=all_data.columns
)


# apply kmeans
kmeans = KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(scaled_data)

# show pairplot with coloring
data_with_coloring = pd.DataFrame(scaled_data, columns=scaled_data.columns)
data_with_coloring["cluster"] = kmeans.labels_
sns.pairplot(data_with_coloring, hue="cluster", palette="viridis")


# calculate PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# show PCA with K-Means Clustering Labels
data_with_coloring = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
data_with_coloring["cluster"] = kmeans.labels_
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="PC1", y="PC2", data=data_with_coloring, hue="cluster", palette="viridis", s=50
)
plt.title("PCA with K-Means Clustering Labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()
