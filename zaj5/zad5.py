import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

all_data = pd.read_csv(
    "titanic.tsv",
    header=0,
    sep="\t",
)
copy_all_data = all_data

# swap Cabin column with 0 for nan and 1 for other
all_data["Cabin"] = all_data["Cabin"].apply(lambda x: 0 if pd.isnull(x) else 1)
# change name from Cabin to HasCabin
all_data.rename(columns={"Cabin": "HasCabin"}, inplace=True)

# swap male and female in sex column for 1 and 0
all_data["Sex"] = all_data["Sex"].map({"male": 1, "female": 0})

# fill age with mean
all_data["Age"].fillna(all_data["Age"].mean(), inplace=True)

# drop nan values in Embarked
all_data.dropna(subset=["Embarked"], inplace=True)

# create new columns from Embarked
all_data = pd.get_dummies(all_data, columns=["Embarked"], dtype=int)

# Scrape title from name
all_data["Title"] = all_data["Name"].apply(
    lambda x: x.split("\t ")[1].split(".")[0] + "."
)
# combine all titles that are less than 3 into other
title_counts = all_data["Title"].value_counts()
less_than_3_titles = title_counts[title_counts < 10].index
all_data.loc[all_data["Title"].isin(less_than_3_titles), "Title"] = "Other"
# drop name column
all_data.drop("Name", axis=1, inplace=True)
# Create new columns for each title type
all_data = pd.get_dummies(all_data, columns=["Title"], dtype=int)

# separate ticket number and ticket string into separate columns
all_data["TicketNumber"] = all_data["Ticket"].apply(lambda x: x.split()[-1])
all_data["TicketString"] = all_data["Ticket"].apply(
    lambda x: " ".join(x.split()[:-1]) if len(x.split()) > 1 else np.nan
)

# swap ticket string with 0 for nan and 1 for other
all_data["TicketString"] = all_data["TicketString"].apply(
    lambda x: 0 if pd.isna(x) else 1
)

all_data.rename(columns={"TicketString": "TicketHasPort"}, inplace=True)
all_data.drop("Ticket", axis=1, inplace=True)

# drop row with "LINE" as TicketNumber
all_data = all_data[all_data["TicketNumber"] != "LINE"]
# change ticket number values form string to int
all_data["TicketNumber"] = all_data["TicketNumber"].astype(int)

# drop unwanted columns
all_data.drop(["TicketNumber"], axis=1, inplace=True)
all_data.drop(["TicketHasPort"], axis=1, inplace=True)

print(all_data)
# save all_data as csv
all_data.to_csv("all_data.csv", index=False)

# Split data into train and test
data_train, data_test = train_test_split(all_data, test_size=0.1)

# Train model
FEATURES = all_data.columns[1:]
y_train = pd.DataFrame(data_train["Survived"])
y_train = np.ravel(y_train)
x_train = pd.DataFrame(data_train[FEATURES])
model = LogisticRegression(max_iter=100000)


# Test model
model.fit(x_train, y_train)
y_expected = pd.DataFrame(data_test["Survived"])
x_test = pd.DataFrame(data_test[FEATURES])
y_predicted = model.predict(x_test)

# evaluate model
precision, recall, fscore, _ = precision_recall_fscore_support(
    y_expected, y_predicted, pos_label=1, average="binary"
)

print(f"Precision:{precision}")
print(f"Recall:{recall}")
print(f"F-score:{fscore}")


score = model.score(x_test, y_expected)

print(f"Model score: {score}")

# Train model from all data for cross validation
FEATURES = all_data.columns[1:]
y_train = pd.DataFrame(all_data["Survived"])
y_train = np.ravel(y_train)
x_train = pd.DataFrame(all_data[FEATURES])
model = LogisticRegression(max_iter=100000)

# cross validate model
print("\nCross validation:")
out = cross_validate(
    model, x_train, y_train, cv=10, scoring=("accuracy", "precision", "recall", "f1")
)
print(f"Accuracy: {out.get('test_accuracy').mean()}")
print(f"Precision: {out.get('test_precision').mean()}")
print(f"Recall: {out.get('test_recall').mean()}")
print(f"F-score: {out.get('test_f1').mean()}")
