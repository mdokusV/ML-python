import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import keras
from keras import layers


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

# Split data into train and test
FEATURES = all_data.columns[1:]
y_vector = pd.DataFrame(all_data["Survived"])
x_vector = pd.DataFrame(all_data[FEATURES])
X_train, X_test, y_train, y_test = train_test_split(x_vector, y_vector, test_size=0.1, random_state=42)


# Prepare model
model = keras.Sequential(
    [
        layers.Dense(2**8, activation="relu", input_shape=[len(FEATURES)]),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])

# Train model
epochs = 50
batch_size = 16
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_results = model.evaluate(X_test, y_test)

print("Test Results:")
print(f"Loss: {test_results[0]}")
print(f"Accuracy: {test_results[1]}")
print(f"Precision: {test_results[2]}")
print(f"Recall: {test_results[3]}")

# Get predictions on the test set
y_test_pred = model.predict(X_test)

# Round the predictions to convert them to binary (0 or 1)
y_test_pred_binary = np.round(y_test_pred).astype(int)

# Create a DataFrame for test results
test_results_df = pd.DataFrame({
    'Survived': y_test['Survived'].values,
    'Predicted': y_test_pred_binary.flatten(),
    'Raw': y_test_pred.flatten()
})

# Add an index column to X_test for joining
X_test_with_index = X_test.reset_index(drop=True)
X_test_with_index = X_test_with_index['PassengerId']

# Join X_test with the test results DataFrame on the index
joined_test_data = pd.concat([X_test_with_index, test_results_df], axis=1)
all_data_with_results = pd.merge(copy_all_data, joined_test_data, left_on='PassengerId', right_on='PassengerId')
all_data_with_results = all_data_with_results.drop(['Survived_x'], axis=1)


# Print the joined DataFrame
print(all_data_with_results.head(10))