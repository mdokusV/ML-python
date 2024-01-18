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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Split data into train and test
FEATURES = all_data.columns[1:]
y_vector = torch.Tensor(all_data["Survived"].values).view(-1, 1)
x_vector = torch.Tensor(all_data[FEATURES].values)
X_train, X_test, y_train, y_test = train_test_split(x_vector, y_vector, test_size=0.1, random_state=42)

# Define the neural network model
class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 2**8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2**8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Initialize the model
model = CustomModel(len(FEATURES))
print(model)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Train the model
epochs = 50
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)

print("Test Results:")
print(f"Loss: {test_loss.item()}, Accuracy: {((test_outputs > 0.5).float() == y_test).float().mean()}")

# Convert outputs to binary predictions
y_test_pred = (test_outputs > 0.5).float()

# Create a DataFrame for test results
test_results_df = pd.DataFrame({
    'Survived': y_test.view(-1).numpy(),
    'Predicted': y_test_pred.view(-1).numpy(),
    'Raw': test_outputs.view(-1).numpy()
})

# Add an index column to X_test for joining
X_test_with_index = pd.DataFrame(X_test.numpy(), columns=FEATURES)
X_test_with_index = X_test_with_index['PassengerId']

# Join X_test with the test results DataFrame on the index
joined_test_data = pd.concat([X_test_with_index, test_results_df], axis=1)
all_data_with_results = pd.merge(copy_all_data, joined_test_data, left_on='PassengerId', right_on='PassengerId')
all_data_with_results = all_data_with_results.drop(['Survived_x'], axis=1)

# Print the joined DataFrame
print(all_data_with_results.head(10))
