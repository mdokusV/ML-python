import os
import numpy as np
import pandas as pd

from icecream import ic
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report


from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.pipeline import make_pipeline


def f1_score(y_true, y_pred):
    """
    Function to calculate the F1 score.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1


RELOAD = True
SPECIAL_OUTPUT_VERBOSE = False
RANDOM_STATE = 12

part_one_data = pd.read_csv(
    "train.csv",
    header=0,
)
part_two_data_no_survived = pd.read_csv(
    "test.csv",
    header=0,
)
part_two_data_only_survived = pd.read_csv(
    "gender_submission.csv",
    header=0,
)
part_two_data_no_survived.insert(0, "Survived", part_two_data_only_survived["Survived"])

all_data = pd.concat([part_one_data, part_two_data_no_survived])
copy_all_data = all_data
ic(copy_all_data.shape[0])


def preprocess(all_data: pd.DataFrame) -> pd.DataFrame:
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
        lambda x: x.split(" ")[1].split(".")[0] + "."
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
    all_data.to_csv("all_data.csv", index=False)
    all_data.dropna(inplace=True)
    return all_data


all_data = preprocess(all_data)
ic(all_data.shape[0])

# Split data into train and test
y_vector = pd.DataFrame(all_data["Survived"])
x_vector = pd.DataFrame(all_data.drop("Survived", axis=1))

# scale and normalize data
scaler = StandardScaler()
scaled_data = pd.DataFrame(
    data=scaler.fit_transform(x_vector), columns=x_vector.columns
)

X_train, X_test, y_train, y_test = train_test_split(
    x_vector,
    y_vector,
    test_size=0.2,
    random_state=RANDOM_STATE,
)
X_train.drop("PassengerId", axis=1, inplace=True)

PassengerId = pd.DataFrame(X_test["PassengerId"])
X_test.drop("PassengerId", axis=1, inplace=True)

ic(X_train.shape[0], X_test.shape[0])


# Prepare model
model = keras.Sequential(
    [
        layers.Dense(2**6, activation="relu", input_shape=[len(X_train.columns)]),
        layers.Dense(2**4, activation="relu", input_shape=[len(X_train.columns)]),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()


# Compile model
model_checkpoint = ModelCheckpoint(
    "best_model_keras.h5",
    monitor="val_f1_score",
    save_best_only=True,
    mode="max",
    verbose=1,
)
optimizer = keras.optimizers.Adam()
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["Accuracy", "Precision", "Recall", f1_score],
)


if os.path.exists("best_model_keras.h5") and not RELOAD:
    best_model_keras = keras.models.load_model(
        "best_model_keras.h5", custom_objects={"f1_score": f1_score}
    )
    ic("Loaded best model from checkpoint.")
else:
    ic("No saved model found. Training a new model.")
    epochs = 200
    batch_size = 2**5
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[model_checkpoint],
    )

    best_model_keras = keras.models.load_model(
        "best_model_keras.h5", custom_objects={"f1_score": f1_score}
    )


assert isinstance(best_model_keras, keras.Model)

# Evaluate the model on the test set
test_results = best_model_keras.evaluate(X_test, y_test)

ic("Test Results:")
ic(f"F1-score: {test_results[4]}")
ic(f"Accuracy: {test_results[1]}")
ic(f"Precision: {test_results[2]}")
ic(f"Recall: {test_results[3]}")

if SPECIAL_OUTPUT_VERBOSE:
    # Get predictions on the test set
    y_test_pred = best_model_keras.predict(X_test)

    # Round the predictions to convert them to binary (0 or 1)
    y_test_pred_binary = np.round(y_test_pred).astype(int)

    # Create a DataFrame for test results
    test_results_df = pd.DataFrame(
        {
            "Survived": y_test["Survived"].values,
            "Predicted": y_test_pred_binary.flatten(),
            "Raw": y_test_pred.flatten(),
        }
    )

    # Add an index column to X_test for joining
    X_test["PassengerId"] = PassengerId
    X_test_with_index = X_test.reset_index(drop=True)
    X_test_with_index = X_test_with_index["PassengerId"]
    X_test.drop("PassengerId", axis=1, inplace=True)

    # Join X_test with the test results DataFrame on the index
    joined_test_data = pd.concat([X_test_with_index, test_results_df], axis=1)
    all_data_with_results = pd.merge(
        copy_all_data, joined_test_data, left_on="PassengerId", right_on="PassengerId"
    )
    all_data_with_results = all_data_with_results.drop(["Survived_x"], axis=1)

    # ic the joined DataFrame
    ic(all_data_with_results.head(20))


##############################################################################
ic("RandomForest Classifier")

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=1_000,
    random_state=RANDOM_STATE,
    verbose=1,
    n_jobs=2,
)

clf.fit(
    X_train,
    y_train,
)
y_pred = clf.predict(X_test)

# Display classification report for more detailed metrics
print("\n")
ic(classification_report(y_test, y_pred, digits=4))


##############################################################################
ic("Logistic Regression")

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    verbose=1,
    max_iter=100_000,
    solver="lbfgs",
)

logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

print("\n")
ic(classification_report(y_test, y_pred, digits=4))
