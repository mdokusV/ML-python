import numpy as np
import pandas as pd

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
less_than_3_titles = title_counts[title_counts < 3].index
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


print(all_data)
# save all_data as csv
all_data.to_csv("all_data.csv", index=False)
