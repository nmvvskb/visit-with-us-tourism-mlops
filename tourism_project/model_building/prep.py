
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/nmvvskb/visit-with-us-tourism-mlops/tourism.csv"
df = pd.read_csv(DATASET_PATH, index_col=0)
print("Dataset loaded successfully.")

# copying data to another variable to avoid any changes to original data
data = df.copy()

# checking shape of the given data
print(f"Shape of the data is {data.shape}")

# checking for null values
print(f"Total null values in dataset: {data.isnull().sum().sum()}")


# checking for duplicate values
print(f"There is no duplicate values in the data: {data.duplicated().sum()}")

print("Among all columns 'TypeofContact, Occupation, Gender, ProductPitched, MaritalStatus, Designation' columns are categorical, remaining are numerical")

print("\n +++++++++++++++ \n")
print("We can remove CustomerID columns as all rows are unique")

#Dropping CustomerID column from dataset
data.drop(["CustomerID"], axis=1, inplace=True)

# Define the target variable for the classification task
target_col = 'ProdTaken'

# doing labeled encode for Occupation
occupation_map = {
    "Salaried": 1,
    "Free Lancer": 2,
    "Small Business": 3,
    "Large Business": 4
}
data['Occupation'] = data['Occupation'].map(occupation_map)


# doing labeled encode for MaritalStatus
maritalstatus_map = {
    "Single": 1,
    "Married": 2,
    "Divorced": 3,
    "Unmarried": 4
}
data['MaritalStatus'] = data['MaritalStatus'].map(maritalstatus_map)


# List of categorical features in the dataset
label_encode_col= [
    'TypeofContact',         # The method by which the customer was contacted(Company Invited or Self Inquiry)
    'Gender',                # Gender of the customer (Male, Female).
    'ProductPitched',        # The type of product pitched to the customer.
    'Designation',           # Customer's designation in their current organization.
]

# Encoding the categorical columns by using LabelEncoder
for col in label_encode_col:
  label_encoder = LabelEncoder()
  data[col] = label_encoder.fit_transform(data[col])


# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)


Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="nmvvskb/visit-with-us-tourism-mlops",
        repo_type="dataset",
    )
