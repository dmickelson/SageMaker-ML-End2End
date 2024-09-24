import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# This preprocessing script is passed in to the processing step for running on the input data
# The training step then uses the preprocessed training features and labels to train a model.
# The evaluation step uses the trained model and preprocessed test features and labels to evaluate the model
# The script uses scikit-learn to do the following:
#  - Fill in missing sex categorical data and encode it so it's suitable for training.
#  - Scale and normalize all numerical fields except for rings and sex.
#  - Split the data into training, test, and validation datasets.

# Define feature and label column names and data types
# Because this is a headerless CSV file, specify the column names here.

feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/abalone-dataset.csv",
        header=None, 
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)
    )
    # Sets up preprocessing pipelines:
    # - For numeric features: imputation (filling missing values with median) and scaling
    # - For categorical features (sex): imputation (filling missing values with \"missing\") and one-hot encoding,

    numeric_features = list(feature_columns_names)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combines these pipelines \n",
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    # Separates the target variable (\"rings\") from the features.
    y = df.pop("rings")
    # Applies the preprocessing to the features.
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)
    # Concatenates the preprocessed features with the target variable
    X = np.concatenate((y_pre, X_pre), axis=1)
    # Shuffles the data randomly.
    np.random.shuffle(X)
    # Splits the data into training (70%), validation (15%), and test (15%) sets.
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])

    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)