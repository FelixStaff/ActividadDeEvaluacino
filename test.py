import json
import joblib
import numpy as np
import pandas as pd
import pickle



if __name__ == "__main__":
    # Load the model from the file
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Example input data
    input_data = {
        'NameStyle': ['Missing'],
        'Title': ['Missing'],
        'FirstName': ['John'],
        'MiddleName': ['Missing'],
        'LastName': ['Doe'],
        'Suffix': ['Missing'],
        'CompanyName': ['Missing'],
        'SalesPerson': ['Missing'],
        'EmailAddress': ['