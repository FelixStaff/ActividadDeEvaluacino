from load_data.login import Config
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
# Visualize the prediction of each user
import matplotlib.pyplot as plt
import numpy as np

from model.model import RegressionModel

# [Main]: Main function to run the script
if __name__ == "__main__":
    # Create an instance of the Config class
    config = Config("load_data/configs.json")
    
    # Connect to the database
    conn = config.connect()
    
    # Example SQL query to fetch data from a table
    sql_query = "SELECT * FROM SalesLT.customer"
    
    # Execute the query and fetch results
    # Save
    if config.located:
        results = config.query(sql_query, use_local=False)
        results = config.save_to_csv(sql_query, "output.csv")
    results = config.query(sql_query, use_local=True)
    
    # Print the results
    print (results.head())  # Display the first few rows of the DataFrame
    df = results.copy()

    # [PIPELINE & MODEL]: Create the pipeline and model
    model = RegressionModel(df)
    model.preprocess()
    model.train()
    # [PREDICTION]: Make predictions on the test set
    mse, r2 = model.evaluate()
    
    # [VISUALIZATION]: Visualize the predictions
    model.visualize(None)
    
    # Close the connection
    config.close()

        
