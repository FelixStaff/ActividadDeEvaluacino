# import the libraries to connect to the databaseimport pyodbc
#YOUR_PASSWORD = json.load(contraseña) #json es sólo el string de su contraseña.
#contraseña.close()
# import json
import json
import pyodbc
import os
import pandas as pd

config = "load_data/config.json"
__server_info__ = {
    'server': "whitedog.database.windows.net",
    'database': "BlackCat"
}

class Config:

    # [CLASS]: Read and stablish the connection to the database
    def __init__(self, config_path, server_info = __server_info__):
        self.config_path = config_path
        # Read the configuration file and extract the user, password 
        with open(config_path, 'r') as file:
            config_data = json.load(file)
            self.user = config_data['user']
            self.password = config_data['password']

        # Set up the connection string
        self.conn_str = (
            f'DRIVER={{ODBC Driver 18 for SQL Server}};'
            f'SERVER={server_info["server"]};'
            f'DATABASE={server_info["database"]};'
            f'UID={self.user};'
            f'PWD={self.password}'
        )
        # self.conn = None
        self.conn = None

        self.file_folder = "load_data/temp_data/"
        # Save if the output.csv exist
        self.located = False if os.path.exists(self.file_folder + "output.csv") else True
        if self.located:
            print("output.csv file not found. Creating a new one.")
        

    # [Method]: Connect to the database
    def connect(self):
        try:
            self.conn = pyodbc.connect(self.conn_str)
            print("Connected to the Azure SQL Database successfully!")
            return self.conn
        except pyodbc.Error as e:
            print(f"Error connecting to the database: {e}")
            return None
        
    # [Method]: Make the query to the database
    def query(self, sql_query : str, use_local : bool = False) -> pd.DataFrame:
        try:

            if use_local:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(self.file_folder + "output.csv")
                print("Data loaded from local CSV file.")
                return df
            # Execute the SQL query if not using local
            cursor = self.conn.cursor()
            # Connect from pandas
            df = pd.read_sql(sql_query, self.conn)
            # return the dataframe
            return df
        except pyodbc.Error as e:
            print(f"Error executing query: {e}")
            return None
        
    # [Method]: Close the connection to the database
    def close(self):
        if self.conn:
            self.conn.close()
            print("Connection closed.")
        else:
            print("No connection to close.")

    # [Method]: Save data from query to a CSV file
    def save_to_csv(self, sql_query: str, file_path: str = "output.csv"):
        try:
            df = self.query(sql_query)
            if df is not None:
                df.to_csv(self.file_folder + file_path, index=False)
                print(f"Data saved to {self.file_folder + file_path} successfully!")
            else:
                print("No data to save.")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")