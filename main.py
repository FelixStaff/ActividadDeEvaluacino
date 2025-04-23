from load_data.login import Config


# [Main]: Main function to run the script
if __name__ == "__main__":
    # Create an instance of the Config class
    config = Config("load_data/configs.json")
    
    # Connect to the database
    conn = config.connect()
    
    # Example SQL query to fetch data from a table
    sql_query = "SELECT * FROM Sales.SalesOrderHeader"
    
    # Execute the query and fetch results
    results = config.query(sql_query)
    
    # Print the results
    if results:
        for row in results:
            print(row)
    
    # Close the connection
    config.close()