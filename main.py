from load_data.login import Config
from model.model import DataProcessor, RegressionModel  # Asegúrate de que estas clases estén en model/model.py

import pandas as pd
import pickle
import matplotlib.pyplot as plt

# [Main]: Main function to run the script
if __name__ == "__main__":
    # [===================PARA EL PROFESOR===================]
    validation_dir = "load_data/validation"
    # [===================PARA EL PROFESOR===================]

    # Cargar configuración
    config = Config("load_data/configs.json")
    conn = config.connect()

    # Consulta a la base de datos
    sql_query = "SELECT * FROM SalesLT.customer"

    if config.located:
        _ = config.query(sql_query, use_local=False)
        _ = config.save_to_csv(sql_query, "output.csv")
    results_df = config.query(sql_query, use_local=True)

    print(results_df.head())
    features = ['NameStyle', 'Title', 'FirstName', 'MiddleName', 'LastName',
                'Suffix', 'CompanyName', 'SalesPerson', 'EmailAddress', 'Phone']

    # Procesamiento
    processor = DataProcessor(results_df, features)
    X_train, X_test, y_train, y_test = processor.process_and_split()

    # Entrenamiento
    regression_model = RegressionModel(features)
    regression_model.train(X_train, y_train)

    # Evaluación
    regression_model.evaluate(X_test, y_test)

    # Guardar modelo sin colisiones de memoria
    regression_model.save("model.pkl")

    # Visualización
    regression_model.visualize(X_test, y_test)

    # Cerrar conexión
    config.close()
