import json
import joblib
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    # Cargar el modelo que incluye el pipeline con preprocesamiento
    model_path = Model.get_model_path('modelreg')  # nombre registrado en Azure
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Convertir JSON a DataFrame
        input_json = json.loads(raw_data)

        # Asegura que 'data' exista y sea una lista
        if isinstance(input_json, dict) and "data" in input_json:
            df = pd.DataFrame(input_json["data"])
        else:
            return json.dumps({"error": "Input must be a JSON with a 'data' key containing a list of records."})

        # Hacer predicción con el pipeline cargado
        predictions = model.predict(df)

        # Retornar los resultados como JSON
        return json.dumps({"result": predictions.tolist()})
    
    except Exception as e:
        return json.dumps({"error": str(e)})

