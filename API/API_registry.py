import json
import joblib
import pandas as pd
from azureml.core import Workspace, Model, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
import requests

class AzureMLModelManager:
    def __init__(self, workspace_name: str, subscription_id: str, resource_group: str, location: str):
        self.workspace_name = workspace_name
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.location = location
        self.workspace = None
        self.registered_model = None
        self.service = None
        self.scoring_uri = None

    def connect_to_workspace(self, my_id_file: str):
        # Load subscription ID from file
        with open(my_id_file, 'r') as id_file:
            mi = json.load(id_file)

        # Connect to Azure ML Workspace
        self.workspace = Workspace.create(
            name=self.workspace_name,
            subscription_id=mi["my_id"],
            resource_group=self.resource_group,
            location=self.location
        )
        print(f"Connected to workspace: {self.workspace_name}")

    def register_model(self, model_path: str, model_name: str):
        # Register the model to the workspace
        self.registered_model = Model.register(
            model_path=model_path,
            model_name=model_name,
            workspace=self.workspace
        )
        print(f"Model '{model_name}' registered successfully.")

    def create_inference_environment(self, env_name: str):
        # Create environment for inference
        virtual_env = Environment(env_name)
        conda_dep = CondaDependencies.create(conda_packages=['pandas', 'scikit-learn'])
        virtual_env.python.conda_dependencies = conda_dep
        return virtual_env

    def create_inference_config(self, entry_script: str, environment: Environment):
        # Create inference configuration
        return InferenceConfig(
            environment=environment,
            entry_script=entry_script
        )

    def deploy_service(self, inference_config: InferenceConfig, aci_config: AciWebservice.deploy_configuration):
        # Deploy the model as a web service
        self.service = Model.deploy(
            workspace=self.workspace,
            name='real-estates-service',
            models=[self.registered_model],
            inference_config=inference_config,
            deployment_config=aci_config,
            overwrite=True
        )
        self.service.wait_for_deployment()
        self.scoring_uri = self.service.scoring_uri
        print(f"Service deployed successfully. Scoring URI: {self.scoring_uri}")

    def score_model(self, input_data: dict):
        # Make a prediction via the deployed service
        input_json = json.dumps({"data": [input_data]})
        headers = {"Content-Type": "application/json"}
        
        # Call the service
        response = requests.post(self.scoring_uri, data=input_json, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: {response.text}")
            return None

    def save_score_script(self, score_script_path: str):
        # Save the score script to a file
        print(f"Score script already saved in {score_script_path}.")


if __name__ == "__main__":
    # Crear el administrador del modelo
    model_manager = AzureMLModelManager(
        workspace_name="hw1_car_pricing",
        subscription_id="my_subscription_id",  # Esto será cargado desde el archivo 'my_id.json'
        resource_group="__hw1__",
        location="eastus"
    )

    # Conectar al workspace de Azure
    model_manager.connect_to_workspace(my_id_file="my_id.json")

    # Registrar el modelo
    model_manager.register_model(model_path="linear_regressor.pkl", model_name="housing_regressor")

    # Crear el entorno de inferencia
    virtual_env = model_manager.create_inference_environment(env_name="env-4-housing")

    # Ruta al script de scoring que ya está guardado en API/score.py
    score_script_path = "API/score.py"

    # Crear configuración de inferencia
    inference_config = model_manager.create_inference_config(entry_script=score_script_path, environment=virtual_env)

    # Crear configuración de despliegue
    aci_config = AciWebservice.deploy_configuration(cpu_cores=0.5, memory_gb=1)

    # Desplegar el servicio de inferencia
    model_manager.deploy_service(inference_config=inference_config, aci_config=aci_config)

    # Datos para hacer la predicción
    two_houses = {
        "longitude": [-230.1, -195.56],
        "latitude": [50.1, 65.3],
        "housing_median_age": [10, 20],
        "total_rooms": [798, 675],
        "total_bedrooms": [26, 79],
        "population": [435346, 346234],
        "households": [123, 4523],
        "median_income": [2.325345, 3.423424]
    }

    # Hacer la predicción
    result = model_manager.score_model(input_data=two_houses)

    # Mostrar el resultado
    if result:
        two_houses["median_house_value"] = result["result"]
        print(pd.DataFrame(two_houses))
