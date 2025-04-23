# Imports for the pipeline
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


class Pipeline:

    # [Method]: Constructor of the class
    def __init__(self, features: list, target: str):
        self.features = features
        self.target = target
        self.model = None
        self.preprocessor = None

        # Create the pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), self.features)]
            )),
            ('regressor', LinearRegression())
        ])