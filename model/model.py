import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, df: pd.DataFrame, features: list[str]):
        self.df = df.copy()
        self.features = features

    def transform_modified_date(self):
        self.df['ModifiedDate'] = pd.to_datetime(self.df['ModifiedDate'])
        self.df['ModifiedDate'] = self.df['ModifiedDate'].astype('int64')
        self.df['ModifiedDate'] = (
            (self.df['ModifiedDate'] - self.df['ModifiedDate'].min()) / 1e9 / 60 / 60 / 24 / 365.25
        )

    def fill_nans(self):
        self.df[self.features] = self.df[self.features].fillna('Missing')

    def process_and_split(self, test_size=0.2, random_state=42):
        self.transform_modified_date()
        self.fill_nans()
        X = self.df[self.features]
        y = self.df['ModifiedDate']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def prepare_external(self, new_data: pd.DataFrame):
        new_data = new_data.copy()
        new_data[self.features] = new_data[self.features].fillna('Missing')
        new_data['ModifiedDate'] = pd.to_datetime(new_data['ModifiedDate'])
        new_data['ModifiedDate'] = new_data['ModifiedDate'].astype('int64')
        new_data['ModifiedDate'] = (
            (new_data['ModifiedDate'] - new_data['ModifiedDate'].min()) / 1e9 / 60 / 60 / 24 / 365.25
        )
        X = new_data[self.features]
        y = new_data['ModifiedDate']
        return X, y


class RegressionModel:
    def __init__(self, features: list[str]):
        self.features = features
        self.model = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), self.features)]
            )),
            ('regressor', LinearRegression())
        ])

    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Training completed.")

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = self.model.score(X_test, y_test)
        print("MSE:", mse)
        print("R²:", r2)
        return mse, r2

    def validate(self, X_val, y_val):
        print("Validating model...")
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = self.model.score(X_val, y_val)
        print("Validation MSE:", mse)
        print("Validation R²:", r2)
        return mse, r2

    def save(self, path: str = "regression_model.pkl"):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def visualize(self, X, y):
        print("Visualizing predictions...")
        y_pred = self.model.predict(X)
        predictions = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        }).sort_values(by='Actual')

        plt.figure(figsize=(10, 6))
        plt.plot(predictions['Actual'].values, label='Actual', marker='o')
        plt.plot(predictions['Predicted'].values, label='Predicted', marker='x')
        plt.title('Actual vs Predicted ModifiedDate')
        plt.xlabel('Sample Index')
        plt.ylabel('ModifiedDate (Years since earliest date)')
        plt.legend()
        plt.tight_layout()
        plt.show()
