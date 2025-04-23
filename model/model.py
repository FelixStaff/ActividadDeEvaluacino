import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt

class RegressionModel:

    # [CLASS]: Perform linear regression on categorical data to predict ModifiedDate
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.features = ['NameStyle', 'Title', 'FirstName', 'MiddleName', 'LastName',
                         'Suffix', 'CompanyName', 'SalesPerson', 'EmailAddress', 'Phone']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), self.features)]
        )

    # [Method]: Preprocess the data
    def preprocess(self):
        print("Starting preprocessing...")

        # Convert ModifiedDate to years since the earliest date
        self.df['ModifiedDate'] = pd.to_datetime(self.df['ModifiedDate'])
        self.df['ModifiedDate'] = self.df['ModifiedDate'].astype('int64')
        self.df['ModifiedDate'] = (
            (self.df['ModifiedDate'] - self.df['ModifiedDate'].min()) / 1e9 / 60 / 60 / 24 / 365.25
        )

        # Fill NaNs
        self.df[self.features] = self.df[self.features].fillna('Missing')

        X = self.df[self.features]
        y = self.df['ModifiedDate']

        # Build pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', LinearRegression())
        ])

        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Preprocessing completed.")

    # [Method]: Train the model
    def train(self):
        print("Training model...")
        self.model.fit(self.X_train, self.y_train)
        print("Training completed.")

    # [Method]: Evaluate the model
    def evaluate(self):
        print("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = self.model.score(self.X_test, self.y_test)
        print("MSE:", mse)
        print("R²:", r2)
        return mse, r2

    # [Method]: Validate the model on new external data
    def validate(self, new_data: pd.DataFrame):
        print("Validating model with new data...")

        new_data = new_data.copy()
        new_data[self.features] = new_data[self.features].fillna('Missing')
        new_data['ModifiedDate'] = pd.to_datetime(new_data['ModifiedDate'])
        new_data['ModifiedDate'] = new_data['ModifiedDate'].astype('int64')
        new_data['ModifiedDate'] = (
            (new_data['ModifiedDate'] - new_data['ModifiedDate'].min()) / 1e9 / 60 / 60 / 24 / 365.25
        )

        X_new = new_data[self.features]
        y_new = new_data['ModifiedDate']

        y_pred = self.model.predict(X_new)
        mse = mean_squared_error(y_new, y_pred)
        r2 = self.model.score(X_new, y_new)
        print("Validation MSE:", mse)
        print("Validation R²:", r2)
        return mse, r2

    # [Method]: Save model to pickle file
    def save_model(self, path: str = "regression_model.pkl"):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    # [Method]: Visualize the prediction of each user
    def visualize(self, new_data: pd.DataFrame = None, use_test: bool = True):
        print("Visualizing predictions...")
        if use_test:
            X = self.X_test
            y = self.y_test
        elif new_data is not None:
            new_data = new_data.copy()
            new_data[self.features] = new_data[self.features].fillna('Missing')
            X = new_data[self.features]
            y = new_data['ModifiedDate']
        else:
            X = self.X_train
            y = self.y_train
        # Convert to DataFrame
        y_pred = self.model.predict(X)
        predictions = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        })
        predictions.sort_values(by='Actual', inplace=True)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions['Actual'].values, label='Actual', marker='o')
        plt.plot(predictions['Predicted'].values, label='Predicted', marker='x')
        plt.title('Actual vs Predicted ModifiedDate')
        plt.xlabel('Sample Index')
        plt.ylabel('ModifiedDate (Years since earliest date)')
        plt.legend()
        plt.show()