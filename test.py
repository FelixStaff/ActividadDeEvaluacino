import json
import joblib
import numpy as np
import pandas as pd
import pickle



if __name__ == "__main__":
    # Load the model from the file
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    #