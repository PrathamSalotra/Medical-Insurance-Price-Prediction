import os
from data_preprocessing import load_data, preprocess_data, split_data
from eda import eda
from feature_engineering import feature_engineering
from model_training import train_model
from model_evaluation import evaluate_model
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == "__main__":

    file_path = os.path.join('data', 'insurance.csv')
    

    data = load_data(file_path)
    

    eda(data)
    

    data = preprocess_data(data)
    

    data = feature_engineering(data)
    

    X_train, X_test, y_train, y_test = split_data(data)
    

    model = train_model(X_train, y_train)
    

    evaluate_model(model, X_test, y_test)


    scaler = StandardScaler().fit(X_train[['age', 'bmi', 'children']])
    joblib.dump(model, 'src/model.joblib')
    joblib.dump(scaler, 'src/scaler.joblib')
