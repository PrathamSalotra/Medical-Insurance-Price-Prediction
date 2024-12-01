import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd

def get_user_input():
    def predict():
        age = int(age_entry.get())
        sex = sex_var.get()
        bmi = float(bmi_entry.get())
        children = int(children_entry.get())
        smoker = smoker_var.get()
        region = region_var.get()
        

        user_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex_male': [1 if sex == 'male' else 0],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northwest': [1 if region == 'northwest' else 0],
            'region_southeast': [1 if region == 'southeast' else 0],
            'region_southwest': [1 if region == 'southwest' else 0]
        })
        

        numerical_features = ['age', 'bmi', 'children']
        user_data[numerical_features] = scaler.transform(user_data[numerical_features])
        

        prediction = model.predict(user_data)[0]
        result_label.config(text=f'Predicted Insurance Charge: ${prediction:.2f}')
    

    model = joblib.load('src/model.joblib')
    scaler = joblib.load('src/scaler.joblib')
    
    root = tk.Tk()
    root.title("Insurance Price Prediction")
    

    ttk.Label(root, text="Age").grid(column=0, row=0, padx=10, pady=5)
    age_entry = ttk.Entry(root)
    age_entry.grid(column=1, row=0, padx=10, pady=5)
    
    ttk.Label(root, text="Sex").grid(column=0, row=1, padx=10, pady=5)
    sex_var = tk.StringVar(value="male")
    ttk.Radiobutton(root, text='Male', variable=sex_var, value='male').grid(column=1, row=1, padx=10, pady=5)
    ttk.Radiobutton(root, text='Female', variable=sex_var, value='female').grid(column=2, row=1, padx=10, pady=5)
    
    ttk.Label(root, text="BMI").grid(column=0, row=2, padx=10, pady=5)
    bmi_entry = ttk.Entry(root)
    bmi_entry.grid(column=1, row=2, padx=10, pady=5)
    
    ttk.Label(root, text="Children").grid(column=0, row=3, padx=10, pady=5)
    children_entry = ttk.Entry(root)
    children_entry.grid(column=1, row=3, padx=10, pady=5)
    
    ttk.Label(root, text="Smoker").grid(column=0, row=4, padx=10, pady=5)
    smoker_var = tk.StringVar(value="no")
    ttk.Radiobutton(root, text='Yes', variable=smoker_var, value='yes').grid(column=1, row=4, padx=10, pady=5)
    ttk.Radiobutton(root, text='No', variable=smoker_var, value='no').grid(column=2, row=4, padx=10, pady=5)
    
    ttk.Label(root, text="Region").grid(column=0, row=5, padx=10, pady=5)
    region_var = tk.StringVar(value="northeast")
    region_combobox = ttk.Combobox(root, textvariable=region_var, values=['northeast', 'northwest', 'southeast', 'southwest'])
    region_combobox.grid(column=1, row=5, padx=10, pady=5)
    

    predict_button = ttk.Button(root, text="Predict", command=predict)
    predict_button.grid(column=0, row=6, padx=10, pady=10)
    

    result_label = ttk.Label(root, text="")
    result_label.grid(column=0, row=7, columnspan=3, padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    get_user_input()
