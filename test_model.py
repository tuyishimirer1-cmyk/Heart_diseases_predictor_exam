import os
import joblib
import pandas as pd
import numpy as np


base_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(base_dir, "artifacts")

model = joblib.load(os.path.join(artifacts_dir, "best_model.pkl"))

with open(os.path.join(artifacts_dir, "feature_columns.txt")) as f:
    feature_columns = [line.strip() for line in f]

with open(os.path.join(artifacts_dir, "class_names.txt")) as f:
    class_names = [line.strip() for line in f]



test_patients_human = [
    {
        "age": 65,
        "sex": "Male",
        "cp": "Asymptomatic",
        "trestbps": 160,
        "chol": 300,
        "fbs": True,
        "restecg": "Left Ventricular Hypertrophy",
        "thalach": 110,
        "exang": "Yes",
        "oldpeak": 3.5,
        "slope": "Flat",
        "ca": 3,
        "thal": "Reversible Defect"
    },
    {
        "age": 45,
        "sex": "Female",
        "cp": "Atypical Angina",
        "trestbps": 120,
        "chol": 220,
        "fbs": False,
        "restecg": "Normal",
        "thalach": 150,
        "exang": "No",
        "oldpeak": 0.0,
        "slope": "Upsloping",
        "ca": 0,
        "thal": "Normal"
    },
    {
        "age": 55,
        "sex": "Male",
        "cp": "Non-anginal Pain",
        "trestbps": 130,
        "chol": 250,
        "fbs": False,
        "restecg": "ST-T Wave Abnormality",
        "thalach": 140,
        "exang": "No",
        "oldpeak": 1.5,
        "slope": "Flat",
        "ca": 1,
        "thal": "Fixed Defect"
    }
]



sex_map = {"Female": 0, "Male": 1}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {False: 0, True: 1}
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"No": 0, "Yes": 1}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}


test_patients_numeric = []

for p in test_patients_human:
    numeric_patient = {
        "age": p["age"],
        "sex": sex_map[p["sex"]],
        "cp": cp_map[p["cp"]],
        "trestbps": p["trestbps"],
        "chol": p["chol"],
        "fbs": fbs_map[p["fbs"]],
        "restecg": restecg_map[p["restecg"]],
        "thalach": p["thalach"],
        "exang": exang_map[p["exang"]],
        "oldpeak": p["oldpeak"],
        "slope": slope_map[p["slope"]],
        "ca": p["ca"],
        "thal": thal_map[p["thal"]]
    }
    test_patients_numeric.append(numeric_patient)



df_test = pd.DataFrame(test_patients_numeric)[feature_columns]

pred_probs = model.predict_proba(df_test)
pred_idx = np.argmax(pred_probs, axis=1)



results_df = df_test.copy()
results_df["Diagnosis"] = [class_names[i] for i in pred_idx]

for i, cls in enumerate(class_names):
    results_df[f"{cls} %"] = np.round(pred_probs[:, i] * 100, 2)



print("\nCLINICAL TEST RESULTS\n")

for i in range(len(test_patients_human)):

    print(f"PATIENT {i+1}\n")
    print("INPUT (Human Readable):")
    for k, v in test_patients_human[i].items():
        print(f"{k:<12}: {v}")

    print("\nOUTPUT:")
    print("DIAGNOSIS:", results_df.loc[i, "Diagnosis"])
    for cls in class_names:
        print(f"{cls.upper():<18}: {results_df.loc[i, f'{cls} %']}%")

    print("\n")
