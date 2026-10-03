import pandas as pd
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st


@st.cache_data
def data_loading():
    df = pd.read_csv("Medicaldataset.csv")
    return df


@st.cache_resource
def train_load_and_save_model(df,model_path,selected_features):
    if os.path.exists(model_path):
         with open(model_path, "rb") as file:
              return pickle.load(file)

    X = df[selected_features]
    y = df["Result"]



    X_train, X_test , y_train, y_test = train_test_split(X, y , shuffle=True, stratify=y, train_size= 0.8)

    pipeline = Pipeline(
        [
                ("scalar", StandardScaler()),
                ("svm", SVC(C=1.0 , kernel="rbf", gamma="scale", random_state=30))
            ]
    )
    pipeline.fit(X_train,y_train)
    with open("heart.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline
     


class HeartAttack():
    def __init__(self):
        self.model_path = "heart.pkl"
        self.selected_features = ['Age', 'Gender', 'Heart rate', 
                            'Systolic blood pressure', 'Diastolic blood pressure', 
                            'Blood sugar', 'CK-MB', 'Troponin']  # ← correct
        self.df = data_loading()
        self.model = train_load_and_save_model(self.df,self.model_path,self.selected_features)


    def input(self):
        st.header("Enter the Values")
        st.write("Fill in the details to check if a patient has an heart attack or not")
        with st.form("Input Form"):
            age = st.number_input("Enter the Age", 0, 100, 0)
            gender = st.selectbox("Select Gender:", ["Male", "Female"])
            heart_rate = st.number_input("Enter the Heart Rate (bpm)", 0, 250, 72)
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", 0, 300, 120)
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 0, 200, 80)
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", 0.0, 500.0, 100.0)
            ck_mb = st.number_input("CK-MB (U/L)", 0.0, 300.0, 5.0)
            troponin = st.number_input("Troponin (ng/mL)", 0.0, 10.0, 0.01, format="%.3f")
            submitted = st.form_submit_button("Predict Heart Attack")

        if submitted:
            gender_val = 1 if gender == "Male" else 0
            input_df = pd.DataFrame([[age, gender_val, heart_rate, systolic_bp,
                                      diastolic_bp, blood_sugar, ck_mb, troponin]],
                                    columns=self.selected_features)
            prediction = self.model.predict(input_df)[0]
            st.divider()
            if prediction == 1:
                st.error("⚠️ High Risk: The model predicts a **Heart Attack**.", icon="🫀")
            else:
                st.success("✅ Low Risk: The model predicts **No Heart Attack**.", icon="❤️")
            st.subheader("Input Summary")
            st.dataframe(input_df)
        


# Wrap test input in a DataFrame with correct column names



def main():
    st.title("Heart Attack Prediction")
    st.markdown("Enter Details to pridict if you are having an heart attack or not.")

    app = HeartAttack()
    app.input()
if __name__ == "__main__":
    main()













