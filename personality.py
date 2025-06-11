import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
import streamlit as st
import seaborn as sns
import os
import matplotlib.pyplot as plt

@st.cache_data
def data_loading():

    df = pd.read_csv("personality_dataset.csv")
    df.rename(columns={
        "Time_spent_Alone": "TimeSpentAlone",
        "Stage_fear": "StageFear",
        "Social_event_attendance": "SocialEventAttendance",
        "Going_outside": "GoingOutside",
        "Drained_after_socializing": "DrainedAfterSocializing",
        "Friends_circle_size": "FriendsCircleSize",
        "Post_frequency": "PostFrequency",
        "Personality": "Personality"
    }, inplace=True)

    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_load_and_save_model(df,model_path,selected_features):

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)


    X = df[selected_features]
    y = df["Personality"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, shuffle=True, stratify=y)
    pipeline = Pipeline([
        ("scalar", StandardScaler()),
        ("svm", SVC(gamma="scale",C=1.0, kernel="rbf", random_state=42))
    ]) 
    pipeline.fit(X_train,y_train)


    stage_Fear = {label : idx for idx, label in enumerate(np.unique(df["StageFear"]))}
    df["StageFear"] = df["StageFear"].map(stage_Fear)

    DrainedAfterSocializing = {label : idx for idx, label in enumerate(np.unique(df["DrainedAfterSocializing"]))}
    df["DrainedAfterSocializing"] = df["DrainedAfterSocializing"].map(DrainedAfterSocializing)


    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    return pipeline



class Personality:
    def __init__(self):
        self.model_path = "Personality.pkl"
        
        self.selected_features = [
            "TimeSpentAlone", "StageFear", "SocialEventAttendance", 
            "GoingOutside", "DrainedAfterSocializing", 
            "FriendsCircleSize", "PostFrequency"
        ]
        self.df = data_loading()
        self.model = train_load_and_save_model(self.df,self.model_path,self.selected_features)





    def data_analysis(self):
        st.sidebar.header("Understanding the Data")
        if st.sidebar.checkbox("Show the Da ta Table"):
            st.header("Show Data Table")
            st.dataframe(self.df.head())
        
        if st.sidebar.checkbox("Show Corelation Heatmap"):
            st.subheader("Correlation Heatmap X_x")
            plt.figure(figsize=(26,22))
            sns.heatmap(self.df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt.gcf())
            plt.clf()


        if st.sidebar.checkbox("Show Pair Plot (Seletcted feature)"):
            st.header("Pair Plot")
            pair_df = self.df[["TimeSpentAlone","SocialEventAttendance","FriendsCircleSize","Personality" ]]
            sns.pairplot(pair_df,hue= "Personality")
            st.pyplot(plt.gcf())
            plt.clf()

        if st.sidebar.checkbox("Show Class Distribution"):
            st.subheader("Personality Distribution -_-")
            sns.countplot(x="Personality", data= self.df)
            st.pyplot(plt.gcf())
            plt.clf()





    def input_form(self):
        st.header("Input values:")
        st.write("Answer the questions below to find out your personality")
        with st.form("Input form"):
            time_spent_alone = st.slider("Time Spent alone (Hours/Day)", 0.0, 11.0, 0.0 , step=0.5)
            stage_fear = st.selectbox("Do you have Stage Fear?", ["Yes", "No"])
            social_event_attendance = st.slider("Social Event Attendance (Events/Month)", 0, 10, 0)
            going_outside = st.slider("Do you like to go outside? (1 = Not at all, 10 = Love it)", 1, 10, 0)
            drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
            friends_circle = st.slider("How many friends do you have?", 0, 15, 0)
            post_frequency = st.slider("Social Media Post Frequency (Posts/Month)", 0, 10, 0)

            submitted = st.form_submit_button("Predict Personality")

            if submitted:
                input_data = {
                    "TimeSpentAlone": time_spent_alone,
                    "StageFear": 1 if stage_fear == "Yes" else 0,
                    "SocialEventAttendance": social_event_attendance, 
                    "GoingOutside": going_outside,
                    "DrainedAfterSocializing": 0 if drained_after_socializing == "Yes" else 0,
                    "FriendsCircleSize": friends_circle,
                    "PostFrequency": post_frequency
                }

                input_df = pd.DataFrame([input_data])

                prediction = self.model.predict(input_df)[0]
                st.success(f"Your Personality is: {prediction}")

                if prediction == "Extrovert":
                    st.success("Go and Sit at home Nigga")
                else: 
                    st.success("Go get a Life Nigga ")



def main():

    st.title("Personality Predictor")
    st.markdown("Enter your details to check your personality that If you are an Introvert or an Extrovert")

    app = Personality()
    app.data_analysis()
    app.input_form()

if __name__ == "__main__":
    # Run the app if script is executed directly
    main()












0
