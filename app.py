import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.datasets import load_diabetes

# Set up our webpage
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("""
This app predicts diabetes progression based on health measurements!
Adjust the sliders to match a person's health information.
""")

# Load our trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the diabetes data for showing information
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Explore Data", "Understand Features", "Predict Diabetes"])

if page == "Home":
    st.header("Welcome to the Diabetes Predictor!")
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=400", width=300)
    st.write("""
    This app uses machine learning to predict diabetes progression based on:
    - Age
    - Sex
    - Body Mass Index (BMI)
    - Blood pressure
    - And other health measurements
    
    Use the navigation on the left to explore!
    """)

elif page == "Explore Data":
    st.header("Explore the Diabetes Data")
    st.write("Here's what our data looks like:")
    st.dataframe(df.head())
    
    st.write("Basic information about our data:")
    st.write(f"Number of patients: {df.shape[0]}")
    st.write(f"Number of health measurements: {df.shape[1] - 1}")  # Subtract target column
    
    # Show a chart of diabetes progression
    st.subheader("Diabetes Progression Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['target'], bins=20, edgecolor='black')
    ax.set_xlabel('Diabetes Progression')
    ax.set_ylabel('Number of Patients')
    st.pyplot(fig)

elif page == "Understand Features":
    st.header("Understand the Health Measurements")
    
    # Explain each feature
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Gender (male/female)',
        'bmi': 'Body Mass Index (weight relative to height)',
        'bp': 'Average blood pressure',
        's1': 'Total serum cholesterol',
        's2': 'Low-density lipoproteins (LDL)',
        's3': 'High-density lipoproteins (HDL)',
        's4': 'Total cholesterol / HDL ratio',
        's5': 'Serum triglycerides level',
        's6': 'Blood sugar level'
    }
    
    selected_feature = st.selectbox("Select a health measurement to learn about", list(feature_descriptions.keys()))
    st.write(f"**{selected_feature}**: {feature_descriptions[selected_feature]}")
    
    # Show distribution of the selected feature
    fig, ax = plt.subplots()
    ax.hist(df[selected_feature], bins=20, edgecolor='black')
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Show relationship with diabetes progression
    st.subheader(f"Relationship between {selected_feature} and Diabetes Progression")
    fig, ax = plt.subplots()
    ax.scatter(df[selected_feature], df['target'])
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('Diabetes Progression')
    st.pyplot(fig)

elif page == "Predict Diabetes":
    st.header("Predict Diabetes Progression!")
    st.write("Adjust the sliders to match a person's health information:")
    
    # Create sliders for user input
    # Note: The diabetes dataset is already normalized (scaled)
    age = st.slider('Age', float(df['age'].min()), float(df['age'].max()), float(df['age'].mean()))
    sex = st.slider('Sex (0=female, 1=male)', 0.0, 1.0, 0.5)
    bmi = st.slider('BMI', float(df['bmi'].min()), float(df['bmi'].max()), float(df['bmi'].mean()))
    bp = st.slider('Blood Pressure', float(df['bp'].min()), float(df['bp'].max()), float(df['bp'].mean()))
    s1 = st.slider('Total Cholesterol', float(df['s1'].min()), float(df['s1'].max()), float(df['s1'].mean()))
    s2 = st.slider('LDL', float(df['s2'].min()), float(df['s2'].max()), float(df['s2'].mean()))
    s3 = st.slider('HDL', float(df['s3'].min()), float(df['s3'].max()), float(df['s3'].mean()))
    s4 = st.slider('Total/HDL Cholesterol', float(df['s4'].min()), float(df['s4'].max()), float(df['s4'].mean()))
    s5 = st.slider('Triglycerides', float(df['s5'].min()), float(df['s5'].max()), float(df['s5'].mean()))
    s6 = st.slider('Blood Sugar', float(df['s6'].min()), float(df['s6'].max()), float(df['s6'].mean()))
    
    # When button is clicked, make prediction
    if st.button('Predict Diabetes Progression!'):
        # Create input array for model
        input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Show results
        st.success(f"Predicted diabetes progression: **{prediction:.2f}**")
        
        # Explain what this means
        st.subheader("What does this mean?")
        st.write("""
        The diabetes progression score is a measure of disease progression one year after baseline.
        Higher values indicate more severe progression of diabetes.
        
        **Note**: This is a prediction tool for educational purposes only.
        Always consult a real doctor for medical advice!
        """)
        
        # Show where this prediction falls in the distribution
        st.subheader("How this compares to other patients")
        fig, ax = plt.subplots()
        ax.hist(df['target'], bins=20, edgecolor='black', alpha=0.7, label='All Patients')
        ax.axvline(prediction, color='red', linestyle='--', linewidth=2, label='Prediction')
        ax.set_xlabel('Diabetes Progression')
        ax.set_ylabel('Number of Patients')
        ax.legend()
        st.pyplot(fig)

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("Made by vishwa")
st.sidebar.warning("This is for educational purposes only. Always consult a doctor for medical advice!")