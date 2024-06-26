import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os

# Cache the loading of the trained pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('FS-EVR-RFE_pipeline.pkl')

# Load the trained pipeline
pipeline = load_pipeline()

# Get the RFE step from the pipeline
rfe = pipeline.named_steps['feature_selection']

# Define all possible feature names
all_feature_names = ['Cement', 'Blast-furnace Slag', 'Fly Ash', 'Water', 'Super-plasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age of testing']

# Get the indices of the selected features
selected_feature_indices = rfe.get_support(indices=True)

# Get the names of the selected features based on the indices
selected_feature_names = [all_feature_names[i] for i in selected_feature_indices]

# Default values for the inputs
default_values = {
    'Cement': 276.5,
    'Blast-furnace Slag': 74.26,
    'Fly Ash': 62.81,
    'Water': 182.98,
    'Super-plasticizer': 6.41,
    'Coarse Aggregate': 0.0,  # Not selected, default to 0
    'Fine Aggregate': 770.49,
    'Age of testing': 44.05
}

# Define the pages
def prediction_page():
    st.title("Compressive Strength of High Performance Concrete Prediction App")
    st.write('Machine learning-powered predictor for compressive strength of HPC')

    # Display the table with input variable details
    st.write("### Descriptive Statistics of the Training Data")
    st.markdown("""
    The training data originally consist of 8 input variables. However, Recursive Feature Elimination removes one input variable during the feature selection process to get an optimal model. The details of the selected features and the output variable are shown in the table below:
    """)

    table_data = {
        "Input Variables": ["Cement kg/m3", "Blast-furnace Slag kg/m3", "Fly Ash kg/m3", "Water kg/m3", "Super-plasticizer kg/m3", "Fine aggregate kg/m3", "Age of testing"],
        "Unit": ["kg/m3", "kg/m3", "kg/m3", "kg/m3", "kg/m3", "kg/m3", "Day"],
        "Minimum": [102.00, 0.00, 0.00, 121.80, 0.00, 594.00, 1.00],
        "Maximum": [540.00, 359.40, 260.00, 247.00, 32.20, 992.60, 365.00]
    }
    output_data = {
        "Output variable": ["Concrete compressive strength"],
        "Unit": ["MPA"],
        "Minimum": [2.30],
        "Maximum": [82.6]
    }

    table_df = pd.DataFrame(table_data)
    output_df = pd.DataFrame(output_data)

    # Display both tables
    st.table(table_df)
    st.table(output_df)

    st.image('HPC App.png', caption='Predicting Compressive Strength of HPC', use_column_width=True)

    # Sidebar for input variables
    st.sidebar.header("Select the Input Features")

    user_input = {}
    for feature in selected_feature_names:
        user_input[feature] = st.sidebar.number_input(f"**{feature}**", value=default_values[feature])

    if st.sidebar.button("Predict Compressive Strength of HPC"):
        # Create a full input dictionary with default values (e.g., 0)
        user_input_all = {feature: 0.0 for feature in all_feature_names}

        # Update the dictionary with the selected feature inputs
        user_input_all.update(user_input)

        # Convert the user input to a DataFrame
        input_df_all = pd.DataFrame([user_input_all])

        # Ensure that the DataFrame columns are in the correct order
        input_df_all = input_df_all[all_feature_names]

        # Make a prediction using the pipeline
        prediction = pipeline.predict(input_df_all)

        # Display the prediction
        st.write(f"### Predicted Compressive Strength of HPC: **{prediction[0]:.2f} MPa**")

def data_exploration_page():
    st.title("Data Exploration")
    st.write("Explore the distribution of the variables and their correlations.")

    st.image('Cement_distribution.png', caption='Cement Distribution', use_column_width=True)
    st.image('Blast-furnace Slag_distribution.png', caption='Blast-furnace Slag Distribution', use_column_width=True)
    st.image('Fly Ash_distribution.png', caption='Fly Ash Distribution', use_column_width=True)
    st.image('Water_distribution.png', caption='Water Distribution', use_column_width=True)
    st.image('Super-plasticizer_distribution.png', caption='Super-plasticizer Distribution', use_column_width=True)
    st.image('Fine Aggregate_distribution.png', caption='Fine Aggregate Distribution', use_column_width=True)
    st.image('Concrete compressive strength_distribution.png', caption='Concrete compressive strength Distribution', use_column_width=True)
    
    # Correlation map
    st.write("### Correlation Matrix")
    st.write("This map shows the correlation between the variables in the data.")
    st.image('correlation_matrix.png', caption='Correlation Matrix', use_column_width=True)
    
def interpretability_page():
    st.title("Model Interpretability")
    st.write("""
    Understanding how a machine learning model makes its predictions is crucial for building trust and ensuring transparency. Model interpretability helps in identifying which features are most influential in determining the outcomes, and provides insights into the model's decision-making process.
    """)

    st.image('Aggregate_FI.png', caption='Inherent Feature Importance of the Model', use_column_width=True)
    st.write("""
    This plot shows the aggregate feature importance of the ensemble model. The model comprises CatBoost and XGBoost combined via stacking. The output highlights that the age of testing, cement, and water are the most important features.
    """)

    st.image('SHAP_Bar.png', caption='SHAP Feature Importance', use_column_width=True)
    st.write("""
    This plot shows the feature importance based on the SHAP (SHapley Additive exPlanations) framework. Again, the top predictors are cement, age of testing, and water.
    """)

    st.image('SHAP_Violin.png', caption='Directional Impact of the Features on the Model', use_column_width=True)
    st.write("""
    This plot shows the impact of each feature on the model. For instance, the plot indicates that higher values of cement content contribute positively to the model, while lower values of water contribute positively to the model.
    """)

# Function to download feedback as CSV
@st.cache_data
def download_feedback():
    if os.path.exists('feedback.csv'):
        feedback_df = pd.read_csv('feedback.csv')
        feedback_csv = feedback_df.to_csv(index=False)
        return feedback_csv
    else:
        return ""

# Function to save feedback to a CSV file
def save_feedback(feedback_text, email):
    feedback_exists = os.path.exists('feedback.csv')
    with open('feedback.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not feedback_exists:
            # Write header if file does not exist
            writer.writerow(["feedback", "email"])
        writer.writerow([feedback_text, email])

# Feedback page
def feedback_page():
    st.title("User Feedback")
    st.write("We value your feedback. Please let us know how we can improve this app.")

    feedback_text = st.text_area("Enter your feedback here")
    email = st.text_input("Your Email (optional)")

    if st.button("Submit Feedback"):
        if feedback_text:
            save_feedback(feedback_text, email)
            st.success("Thank you for your feedback, we appreciate your input!")
        else:
            st.error("Feedback cannot be empty.")

def admin_page():
    st.title("Admin Page")
    st.write("Download feedback data. This user feedback data can only be downloaded by the admin or their delegates.")

    password = st.text_input("Enter password", type="password")
    submit_button = st.button("Submit")

    if submit_button:
        if password == "Abcd":  # Replace with your secure password
            feedback_csv = download_feedback()
            if feedback_csv:
                st.download_button(
                    label="Download Feedback as CSV",
                    data=feedback_csv,
                    file_name="feedback.csv",
                    mime="text/csv"
                )
            else:
                st.error("No feedback data available.")
        else:
            st.error("Incorrect password, contact the admin")

# Main app
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Prediction", "Data Exploration", "Model Interpretability", "User Feedback", "Admin Page"])

# Route to the selected page
if page == "Prediction":
    prediction_page()
elif page == "Data Exploration":
    data_exploration_page()
elif page == "Model Interpretability":
    interpretability_page()
elif page == "User Feedback":
    feedback_page()
elif page == "Admin Page":
    admin_page()
