import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained pipeline
pipeline = joblib.load('FS-EVR-RFE_pipeline.pkl')

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
    st.title("High Performance Concrete Prediction App")
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

    # Load the data
    data = pd.read_excel('data.xlsx')
    
    

    # Distribution of the variables
    st.write("### Distribution of Variables")
    fig, axes = plt.subplots(len(data.columns), 1, figsize=(10, len(data.columns) * 5))
    fig.tight_layout(pad=10.0)  # Increased padding between plots

    for i, col in enumerate(data.columns):
        color = 'blue' if col != 'Concrete compressive strength' else 'red'
        sns.histplot(data[col], kde=True, ax=axes[i], color=color)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    st.pyplot(fig)

    # Correlation map
    st.write("### Correlation Map")
    corr_matrix = data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix of Variables')
    st.pyplot(fig)


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

def feedback_page():
    st.title("User Feedback")
    st.write("We value your feedback! Please provide your thoughts and suggestions below:")

    feedback = st.text_area("Your feedback")
    email = st.text_input("Your email (optional)")

    if st.button("Submit Feedback"):
        st.write("Thank you for your feedback!")
        # Here you could add code to save the feedback, send an email, etc.

# Main app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Data Exploration", "Model Interpretability", "User Feedback"])

if page == "Prediction":
    prediction_page()
elif page == "Data Exploration":
    data_exploration_page()
elif page == "Model Interpretability":
    interpretability_page()
else:
    feedback_page()