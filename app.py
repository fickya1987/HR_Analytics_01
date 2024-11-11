import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Sidebar for file upload
st.sidebar.title("HR Analytics Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

# Menu selection for Turnover Prediction or Salary Prediction
option = st.sidebar.selectbox("Choose Analysis Type", ("Turnover Prediction", "Salary Prediction"))

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load data from the uploaded file
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)

    # Convert columns to numeric if needed and remove commas
    for col in ['Age', 'Years_with_Company', 'Performance_Score', 'Job_Level', 'Salary']:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')

    # Fill any NaN values with the mean of each column
    data = data.fillna(data.mean(numeric_only=True))

    # Encode categorical columns
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Department']
    le_dict = {}

    for col in categorical_columns:
        le = LabelEncoder()
        # Fit LabelEncoder on the entire column to ensure consistency
        data[col] = le.fit_transform(data[col].astype(str))
        le_dict[col] = le  # Store the encoder for each column

    # Turnover Prediction Section
    if option == "Turnover Prediction":
        st.header("Turnover Prediction")

        # Split data into features and target
        X_turnover = data[['Age', 'Gender', 'Education_Level', 'Marital_Status', 'Years_with_Company', 
                           'Department', 'Job_Level', 'Salary']]
        y_turnover = data['Turnover']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_turnover, y_turnover, test_size=0.2, random_state=42)

        # Train a Random Forest model for turnover prediction
        model_turnover = RandomForestClassifier(n_estimators=100, random_state=42)
        model_turnover.fit(X_train, y_train)

        # Predictions
        y_pred = model_turnover.predict(X_test)

        # Accuracy and report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.write(f"Turnover Prediction Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(report)

        st.subheader("Predict Turnover for New Employee")
        input_data = {
            'Age': st.slider('Age', 20, 60, 30),
            'Gender': st.selectbox('Gender', ['Male', 'Female']),
            'Education_Level': st.selectbox('Education Level', ['High School', 'Bachelor', 'Master']),
            'Marital_Status': st.selectbox('Marital Status', ['Single', 'Married']),
            'Years_with_Company': st.slider('Years with Company', 1, 20, 5),
            'Department': st.selectbox('Department', ['IT', 'HR', 'Finance', 'Marketing']),
            'Job_Level': st.slider('Job Level', 1, 5, 3),
            'Salary': st.slider('Salary', 3000, 10000, 5000)
        }

        # Encode input data using the same encoders
        for key, value in input_data.items():
            if key in le_dict:
                input_data[key] = le_dict[key].transform([value])[0]

        turnover_prediction = model_turnover.predict(np.array(list(input_data.values())).reshape(1, -1))
        st.write("Predicted Turnover:", "Yes" if turnover_prediction[0] == 1 else "No")

    # Salary Prediction Section
    elif option == "Salary Prediction":
        st.header("Salary Prediction")

        # Split data into features and target for salary prediction
        X_salary = data[['Age', 'Gender', 'Education_Level', 'Marital_Status', 'Years_with_Company', 
                         'Department', 'Job_Level']]
        y_salary = data['Salary']

        # Train-test split
        X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

        # Train a Random Forest model for salary prediction
        model_salary = RandomForestRegressor(n_estimators=100, random_state=42)
        model_salary.fit(X_train_salary, y_train_salary)

        # Predictions
        y_pred_salary = model_salary.predict(X_test_salary)

        # Model evaluation
        mse = mean_squared_error(y_test_salary, y_pred_salary)
        st.write(f"Salary Prediction Model MSE: {mse:.2f}")

        st.subheader("Predict Salary for New Employee")
        input_data_salary = {
            'Age': st.slider('Age', 20, 60, 30),
            'Gender': st.selectbox('Gender', ['Male', 'Female']),
            'Education_Level': st.selectbox('Education Level', ['High School', 'Bachelor', 'Master']),
            'Marital_Status': st.selectbox('Marital Status', ['Single', 'Married']),
            'Years_with_Company': st.slider('Years with Company', 1, 20, 5),
            'Department': st.selectbox('Department', ['IT', 'HR', 'Finance', 'Marketing']),
            'Job_Level': st.slider('Job Level', 1, 5, 3)
        }

        # Encode input data using the same encoders
        for key, value in input_data_salary.items():
            if key in le_dict:
                input_data_salary[key] = le_dict[key].transform([value])[0]

        # Predict the salary based on input data
        salary_prediction = model_salary.predict(np.array(list(input_data_salary.values())).reshape(1, -1))
        st.write(f"Predicted Salary: ${salary_prediction[0]:,.2f}")

else:
    st.write("Please upload a dataset to continue.")
