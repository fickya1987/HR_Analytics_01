import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Sidebar untuk upload file
st.sidebar.title("HR Analytics Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

# Load data
if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
else:
    st.write("Please upload a dataset to continue.")
    st.stop()

# Encode categorical columns if needed
le = LabelEncoder()
for col in ['Gender', 'Education_Level', 'Marital_Status', 'Department']:
    if col in data.columns:
        data[col] = le.fit_transform(data[col])

# Sidebar selection for analysis type
option = st.sidebar.selectbox("Choose Analysis Type", 
                              ("Descriptive Analysis", "Predictive Analysis", "Prescriptive Analysis"))

st.title("HR Analytics Dashboard")

# Descriptive Analysis
if option == "Descriptive Analysis":
    st.header("Descriptive Analysis")
    st.write("Basic Statistics of HR Data")
    st.write(data.describe())
    
    st.subheader("Employee Distribution by Department")
    st.bar_chart(data['Department'].value_counts())
    
    st.subheader("Average Salary by Job Level")
    avg_salary = data.groupby('Job_Level')['Salary'].mean()
    st.bar_chart(avg_salary)

    st.subheader("Average Performance by Years with Company")
    avg_performance = data.groupby('Years_with_Company')['Performance_Score'].mean()
    st.line_chart(avg_performance)

# Predictive Analysis
elif option == "Predictive Analysis":
    st.header("Predictive Analysis")
    
    # Select features and target variable
    X = data[['Age', 'Gender', 'Education_Level', 'Marital_Status', 'Years_with_Company', 
              'Department', 'Job_Level', 'Salary']]
    y = data['Turnover']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy and report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    st.write(f"Prediction Accuracy: {accuracy:.2f}")
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
    
    # Encode input data
    for key, value in input_data.items():
        if key in ['Gender', 'Education_Level', 'Marital_Status', 'Department']:
            input_data[key] = le.transform([value])[0]
    
    prediction = model.predict(np.array(list(input_data.values())).reshape(1, -1))
    st.write("Predicted Turnover:" , "Yes" if prediction[0] == 1 else "No")

# Prescriptive Analysis
elif option == "Prescriptive Analysis":
    st.header("Prescriptive Analysis")
    
    st.write("Based on Predictive Analysis, here are some recommendations:")
    
    st.subheader("Retention Strategies")
    st.write("- For employees likely to leave, consider offering additional training, career growth opportunities, and bonuses.")
    st.write("- Improve work-life balance initiatives for high-risk employees.")
    
    st.subheader("Training & Development")
    st.write("- For employees with low performance scores, offer targeted skill development programs.")
    st.write("- Encourage career growth through mentorship programs for junior staff.")
    
    st.subheader("Salary Recommendations")
    st.write("- Offer competitive salary packages for high-performing employees to increase retention.")
    st.write("- Consider adjusting salaries based on job level and department performance trends.")
