# Load data
if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
else:
    st.write("Please upload a dataset to continue.")
    st.stop()

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

# Split data into features and target
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

# Encode input data using the same encoders
for key, value in input_data.items():
    if key in le_dict:
        input_data[key] = le_dict[key].transform([value])[0]

prediction = model.predict(np.array(list(input_data.values())).reshape(1, -1))
st.write("Predicted Turnover:", "Yes" if prediction[0] == 1 else "No")
