import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report

# Main heading
st.title("Find Out Your Personality")
data_file="dataSet\personality_synthetic_dataset.csv"
# Load data
df = pd.read_csv(data_file)

# Side columns for input
st.sidebar.header("Enter Your Details (1 to 10)")
feature_names = df.columns[1:]

user_data = []
for name in feature_names:
    val = st.sidebar.slider(
        label=f"{name}",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )
    user_data.append(val)

user_data_np = np.array(user_data).reshape(1, -1)

# Display the head of dataset
st.subheader("Sample Data")
st.dataframe(df.head())

# Prepare data for classifier
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

# Accuracy
n_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, n_pred) * 100
st.subheader("Model Accuracy")
st.write(f"{accuracy:.2f}%")

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, n_pred, output_dict=False)
st.text(report)

# Prediction button logic
if st.sidebar.button("Predict Personality"):
    # Transform the input data
    user_data_scaled = scaler.transform(user_data_np)
    # Predict
    user_pred = model.predict(user_data_scaled)[0]
    # Display the personality type in green
    st.subheader("Predicted Personality Type")
    st.markdown(f"<span style='color:green; font-size:24px;'>**{user_pred}**</span>", unsafe_allow_html=True)
else:
    st.subheader("Predicted Personality Type")
    st.write("Please enter your details and click 'Predict Personality' button.")

# Dataset download
csv = df.to_csv(index=False).encode('utf-8')
st.subheader("Download Dataset")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='personality_synthetic_dataset.csv',
    mime='text/csv'
)