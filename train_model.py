import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the customer data
data = pd.read_csv("data\customer_data.csv")

# Convert categorical columns into numeric (Label Encoding for simplicity)
label_encoder = LabelEncoder()
data['Segment'] = label_encoder.fit_transform(data['Segment'])
data['Region'] = label_encoder.fit_transform(data['Region'])

# Convert the 'Churn_Flag' column into a binary target variable
X = data.drop(['Customer_ID', 'Churn_Flag', 'Signup_Date', 'Last_Active_Date', 'Purchase_History'], axis=1)
y = data['Churn_Flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model as a .pkl file
joblib.dump(model, "data/churn_prediction_model.pkl")

print("Model trained and saved successfully!")
