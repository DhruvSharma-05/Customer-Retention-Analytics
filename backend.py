from flask import Flask, jsonify, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the churn prediction model
model = joblib.load(r'data\churn_prediction_model.pkl')

# Load dataset (just for checking column names)
DATA_PATH = r"data\churn_prediction_model.pkl"  # This path should point to your dataset, not model

# Create a LabelEncoder for encoding categorical variables
label_encoders = {}

@app.route("/get_data", methods=["GET"])
def get_data():
    """
    API endpoint to retrieve customer data.
    """
    try:
        # Load the dataset
        data = pd.read_csv(DATA_PATH)
        
        # Convert DataFrame to JSON
        return jsonify(data.to_dict(orient="records"))
    except Exception as e:
        app.logger.error(f"Error in get_data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_churn", methods=["POST"])
def predict_churn():
    """
    API endpoint to predict churn for a given customer data.
    """
    try:
        # Get JSON input (new customer data)
        input_data = request.get_json()

        # Prepare data for prediction
        input_df = pd.DataFrame(input_data)

        # Remove 'Customer_ID' column if it exists
        if 'Customer_ID' in input_df.columns:
            input_df = input_df.drop(columns=['Customer_ID'])

        # Encode categorical columns using LabelEncoder (if they exist)
        for column in input_df.select_dtypes(include=["object"]).columns:
            if column not in label_encoders:
                le = LabelEncoder()
                le.fit(input_df[column])
                label_encoders[column] = le
            input_df[column] = label_encoders[column].transform(input_df[column])

        # Predict churn using the model
        predictions = model.predict(input_df)
        
        # Add predictions to the data
        input_df["Churn_Prediction"] = predictions.tolist()
        
        # Return the prediction results
        return jsonify(input_df[["Churn_Prediction"]].to_dict(orient="records"))
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
