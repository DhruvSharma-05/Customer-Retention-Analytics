import pandas as pd

def analyze_uploaded_data(data):
    try:
        summary = {
            "Column Names": list(data.columns),
            "Missing Values": data.isnull().sum().to_dict(),
            "Data Types": data.dtypes.to_dict(),
            "Basic Stats": data.describe(include='all').to_dict()
        }
        return summary
    except Exception as e:
        return f"Error analyzing data: {str(e)}"
