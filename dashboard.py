import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

# Set page configuration with custom theme
st.set_page_config(
    page_title="Customer Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .stSelectbox label, .stSlider label {
        color: #fff !important;
        font-size: 1.1em !important;
        font-weight: 500 !important;
    }
    .plot-container {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #fff !important;
    }
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .prediction-table {
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:5000"

# Function to prepare data for prediction
def prepare_prediction_data(data):
    # Create a copy of the data
    pred_data = data.copy()
    
    # Remove columns not used in training
    columns_to_drop = ['Customer_ID', 'Churn_Flag', 'Last_Active_Date', 
                      'Purchase_History', 'Signup_Date']
    pred_data = pred_data.drop(columns=columns_to_drop, errors='ignore')
    
    # Ensure only necessary columns are included
    required_columns = ['Segment', 'Retention_Rate', 'Engagement_Score', 'Region']
    pred_data = pred_data[required_columns]
    
    return pred_data

# Function to display metric cards
def display_metric_cards(data):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3 style='text-align: center; color: #4CAF50;'>Total Customers</h3>
                <h2 style='text-align: center; color: white;'>{}</h2>
            </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        avg_retention = data['Retention_Rate'].mean()
        st.markdown("""
            <div class="metric-card">
                <h3 style='text-align: center; color: #2196F3;'>Average Retention</h3>
                <h2 style='text-align: center; color: white;'>{:.1f}%</h2>
            </div>
        """.format(avg_retention * 100), unsafe_allow_html=True)
    
    with col3:
        avg_engagement = data['Engagement_Score'].mean()
        st.markdown("""
            <div class="metric-card">
                <h3 style='text-align: center; color: #FF9800;'>Avg Engagement</h3>
                <h2 style='text-align: center; color: white;'>{:.1f}</h2>
            </div>
        """.format(avg_engagement), unsafe_allow_html=True)

# Main title
st.title("📊 Customer Retention Analytics")
st.markdown("---")

# Sidebar upload section
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload a CSV file with customer data"
    )

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Display metric cards
    display_metric_cards(data)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Analytics", "🔍 Data Preview"])
    
    with tab1:
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            segment_options = ['All'] + list(data['Segment'].unique())
            segment_filter = st.selectbox("Select Segment", options=segment_options)
        
        with col2:
            retention_min, retention_max = float(data['Retention_Rate'].min()), float(data['Retention_Rate'].max())
            retention_range = st.slider(
                "Retention Rate Range",
                min_value=retention_min,
                max_value=retention_max,
                value=(retention_min, retention_max),
                format="%.2f"
            )
        
        # Filter data
        filtered_data = data.copy()
        if segment_filter != 'All':
            filtered_data = filtered_data[filtered_data['Segment'] == segment_filter]
        filtered_data = filtered_data[
            (filtered_data['Retention_Rate'] >= retention_range[0]) &
            (filtered_data['Retention_Rate'] <= retention_range[1])
        ]
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment Distribution
            fig_segment = px.pie(
                filtered_data,
                names='Segment',
                title='Segment Distribution',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_segment.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        
        with col2:
            # Region Distribution
            fig_region = px.bar(
                filtered_data,
                x='Region',
                title='Region Distribution',
                color='Region',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_region.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_region, use_container_width=True)
        
        # Retention Rate by Segment
        fig_retention = px.line(
            data.groupby('Segment')['Retention_Rate'].mean().reset_index(),
            x='Segment',
            y='Retention_Rate',
            title='Average Retention Rate by Segment',
            markers=True,
            color_discrete_sequence=['#00ff00']
        )
        fig_retention.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis_title='Average Retention Rate'
        )
        st.plotly_chart(fig_retention, use_container_width=True)
        
        # Churn Predictions
        st.subheader("🔮 Churn Predictions")
        try:
            # Prepare data for prediction
            prediction_data = prepare_prediction_data(filtered_data)
            
            # Make prediction request
            response = requests.post(f"{API_URL}/predict_churn", json=prediction_data.to_dict(orient='records'))
            
            if response.status_code == 200:
                predictions = response.json()
                pred_df = pd.DataFrame(predictions)
                
                # Add original Customer_ID
                pred_df['Customer_ID'] = filtered_data['Customer_ID'].values
                
                # Add prediction labels
                pred_df['Churn_Risk'] = pred_df['Churn_Prediction'].map({
                    0: '✅ Low Risk',
                    1: '⚠️ High Risk'
                })
                
                # Display predictions
                st.dataframe(
                    pred_df[['Customer_ID', 'Churn_Risk']],
                    use_container_width=True,
                    column_config={
                        "Customer_ID": "Customer ID",
                        "Churn_Risk": "Churn Risk Status"
                    }
                )
                
                # Prediction summary
                col1, col2 = st.columns(2)
                with col1:
                    num_high_risk = (pred_df['Churn_Prediction'] == 1).sum()
                    st.metric("High Risk Customers", num_high_risk)
                with col2:
                    risk_percentage = (num_high_risk / len(pred_df) * 100)
                    st.metric("Risk Percentage", f"{risk_percentage:.1f}%")
                
            else:
                st.error(f"Error fetching predictions: {response.text}")
        except Exception as e:
            st.error(f"Error during API request: {str(e)}")
            st.error("Please check if the API server is running at " + API_URL)
    
    with tab2:
        st.subheader("📋 Data Preview")
        st.dataframe(filtered_data, use_container_width=True)
        
        st.subheader("📊 Basic Statistics")
        st.dataframe(filtered_data.describe(), use_container_width=True)
else:
    st.info("👆 Please upload a CSV file to get started")