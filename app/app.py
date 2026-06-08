import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Configuration
API_URL = "http://localhost:8080/predict" # Adjust to Cloud Run Endpoint when deployed

st.set_page_config(page_title="NLP Review Intelligence Dashboard", layout="wide")

st.title("📊 NLP Review Intelligence System")
st.markdown("Production-Ready Transformer-based sentiment & topic modeling ecosystem.")

# Sidebar Controls
st.sidebar.header("Data Actions")
uploaded_file = st.sidebar.file_uploader("Upload Product Reviews (CSV)", type="csv")

# Tab Layout
tab1, tab2 = st.tabs(["🔴 Live Inference & API Testing", "📈 Executive Business Intelligence"])

with tab1:
    st.subheader("Real-time Transformer Inference")
    user_input = st.text_area("Enter review text to evaluate against fine-tuned BERT model:", 
                              "The customer support team resolved my issue immediately. Highly recommend!")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            try:
                res = requests.post(API_URL, json={"text": user_input})
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"**Predicted Sentiment:** {data['sentiment']}")
                    st.metric(label="Model Confidence Score", value=f"{data['confidence'] * 100:.2f}%")
                else:
                    st.error("API error. Did you spin up your FastAPI docker container locally?")
            except Exception as e:
                st.warning("Running in local mock mode: Live API connection unreachable.")
                st.info(f"Mock Output Prediction: **Positive** (Confidence: 94.8%)")

with tab2:
    st.subheader("Data Warehouse Aggregations (Mocked Data Lakehouse Output)")
    
    # Mocking Data coming out of dbt data marts
    chart_data = pd.DataFrame({
        'Topic': ['Battery Life', 'Customer Support', 'UI/UX Design', 'Pricing/Value', 'Shipping Speed'],
        'Volume': [142, 98, 76, 50, 32],
        'Average Sentiment Score': [0.85, -0.42, 0.61, 0.20, 0.91]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(chart_data, x='Topic', y='Volume', title='Auto-Discovered Topics Volume (BERTopic)')
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.scatter(chart_data, x='Topic', y='Average Sentiment Score', size='Volume', color='Average Sentiment Score',
                         color_continuous_scale=px.colors.sequential.RdBu, title='Topic Sentiment Matrix')
        st.plotly_chart(fig2, use_container_width=True)