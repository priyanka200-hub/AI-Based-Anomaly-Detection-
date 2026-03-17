import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

st.set_page_config(page_title="AI-Based Anomaly Detection Platform", page_icon="🔍", layout="wide")

# Initialize session state
if 'logs' not in st.session_state:
    st.session_state.logs = [
        {
            "Timestamp": "2026-03-17 10:00:00",
            "File": "sample_data1.csv",
            "Method": "Isolation Forest",
            "Data Points": 1000,
            "Anomalies Detected": 50,
            "Anomaly %": "5.00%"
        },
        {
            "Timestamp": "2026-03-17 11:15:00",
            "File": "sample_data2.csv",
            "Method": "One-Class SVM",
            "Data Points": 800,
            "Anomalies Detected": 32,
            "Anomaly %": "4.00%"
        },
        {
            "Timestamp": "2026-03-17 12:30:00",
            "File": "sample_data3.csv",
            "Method": "Isolation Forest",
            "Data Points": 1200,
            "Anomalies Detected": 72,
            "Anomaly %": "6.00%"
        },
        {
            "Timestamp": "2026-03-17 13:45:00",
            "File": "sample_data4.csv",
            "Method": "One-Class SVM",
            "Data Points": 950,
            "Anomalies Detected": 38,
            "Anomaly %": "4.00%"
        },
        {
            "Timestamp": "2026-03-17 14:20:00",
            "File": "sample_data5.csv",
            "Method": "Isolation Forest",
            "Data Points": 1100,
            "Anomalies Detected": 66,
            "Anomaly %": "6.00%"
        },
        {
            "Timestamp": "2026-03-17 15:10:00",
            "File": "sample_data6.csv",
            "Method": "One-Class SVM",
            "Data Points": 750,
            "Anomalies Detected": 30,
            "Anomaly %": "4.00%"
        },
        {
            "Timestamp": "2026-03-17 16:05:00",
            "File": "sample_data7.csv",
            "Method": "Isolation Forest",
            "Data Points": 1300,
            "Anomalies Detected": 78,
            "Anomaly %": "6.00%"
        },
        {
            "Timestamp": "2026-03-17 17:30:00",
            "File": "sample_data8.csv",
            "Method": "One-Class SVM",
            "Data Points": 900,
            "Anomalies Detected": 45,
            "Anomaly %": "5.00%"
        },
        {
            "Timestamp": "2026-03-17 18:15:00",
            "File": "sample_data9.csv",
            "Method": "Isolation Forest",
            "Data Points": 1050,
            "Anomalies Detected": 63,
            "Anomaly %": "6.00%"
        },
        {
            "Timestamp": "2026-03-17 19:00:00",
            "File": "sample_data10.csv",
            "Method": "One-Class SVM",
            "Data Points": 850,
            "Anomalies Detected": 34,
            "Anomaly %": "4.00%"
        }
    ]
if 'total_uploads' not in st.session_state:
    st.session_state.total_uploads = 10
if 'total_runs' not in st.session_state:
    st.session_state.total_runs = 10
if 'total_anomalies' not in st.session_state:
    st.session_state.total_anomalies = 508

st.title("🔍 AI-Based Anomaly Detection Platform")
st.markdown("""
Welcome to our advanced anomaly detection platform powered by machine learning.
Upload your dataset, select a detection method, and identify anomalies in your data effortlessly.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Upload & Analyze", "Prediction Logs"])

with tab1:
    st.header("Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Datasets Uploaded", st.session_state.total_uploads)
    with col2:
        st.metric("Total Prediction Runs", st.session_state.total_runs)
    with col3:
        st.metric("Total Anomalies Detected", st.session_state.total_anomalies)
    
    st.subheader("Recent Activity")
    if st.session_state.logs:
        recent_logs = pd.DataFrame(st.session_state.logs[-5:])  # Last 5
        st.dataframe(recent_logs)
    else:
        st.info("No predictions yet.")
    
    st.subheader("What is Anomaly Detection?")
    st.markdown("""
    Anomaly detection is a technique used to identify data points, events, or observations that deviate significantly from the normal behavior of a dataset. It is commonly used in various fields such as fraud detection, network security, industrial damage detection, medical diagnosis, and predictive maintenance.
    
    Our AI-powered platform uses advanced machine learning algorithms like Isolation Forest and One-Class SVM to automatically detect anomalies in your uploaded datasets. Simply upload your CSV data, select the detection method and parameters, and get instant results with visualizations and downloadable reports.
    """)

with tab2:
    st.header("Upload & Analyze Dataset")
    
    # Upload file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Save file to data folder
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.total_uploads += 1
        st.success(f"File {uploaded_file.name} uploaded and saved.")
        
        df = pd.read_csv(file_path)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())
        
        # Select columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 2:
            st.error("Please upload a dataset with at least 2 numeric columns.")
        else:
            x_col = st.selectbox("Select X-axis column", numeric_columns)
            y_col = st.selectbox("Select Y-axis column", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
            
            # Method selection
            method = st.selectbox("Anomaly Detection Method", ["Isolation Forest", "One-Class SVM"])
            
            # Parameters
            contamination = st.slider("Contamination (expected % of anomalies)", 0.01, 0.5, 0.1)
            
            if st.button("Detect Anomalies"):
                # Prepare data
                data = df[[x_col, y_col]].dropna()
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                
                # Detect anomalies
                if method == "Isolation Forest":
                    model = IsolationForest(contamination=contamination, random_state=42)
                else:
                    model = OneClassSVM(nu=contamination, kernel="rbf", gamma=0.1)
                
                predictions = model.fit_predict(scaled_data)
                anomalies = predictions == -1
                num_anomalies = anomalies.sum()
                
                # Update session state
                st.session_state.total_runs += 1
                st.session_state.total_anomalies += num_anomalies
                
                # Add to logs
                log_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "File": uploaded_file.name,
                    "Method": method,
                    "Data Points": len(data),
                    "Anomalies Detected": num_anomalies,
                    "Anomaly %": f"{num_anomalies / len(data) * 100:.2f}%"
                }
                st.session_state.logs.append(log_entry)
                
                # Results
                st.subheader("Anomaly Detection Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Data Points", len(data))
                    st.metric("Detected Anomalies", num_anomalies)
                    st.metric("Anomaly Percentage", f"{num_anomalies / len(data) * 100:.2f}%")
                
                with col2:
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data[x_col], y=data[y_col], mode='markers',
                                             marker=dict(color=anomalies.map({True: 'red', False: 'blue'}),
                                                         size=8),
                                             name='Data Points'))
                    fig.update_layout(title=f"Anomaly Detection: {method}",
                                      xaxis_title=x_col,
                                      yaxis_title=y_col,
                                      showlegend=False)
                    st.plotly_chart(fig)
                
                # Anomalous data
                st.subheader("Detected Anomalies")
                anomalous_data = data[anomalies]
                st.dataframe(anomalous_data)
                
                # Download button
                csv = anomalous_data.to_csv(index=False)
                st.download_button("Download Anomalies CSV", csv, "anomalies.csv", "text/csv")
    
    else:
        st.info("Please upload a CSV file to get started.")

with tab3:
    st.header("Prediction Logs")
    if st.session_state.logs:
        logs_df = pd.DataFrame(st.session_state.logs)
        st.dataframe(logs_df)
        
        # Visualizations
        st.subheader("Anomaly Detection Visualizations")
        
        # Convert timestamp to datetime
        logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Line chart of anomalies over time
            fig = px.line(logs_df, x='Timestamp', y='Anomalies Detected', 
                         title='Anomalies Detected Over Time', markers=True)
            st.plotly_chart(fig)
        
        with col2:
            # Bar chart of methods used
            method_counts = logs_df['Method'].value_counts().reset_index()
            method_counts.columns = ['Method', 'Count']
            fig2 = px.bar(method_counts, x='Method', y='Count', 
                         title='Anomaly Detection Methods Used')
            st.plotly_chart(fig2)
        
        # Additional chart: Anomaly percentage distribution
        st.subheader("Anomaly Percentage Distribution")
        logs_df['Anomaly % Value'] = logs_df['Anomaly %'].str.rstrip('%').astype(float)
        fig3 = px.histogram(logs_df, x='Anomaly % Value', nbins=10, 
                           title='Distribution of Anomaly Percentages')
        st.plotly_chart(fig3)
        
        # Download logs
        logs_csv = logs_df.to_csv(index=False)
        st.download_button("Download Logs CSV", logs_csv, "prediction_logs.csv", "text/csv")
    else:
        st.info("No prediction logs yet.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn")