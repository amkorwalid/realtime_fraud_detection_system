import streamlit as st
import time
import pandas as pd

# metrics : 4 cols
# end to end latency
# num request per second 
# num of frauds per se
# total resquests

# plots : 2 cols
# line plot of request frequency
# line plot of fraud frequency

# 1. Initialize data tracking inside Streamlit's session state
if "faq_data" not in st.session_state:
    st.session_state.faq_data = pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9, 10], columns=["Fraud Metric"])

st.title("Fraud Detection Dashboard")
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric(label="Total Transactions", value=1000)
with metrics_col2:
    st.metric(label="End to End Latency", value="120ms")
with metrics_col3:
    st.metric(label="Requests per Second", value=50)
with metrics_col4:
    st.metric(label="FRAUDS per Second", value=2)

line_plot_col1, line_plot_col2 = st.columns(2)

with line_plot_col1:
    p = st.line_chart(data=st.session_state.faq_data)

with line_plot_col2:
    st.line_chart(data=[5, 4, 3, 2, 1])

for i in range(40):
    time.sleep(1)

    # Create the new row point
    new_row = pd.DataFrame([i], columns=["Fraud Metric"])
    
    # Concatenate the new row to our persistent DataFrame state
    st.session_state.faq_data = pd.concat([st.session_state.faq_data, new_row], ignore_index=True)
    
    # Re-draw the chart directly into its designated screen container slot
    p.line_chart(st.session_state.faq_data)
    