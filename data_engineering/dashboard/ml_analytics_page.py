import streamlit as st
import time
import pandas as pd


# 4 cols :
# TP TN FP FN + their frequency
# plot FP FN
# Live inference


st.title("ML Analytics Dashboard")
metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric(label="True Positives", value=100)
    
with metrics_col2:
    st.metric(label="True Negatives", value=200)

with metrics_col3:
    st.metric(label="False Positives", value=300)

with metrics_col4:
    st.metric(label="False Negatives", value=400)



chart_data = pd.DataFrame({
    "False Positives": [300, 180, 270, 100, 360],
    "False Negatives": [400, 250, 320, 150, 420]
})

plot_fp_fn = st.line_chart(data=chart_data)


for i in range(60):
    time.sleep(1)

    # Create the new row point
    new_row = pd.DataFrame([[i*50, i*55]], columns=["False Positives", "False Negatives"])
    
    # Concatenate the new row to our persistent DataFrame state
    chart_data = pd.concat([chart_data, new_row], ignore_index=True)
    
    # Re-draw the chart directly into its designated screen container slot
    plot_fp_fn.line_chart(chart_data)