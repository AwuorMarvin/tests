
import pandas as pd
import streamlit as st
import plotly.express as px

df = pd.read_csv('sales_data_sample.csv')

st.title("Sales Data Dashboard")

st.write("Preview of data:")
st.dataframe(df.head())

# Example interactive plot
fig1 = px.bar(df, x='PRODUCTLINE', y='SALES', title='Sales by Product')
fig2 = px.bar(df, x='COUNTRY', y='SALES', title='Sales by Country')

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)