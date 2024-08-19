import streamlit as st
import pandas as pd
import altair as alt

# Sample DataFrame
data = {
    'Time_bins': ['2024-08-01', '2024-08-01', '2024-08-02', '2024-08-02', '2024-08-03', '2024-08-03'],
    'Categorie': ['0-5min', '5-10min', '0-5min', '>10min', '5-10min', '>10min'],
    'Counts': [10, 5, 8, 3, 12, 7]
}

df = pd.DataFrame(data)

# Create a stacked bar chart with Altair
chart = alt.Chart(df).mark_bar().encode(
    x='Time_bins:O',
    y='sum(Counts):Q',
    color='Categorie:N',
    tooltip=['Time_bins', 'Categorie', 'Counts']
).properties(
    title='Stacked Bar Chart Example',
    width=600
).configure_title(
    fontSize=18,
    anchor='start'
)

# Display in Streamlit
st.altair_chart(chart, use_container_width=True)

