import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import streamlit as st

# Page settings
st.set_page_config(page_title="FORAGER Analytics", layout="wide")

st.markdown("""
<style>

/* Background & text */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #E5E7EB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1c1f26;
    color: #E5E7EB;
    border-right: 1px solid #374151;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #F9FAFB;
    font-weight: 600;
}

/* Metric box styling */
[data-testid="stMetric"] {
    background-color: #1f2937;
    border-left: 6px solid #3B82F6;  /* blue by default */
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 10px;
}

[data-testid="stMetricLabel"] {
    color: #A1A1AA;
}

[data-testid="stMetricValue"] {
    color: #10B981;
    font-size: 24px;
    font-weight: bold;
}

/* Chart container */
.element-container:has(.plotly-graph-div) {
    background-color: #1f232b;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #2d2f33;
}

/* Dataframe tables */
.css-1d391kg {
    background-color: #1f2937;
    border-radius: 6px;
    padding: 10px;
    color: #D1D5DB;
    font-size: 14px;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #1f232b;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 0.5rem;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #6b7280;
    border-radius: 4px;
}
::-webkit-scrollbar-track {
    background: #1e1e1e;
}

</style>
""", unsafe_allow_html=True)

st.title("📊 FORAGER Analytics Dashboard")

# PLL Statistics
st.subheader("🔁 PLL Retry & Lock-in Rates")
pll_data = pd.DataFrame({
    'Run ID': [f'Run {i}' for i in range(1, 11)],
    'Retry Rate (%)': np.random.randint(20, 80, size=10),
    'Lock-in Rate (%)': np.random.randint(30, 95, size=10)
})
st.dataframe(pll_data)
st.altair_chart(
    alt.Chart(pll_data).transform_fold(
        ['Retry Rate (%)', 'Lock-in Rate (%)'],
        as_=['Type', 'Value']
    ).mark_line(point=True).encode(
        x='Run ID',
        y='Value:Q',
        color='Type:N'
    ).properties(height=300),
    use_container_width=True
)

# Confidence / Similarity
st.subheader("📈 Confidence & Similarity Trends")
conf_data = pd.DataFrame({
    'Attempt': list(range(1, 11)),
    'Confidence': np.random.uniform(0.6, 0.95, 10),
    'Similarity': np.random.uniform(0.5, 0.9, 10)
})
st.altair_chart(
    alt.Chart(conf_data).transform_fold(
        ['Confidence', 'Similarity'],
        as_=['Metric', 'Value']
    ).mark_line(point=True).encode(
        x='Attempt',
        y='Value:Q',
        color='Metric:N'
    ).properties(height=300),
    use_container_width=True
)

# BS Label Pie Chart
st.subheader("🧪 BS Label Distribution")
bs_data = pd.DataFrame({
    'Label': ['Supported', 'Unsupported', 'Contradicted'],
    'Count': [45, 30, 25]
})
st.plotly_chart(
    px.pie(bs_data, names='Label', values='Count', title="Label Distribution"),
    use_container_width=True
)

# Rephrased Output Comparison
st.subheader("🧠 Output Comparison Across Rephrases")
rephrase_data = pd.DataFrame({
    'Prompt': [f'Prompt {i}' for i in range(1, 6)],
    'Original Score': np.random.uniform(0.65, 0.9, 5),
    'Rephrased Score': np.random.uniform(0.7, 0.95, 5)
})
st.dataframe(rephrase_data)
st.altair_chart(
    alt.Chart(rephrase_data).transform_fold(
        ['Original Score', 'Rephrased Score'],
        as_=['Version', 'Score']
    ).mark_bar().encode(
        x='Prompt:N',
        y='Score:Q',
        color='Version:N'
    ).properties(height=300),
    use_container_width=True
)

# Batch vs Single-Run Mode
st.subheader("⚙️ Batch vs Single-Run Comparison")
mode_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Confidence', 'Runtime (s)'],
    'Batch Mode': [0.88, 0.9, 30],
    'Single-Run': [0.81, 0.85, 15]
})
st.dataframe(mode_data)
st.bar_chart(mode_data.set_index('Metric'), use_container_width=True)