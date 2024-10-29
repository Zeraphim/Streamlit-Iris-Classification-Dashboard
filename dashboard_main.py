# -------------------------

# Import libraries
import streamlit as st
import pandas as pd
import altair as alt

# -------------------------

# Page configuration
st.set_page_config(
    page_title="Iris Classification", 
    page_icon="assets/icon/icon.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# -------------------------

# Sidebar

page_selection = 'about'

with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Iris Classification')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True):
        page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True):
        page_selection = 'dataset'

    if st.button("EDA", use_container_width=True):
        page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True):
        page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True): 
        page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True): 
        page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True):
        page_selection = "conclusion"

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("by: `Zeraphim`")

# -------------------------

# Data

# Load data
dataset = pd.read_csv("data/IRIS.csv")

# -------------------------

# Plots

# -------------------------

# Pages

# About Page
if page_selection == "about":
    st.header("ℹ️ About")

# Dataset Page
elif page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("IRIS Flower Dataset")
    st.write("")

# EDA Page
elif page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')
        
# Data Cleaning Page
elif page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

# Machine Learning Page
elif page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

# Prediction Page
elif page_selection == "prediction":
    st.header("👀 Prediction")

# Conclusions Page
elif page_selection == "conclusion":
    st.header("📝 Conclusion")