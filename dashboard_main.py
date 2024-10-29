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

    st.markdown(""" 

    This notebook explores the Iris flower dataset by performing:
    1. Exploratory Data Analysis (EDA)
    2. Data Pre-processing
    3. Training two supervised classification models: **Decision Tree Classifier** and **Random Forest Regressor**
    4. Making predictions on a new **unseen data**


    """)

# Dataset Page
elif page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""

    The **Iris flower dataset** was introduced by **Ronald Fisher** in 1936, it is a dataset used widely in machine learning. Originally collected by Edgar Anderson, it includes **50 samples** each from three Iris species (**Iris Setosa**, **Iris Virginica**, and **Iris Versicolor**).  

    For each sample, four features are measured: sepal length, sepal width, petal length, and petal width (in centimeters). This dataset is commonly used to test classification techniques like support vector machines. The same dataset that is used for this data science activity was uploaded to Kaggle by the user named **Mathnerd**.

    **Content**  
    The dataset has **150** rows containing **5 primary attributes** that are related to the iris flower, the columns are as follows: **Petal Length**, **Petal Width**, **Sepal Length**, **Sepal width**, and **Class(Species)**.

    `Link:` https://www.kaggle.com/datasets/arshid/iris-flower-dataset            
                
    """)

    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(dataset, use_container_width=True)

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