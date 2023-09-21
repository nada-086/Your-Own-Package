import streamlit as st
import requests
from streamlit_lottie import st_lottie
import modeling as ml
import pandas as pd
import seaborn as sns

# Globals
data = pd.DataFrame()
original_data = pd.DataFrame()
data_desc = pd.DataFrame()
target = ''


# Configuring The Web Page
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url = load_lottie("https://lottie.host/662e8f70-36c1-4ee1-a48b-df95b59e1a24/C1dqDyJanI.json")
st.set_page_config(page_title="Your Own Package", page_icon="https://cdn-icons-png.flaticon.com/128/610/610128.png",
                   layout="wide")

# Section of Information
with st.container():
    st.title("Your Own Package :wink:")
    st.write("Here, You can load data, perform EDA, train machine learning models, and evaluate model performance.")

# Section Of Getting Data File Path From The User
with st.container():
    file_path = ''
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("# Getting Some Information")
        st.write("Make sure that the entered path is relatively to current directory or the absolute path.")
        file_path = st.text_input("Enter the File Path")
        try:
            original_data, data_desc = ml.reading_data(file_path)
        except Exception as e:
            print(e)
    with right_column:
        st_lottie(lottie_url, height=300, key="charts")

# Section Of Summarizing The Data
with st.container():
    st.write('---')
    st.write('# Take an Overview on Your Data')
    left_column, right_column = st.columns(2)
    with left_column:
        st.write('## Data Summary')
        st.write(data_desc)
    with right_column:
        st.write('## Original Data')
        st.write(original_data)

# Preprocessing The Data
with st.container():
    st.write('---')
    st.write('# Getting More Information')
    left_column, right_column = st.columns(2)
    with left_column:
        encoder = st.selectbox("How to encode categorical data??", ['Label Encoder', 'OneHot Encoder'])
        null_values = st.selectbox("How to deal with null values??", ['Remove Them', 'Use Imputer'])
    with right_column:
        scaler = st.selectbox('How to scale numerical data??', ['MinMax Scaler', 'Standard Scaler'])
        chosen = st.multiselect('Choose the columns you want to drop', original_data.columns)
    if not original_data.empty:
        data = ml.preprocess(original_data, encoder, scaler, chosen, null_values)
        target = st.radio("Choose a column to predict", data.columns)

# Visualizing The Data
with st.container():
    st.write('---')
    st.write('# Visualize Your Data')
    if not (len(data.columns) <= 1 or target == '') and target in data.columns:
        pairplot = sns.pairplot(data, hue=target)
        st.pyplot(pairplot)

# After Preprocessing
with st.container():
    st.write('---')
    st.write('# A New Data')
    if not (data.empty or target == ''):
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("## Your Data :confused:")
            st.write(original_data)
        with right_column:
            st.write("## Our New Data :smile:")
            st.write(data)

# Modeling
with st.container():
    st.write('---')
    st.write('# Modeling The Data')
    if not (data.empty or target == ''):
        best, models = ml.modeling(data, target)
        st.table(models)
        st.write("## The Best")
        st.write(best)
