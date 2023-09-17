## Project Overview

Welcome to this exciting project! Here, I'll provide an overview of the work I've accomplished.

### Exploring PyCaret

The initial phase of this project involved an in-depth exploration of PyCaret, a remarkable open-source, low-code machine learning library for Python. Within this stage, I achieved the following key objectives:

- **Data Handling**: I learned how to use PyCaret to efficiently load and preprocess data, streamlining the data preparation process.

- **Exploratory Data Analysis (EDA)**: PyCaret enabled me to conduct EDA effortlessly. I explored data distributions, relationships, and outliers, gaining valuable insights into the datasets.

- **Model Training**: Leveraging PyCaret's capabilities, I successfully trained a variety of machine learning models, including regression and classification algorithms.

- **Model Evaluation**: I assessed the performance of these models using PyCaret's built-in evaluation tools. This step allowed me to choose the best-performing model for each dataset.

- **AutoML**: One of the highlights of this exploration was utilizing PyCaret's AutoML functionality. It significantly simplified the process of identifying the most suitable machine learning models, saving time and effort.

### Building My Own Package

With a solid foundation in PyCaret, I transitioned to the next phase of the project, which involved creating a custom Python package tailored to my needs. This package was designed to offer the following functionalities:

- **Data Handling**: My package can load and preprocess data efficiently, ensuring that it's ready for machine learning tasks.

- **Automated Model Selection**: To simplify model selection, my package incorporates automated techniques for choosing between regression and classification models.

- **User Control**: I made sure that users can select specific machine learning models based on their requirements, providing flexibility and customization options.

### Creating a User-Friendly Web App

To make my package even more accessible, I integrated Streamlit, a Python library for building interactive web applications. The web app I developed allows users to:

- **Upload Data**: Users can easily upload their datasets directly into the app.

- **Specify Target Variables**: The app empowers users to select their target variables, an essential step in the machine learning process.

- **Choose Models**: Users have the freedom to choose the machine learning models they want to apply to their data, tailoring the analysis to their unique needs.

### Testing and Validation

In the final phase of this project, I conducted rigorous testing and validation to ensure the reliability and robustness of my custom package. I assessed its performance on various datasets, ensuring that it consistently identified the most suitable machine learning models for each scenario. Furthermore, I meticulously checked for errors to guarantee the package's stability and effectiveness.


## Prerequisites and Setup

Before you can get started with this project, you'll need to ensure you have the necessary libraries and setup in place. Follow the steps below to set up your environment.

### 1. Python Environment

This project is developed using Python. Make sure you have Python 3.10 installed on your system. You need to use this old version of python as PyCaret is only supported in python 3.8, 3.9. 3.10. You can download Python 3.10 from: [Python Downloads](https://www.python.org/downloads/release/python-3100/)

### 2. Required Libraries

```bash
pip install streamlit streamlit-lottie requests pycaret
```

- [PyCaret](https://pycaret.org/): An open-source machine learning library for low-code automation.
- [Streamlit](https://streamlit.io/): A Python library for creating interactive web applications.
- [Streamlit-Lottie](https://pypi.org/project/streamlit-lottie/): To load lottie animations

### 3. Clone the Repository

```bash
git clone https://github.com/nada-086/Your-Own-Package
```

### 4. Running the Project

```bash
streamlit run App.py
```
