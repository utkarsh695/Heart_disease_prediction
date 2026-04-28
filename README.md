📌 Heart Disease Prediction – Machine Learning Project

This project uses Machine Learning & Data Analytics to predict whether a patient is likely to have heart disease based on clinical features.
It includes Data Cleaning, EDA, Feature Engineering, Model Training, Evaluation, and a Streamlit Web App for real-time predictions.

🚀 Project Overview

Heart disease is a leading cause of death globally. Early prediction can significantly improve patient outcomes.
This project analyzes patient health data to build a model that predicts heart disease using key features such as:

Age

Gender

Chest pain type

Blood pressure

Cholesterol level

Maximum heart rate

Exercise-induced angina

And more…

🧠 Tech Stack
Component	Technology
Language	Python
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
ML Models	Logistic Regression, Random Forest
Model Saving	Pickle
Deployment	Streamlit
IDE	Jupyter Notebook / VS Code
📂 Project Structure
Heart-Disease-Prediction
│── heart.csv                      # Dataset
│── Heart_Disease_Analysis.ipynb   # Jupyter Notebook (EDA + ML)
│── app.py                         # Streamlit Application
│── heart_model.pkl                # Trained Model
│── scaler.pkl                     # StandardScaler
│── README.md                      # Project Documentation

🔎 Data Processing & Modeling Steps
🧹 Data Cleaning

Handling missing values using mean imputation

Removing duplicate rows

Feature scaling using StandardScaler()

📊 Exploratory Data Analysis

Distribution of heart disease cases

Age vs Heart Disease visualization

Correlation heatmap

Feature importance using Random Forest

🤖 Model Training

Models built & evaluated:

Model	Accuracy
Logistic Regression	~
Random Forest	(Best Model)
🩺 Sample Prediction

Model predicts "Heart Disease" or "No Heart Disease" based on input values.

🌐 Streamlit Web App

Run the web app locally:

streamlit run app.py


User enters medical details → Model returns a prediction instantly.

📈 Visualizations Included

Heart disease target distribution

Age distribution by disease status

Feature correlation heatmap

Feature importance ranking

ROC-AUC curve



Install required libraries
pip install -r requirements.txt

Run Notebook
jupyter notebook

Run Streamlit App
streamlit run app.py

🔮 Future Enhancements

Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)

Deploy on AWS / Azure / Heroku

Add more ML models such as SVM, XGBoost

Build a Flask UI + database integration

🙌 Contributing

Contributions are welcome! Please open an issue or pull request.

🏆 Acknowledgements

Dataset Source: UCI Machine Learning Repository / Kaggle Heart Disease Dataset

📜 License

This project is licensed under the MIT License.
!Alt(https://github.com/utkarsh695/Heart_disease_prediction/blob/main/Snapshotapp.png)
