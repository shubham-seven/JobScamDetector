#Job Scam Detector

Project Overview:
This project, Job Scam Detector, is a machine learning-powered web application . The app helps users identify potentially fraudulent job listings by analyzing job data and predicting the likelihood of fraud. Built using Streamlit, it provides an interactive interface for both bulk predictions (via CSV uploads) and real-time scanning of individual job postings. The project leverages a pre-trained XGBoost model, advanced text preprocessing with NLTK, and explainability through SHAP to deliver actionable insights for job seekers and recruiters.

Key Features & Technologies Used:
Key Features:

Bulk Prediction with Dashboard: Upload a CSV file of job listings to predict fraud probability, complete with visualizations like fraud probability distribution and a genuine vs. fraudulent pie chart.
Real-Time Job Scanner: Enter details of a single job listing to instantly scan for fraud risk, with a fraud probability score.
Model Retraining: Retrain the XGBoost model with new labeled data (works locally; limited in Streamlit Cloud due to write permissions).
Explainability with SHAP: Use SHAP to understand why a job listing was classified as fraudulent or genuine, with visual force plots.
Email Alerts: Send email notifications for job listings with a fraud probability greater than 80%.

Technologies Used:

Python: Core programming language.
Streamlit (v1.32.0): For building the interactive web app.
XGBoost (v2.1.4): Machine learning model for fraud prediction.
NLTK (v3.8.1): For text preprocessing (tokenization, lemmatization, stopwords removal).
SHAP (v0.48.0): For model explainability.
Pandas (v2.2.3) & NumPy (v1.26.4): For data manipulation.
Plotly (v5.22.0) & Matplotlib (v3.8.4): For interactive and static visualizations.
Scikit-learn (v1.4.2): For TF-IDF vectorization and other utilities.
Joblib (v1.4.2): For loading the model and preprocessors.

Setup Instructions (Step-by-Step)
Follow these steps to run the Job Scam Detector locally on your machine.

Set Up a Virtual Environment (optional but recommended):

Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
Install the required packages using the provided requirements.txt:pip install -r requirements.txt

Test the App:
Use the "CSV Prediction" page to upload a CSV file (ensure it has title and description columns).
Use the "Real-Time Job Scanner" page to scan individual job listings.
Explore other features like retraining, explainability, and email alerts.

Live Demo:
The app is deployed on Streamlit Community Cloud. Check it out here: 
Live Demo URL: jobscamdetector-butekwdtwncv2mxhkmmx4m.streamlit.app

Project Structure:
app.py: Main Streamlit app file.
requirements.txt: List of Python dependencies with specific versions.
job_fraud_xgb_revised_threshold_0_2.pkl, tfidf_desc_revised.pkl, tfidf_comp_revised.pkl, tfidf_title_revised.pkl, fraud_rate.pkl: Pre-trained model and preprocessor files.

Clone the Repository:
git clone https://github.com/shubham-seven/JobScamDetector.git
cd JobScamDetector

Run the Streamlit App:
Start the app with Streamlit:streamlit run app.py
Open your browser and go to http://localhost:8501 to access the app.

Author:
Developed by Interpretive Edge for the Anveshan hackathon, June 2025.
