import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import shap
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stSelectbox, .stTextInput, .stTextArea, .stFileUploader {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stSpinner {
        color: #4CAF50;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .dashboard-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for persisting df and X
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X' not in st.session_state:
    st.session_state.X = None

# Load the model and preprocessors
try:
    xgb_model = joblib.load('job_fraud_xgb_revised_threshold_0_2.pkl')
    tfidf_desc = joblib.load('tfidf_desc_revised.pkl')
    tfidf_comp = joblib.load('tfidf_comp_revised.pkl')
    tfidf_title = joblib.load('tfidf_title_revised.pkl')
    fraud_rate = joblib.load('fraud_rate.pkl')
    st.success("Model and preprocessors loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please ensure all .pkl files are in the same directory as this script.")
    st.stop()

# Initialize SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Utility functions for preprocessing
def advanced_preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def parse_salary(salary):
    if pd.isna(salary) or not isinstance(salary, str):
        return 0
    try:
        salary = salary.replace('$', '').replace(',', '')
        low, high = salary.split('-')
        return (float(low) + float(high)) / 2
    except:
        return 0

# Prediction function (moved from api.py)
def scan_job(job_data):
    # Convert the incoming job data to a DataFrame
    df = pd.DataFrame([job_data])

    # Preprocess the text fields
    df['description_clean'] = df['description'].apply(advanced_preprocess_text)
    df['company_profile_clean'] = df['company_profile'].apply(advanced_preprocess_text)
    df['title_clean'] = df['title'].apply(advanced_preprocess_text)

    # Transform text using TF-IDF vectorizers
    X_desc = tfidf_desc.transform(df['description_clean'])
    X_comp = tfidf_comp.transform(df['company_profile_clean'])
    X_title = tfidf_title.transform(df['title_clean'])

    # Process salary and binary features
    df['has_salary_range'] = 1 if job_data['salary_range'] else 0
    df['salary_avg'] = parse_salary(job_data['salary_range'])

    # Target encoding for categorical features
    cat_features = ['employment_type', 'required_experience', 'industry']
    for col in cat_features:
        df[f'{col}_target_enc'] = df[col].map(fraud_rate.get(col, {})).fillna(0.048715)

    # Additional features
    df['desc_length'] = df['description'].apply(len)
    df['urgent_flag'] = df['description'].str.contains('urgent|immediate|asap|now|pressing|hurry|limited time', case=False, na=False).astype(int)

    # Prepare the feature matrix
    binary_features = ['telecommuting', 'has_company_logo', 'has_questions', 'has_salary_range']
    X_binary = df[binary_features].values
    X_extra = df[['salary_avg', 'desc_length', 'urgent_flag', 'employment_type_target_enc', 'required_experience_target_enc', 'industry_target_enc']].values
    X = np.hstack((X_desc.toarray(), X_comp.toarray(), X_title.toarray(), X_binary, X_extra))

    # Make prediction
    fraud_prob = xgb_model.predict_proba(X)[:, 1][0]
    prediction = 1 if fraud_prob >= 0.2 else 0
    prediction_label = "Fraudulent" if prediction == 1 else "Genuine"

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "fraud_probability": float(fraud_prob)
    }

# Streamlit App Setup with Sidebar Navigation
st.title("Job Scam Detector")
st.sidebar.title("Navigation")
st.sidebar.markdown("### Explore Features")
page = st.sidebar.selectbox("Choose a feature:", ["CSV Prediction", "API Job Scanner", "Retrain Model", "Explain Predictions", "Email Alerts"])

# Page 1: CSV Prediction with Dashboard
if page == "CSV Prediction":
    st.markdown("## Upload a CSV file to detect potential job scams")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_columns = ['title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}. Please ensure the CSV contains 'title' and 'description'.")
            st.stop()

        st.success("CSV file uploaded successfully!")
        st.markdown("### Data Preview")
        st.dataframe(df.head())

        # Clean the Job Data
        df = df.drop_duplicates()
        if 'job_id' in df.columns:
            df = df.drop(columns=['job_id'])

        text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        categorical_columns = ['location', 'department', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna("Not Provided")
            else:
                df[col] = "Not Provided"
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = "Unknown"

        binary_columns = ['telecommuting', 'has_company_logo', 'has_questions']
        for col in binary_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0

        df['description_clean'] = df['description'].apply(advanced_preprocess_text)
        df['company_profile_clean'] = df['company_profile'].apply(advanced_preprocess_text)
        df['title_clean'] = df['title'].apply(advanced_preprocess_text)

        X_desc = tfidf_desc.transform(df['description_clean'])
        X_comp = tfidf_comp.transform(df['company_profile_clean'])
        X_title = tfidf_title.transform(df['title_clean'])

        if 'salary_range' in df.columns:
            df['has_salary_range'] = df['salary_range'].notnull().astype(int)
            df['salary_avg'] = df['salary_range'].apply(parse_salary)
        else:
            df['has_salary_range'] = 0
            df['salary_avg'] = 0

        cat_features = ['employment_type', 'required_experience', 'industry']
        for col in cat_features:
            df[f'{col}_target_enc'] = df[col].map(fraud_rate.get(col, {})).fillna(0.048715)

        df['desc_length'] = df['description'].apply(len)
        df['urgent_flag'] = df['description'].str.contains('urgent|immediate|asap|now|pressing|hurry|limited time', case=False, na=False).astype(int)

        binary_features = ['telecommuting', 'has_company_logo', 'has_questions', 'has_salary_range']
        X_binary = df[binary_features].values
        X_extra = df[['salary_avg', 'desc_length', 'urgent_flag', 'employment_type_target_enc', 'required_experience_target_enc', 'industry_target_enc']].values
        X = np.hstack((X_desc.toarray(), X_comp.toarray(), X_title.toarray(), X_binary, X_extra))

        if len(df) == 0:
            st.error("No valid data to process after preprocessing. Please check your CSV file.")
            st.stop()

        with st.spinner('Making predictions...'):
            fraud_prob = xgb_model.predict_proba(X)[:, 1]
            predictions = (fraud_prob >= 0.2).astype(int)
            df['Prediction'] = predictions
            df['Fraud_Probability'] = fraud_prob
            df['Prediction_Label'] = df['Prediction'].map({0: 'Genuine', 1: 'Fraudulent'})

        # Store df and X in session state
        st.session_state.df = df
        st.session_state.X = X

        st.markdown("## Prediction Dashboard", unsafe_allow_html=True)
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

        # Dashboard Layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Fraud Probability Distribution")
            fig = px.histogram(df, x='Fraud_Probability', nbins=20, title='Fraud Probability Distribution',
                               color_discrete_sequence=['#4CAF50'])
            fig.update_layout(bargap=0.1, xaxis_title="Fraud Probability", yaxis_title="Number of Jobs")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Genuine vs. Fraudulent Jobs")
            fraud_counts = df['Prediction_Label'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=fraud_counts.index, values=fraud_counts.values,
                                         textinfo='label+percent', marker=dict(colors=['#4CAF50', '#FF5733']))])
            fig.update_layout(title="Genuine vs. Fraudulent Jobs")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Job Listings with Fraud Predictions")
        result_df = df[['title', 'description', 'Prediction_Label', 'Fraud_Probability']]
        st.dataframe(result_df, use_container_width=True)

        st.markdown("### Top-10 Jobs with Highest Fraud Risk")
        top_10_shady = df.nlargest(10, 'Fraud_Probability')[['title', 'description', 'Prediction_Label', 'Fraud_Probability']]
        st.dataframe(top_10_shady, use_container_width=True)

        st.markdown("### Download Predictions")
        csv = df[['title', 'description', 'Prediction_Label', 'Fraud_Probability']].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="job_fraud_predictions.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Page 2: API Job Scanner (Modified to use local function)
elif page == "API Job Scanner":
    st.subheader("Real-Time Job Scanner")
    st.write("Enter a single job listing to scan for fraud.")

    with st.form("job_scan_form"):
        title = st.text_input("Job Title:", value="Software Engineer")
        description = st.text_area("Job Description:", value="Urgent hiring for software engineer! Apply now.")
        company_profile = st.text_area("Company Profile:", value="Tech Corp")
        employment_type = st.selectbox("Employment Type:", ["Full-time", "Part-time", "Contract", "Temporary", "Unknown"], index=0)
        required_experience = st.selectbox("Required Experience:", ["Entry level", "Mid-Senior level", "Associate", "Executive", "Unknown"], index=1)
        industry = st.selectbox("Industry:", ["Technology", "Finance", "Healthcare", "Education", "Unknown"], index=0)
        salary_range = st.text_input("Salary Range (e.g., 50000-70000):", value="")
        telecommuting = st.checkbox("Telecommuting", value=False)
        has_company_logo = st.checkbox("Has Company Logo", value=True)
        has_questions = st.checkbox("Has Questions", value=False)
        submit_button = st.form_submit_button("Scan Job")

    if submit_button:
        job_data = {
            "title": title,
            "description": description,
            "company_profile": company_profile,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "industry": industry,
            "salary_range": salary_range,
            "telecommuting": int(telecommuting),
            "has_company_logo": int(has_company_logo),
            "has_questions": int(has_questions)
        }

        with st.spinner("Scanning job..."):
            result = scan_job(job_data)
            st.success("Job scanned successfully!")
            st.write(f"*Prediction*: {result['prediction_label']}")
            st.write(f"*Fraud Probability*: {result['fraud_probability']:.2f}")

# Page 3: Retrain Model
elif page == "Retrain Model":
    st.subheader("Retrain the Model")
    st.write("Upload a labeled CSV file to retrain the model. The CSV must include a 'fraudulent' column with labels (0 or 1).")

    retrain_file = st.file_uploader("Choose a labeled CSV file", type="csv", key="retrain_file")

    if retrain_file is not None:
        new_data = pd.read_csv(retrain_file)
        required_columns = ['title', 'description', 'fraudulent']
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}. Please ensure the CSV contains 'title', 'description', and 'fraudulent'.")
            st.stop()

        st.success("Labeled data uploaded successfully!")
        st.markdown("### Data Preview")
        st.dataframe(new_data.head())

        if st.button("Retrain Model"):
            with st.spinner("Retraining the model..."):
                # Preprocessing
                new_data = new_data.drop_duplicates()
                if 'job_id' in new_data.columns:
                    new_data = new_data.drop(columns=['job_id'])

                text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
                categorical_columns = ['location', 'department', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
                for col in text_columns:
                    if col in new_data.columns:
                        new_data[col] = new_data[col].fillna("Not Provided")
                    else:
                        new_data[col] = "Not Provided"
                for col in categorical_columns:
                    if col in new_data.columns:
                        new_data[col] = new_data[col].fillna("Unknown")
                    else:
                        new_data[col] = "Unknown"

                binary_columns = ['telecommuting', 'has_company_logo', 'has_questions']
                for col in binary_columns:
                    if col in new_data.columns:
                        new_data[col] = new_data[col].fillna(0)
                    else:
                        new_data[col] = 0

                new_data['description_clean'] = new_data['description'].apply(advanced_preprocess_text)
                new_data['company_profile_clean'] = new_data['company_profile'].apply(advanced_preprocess_text)
                new_data['title_clean'] = new_data['title'].apply(advanced_preprocess_text)

                # Retrain TF-IDF vectorizers
                tfidf_desc = TfidfVectorizer(max_features=5000)
                tfidf_comp = TfidfVectorizer(max_features=5000)
                tfidf_title = TfidfVectorizer(max_features=5000)
                X_desc = tfidf_desc.fit_transform(new_data['description_clean'])
                X_comp = tfidf_comp.fit_transform(new_data['company_profile_clean'])
                X_title = tfidf_title.fit_transform(new_data['title_clean'])

                if 'salary_range' in new_data.columns:
                    new_data['has_salary_range'] = new_data['salary_range'].notnull().astype(int)
                    new_data['salary_avg'] = new_data['salary_range'].apply(parse_salary)
                else:
                    new_data['has_salary_range'] = 0
                    new_data['salary_avg'] = 0

                # Recalculate fraud rates for target encoding
                fraud_rate = {}
                cat_features = ['employment_type', 'required_experience', 'industry']
                for col in cat_features:
                    fraud_rate[col] = new_data.groupby(col)['fraudulent'].mean().to_dict()
                for col in cat_features:
                    new_data[f'{col}_target_enc'] = new_data[col].map(fraud_rate[col]).fillna(new_data['fraudulent'].mean())

                new_data['desc_length'] = new_data['description'].apply(len)
                new_data['urgent_flag'] = new_data['description'].str.contains('urgent|immediate|asap|now|pressing|hurry|limited time', case=False, na=False).astype(int)

                binary_features = ['telecommuting', 'has_company_logo', 'has_questions', 'has_salary_range']
                X_binary = new_data[binary_features].values
                X_extra = new_data[['salary_avg', 'desc_length', 'urgent_flag', 'employment_type_target_enc', 'required_experience_target_enc', 'industry_target_enc']].values
                X = np.hstack((X_desc.toarray(), X_comp.toarray(), X_title.toarray(), X_binary, X_extra))
                y = new_data['fraudulent'].values

                # Retrain the XGBoost model
                xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                xgb_model.fit(X, y)

                # Save the updated models and vectorizers
                joblib.dump(xgb_model, 'job_fraud_xgb_revised_threshold_0_2.pkl')
                joblib.dump(tfidf_desc, 'tfidf_desc_revised.pkl')
                joblib.dump(tfidf_comp, 'tfidf_comp_revised.pkl')
                joblib.dump(tfidf_title, 'tfidf_title_revised.pkl')
                joblib.dump(fraud_rate, 'fraud_rate.pkl')

                st.success("Model retrained and saved successfully! Please restart the app to use the updated model.")

# Page 4: Explain Predictions with SHAP
elif page == "Explain Predictions":
    st.subheader("Explain Predictions with SHAP")
    st.write("Select a job to see why it was classified as fraudulent or genuine.")

    # Check session state for df and X
    if st.session_state.df is None or st.session_state.X is None:
        st.error("Please run a CSV prediction first on the 'CSV Prediction' page.")
    else:
        df = st.session_state.df
        X = st.session_state.X
        job_titles = df['title'].tolist()
        selected_job = st.selectbox("Select a job to explain:", job_titles)
        selected_idx = df[df['title'] == selected_job].index[0]

        shap_values = explainer.shap_values(X[selected_idx:selected_idx+1])
        shap_expected_value = explainer.expected_value

        st.markdown("### SHAP Force Plot")
        plt.figure()
        shap.force_plot(shap_expected_value, shap_values, X[selected_idx:selected_idx+1], matplotlib=True, show=False)
        st.pyplot(plt)

        # Define binary_features
        binary_features = ['telecommuting', 'has_company_logo', 'has_questions', 'has_salary_range']
        
        feature_names = (tfidf_desc.get_feature_names_out().tolist() +
                         tfidf_comp.get_feature_names_out().tolist() +
                         tfidf_title.get_feature_names_out().tolist() +
                         binary_features +
                         ['salary_avg', 'desc_length', 'urgent_flag', 'employment_type_target_enc',
                          'required_experience_target_enc', 'industry_target_enc'])
        st.write("Note: Features are in order: description, company profile, title, binary features, and extra features.")

# Page 5: Email Alerts for High-Risk Jobs
elif page == "Email Alerts":
    st.subheader("Email Alerts for High-Risk Jobs")
    st.write("Enter your email credentials to send alerts for jobs with fraud probability > 80%.")

    # Check session state for df
    if st.session_state.df is None:
        st.error("Please run a CSV prediction first on the 'CSV Prediction' page.")
    else:
        df = st.session_state.df
        recipient_email = st.text_input("Recipient Email Address:")
        sender_email = st.text_input("Sender Email Address (e.g., your Gmail):")
        sender_password = st.text_input("Sender Email Password (e.g., Gmail App Password):", type="password")

        if st.button("Send Email Alert"):
            if not recipient_email or not sender_email or not sender_password:
                st.error("Please fill in all email fields.")
            else:
                high_risk_jobs = df[df['Fraud_Probability'] > 0.8][['title', 'description', 'Fraud_Probability']]

                if len(high_risk_jobs) == 0:
                    st.warning("No high-risk jobs (fraud probability > 80%) found.")
                else:
                    subject = "High-Risk Job Listings Alert"
                    body = "The following jobs have a fraud probability greater than 80%:\n\n"
                    for idx, row in high_risk_jobs.iterrows():
                        body += f"Title: {row['title']}\nDescription: {row['description']}\nFraud Probability: {row['Fraud_Probability']:.2f}\n\n"

                    msg = MIMEMultipart()
                    msg['From'] = sender_email
                    msg['To'] = recipient_email
                    msg['Subject'] = subject
                    msg.attach(MIMEText(body, 'plain'))

                    try:
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, recipient_email, msg.as_string())
                        server.quit()
                        st.success(f"Email alert sent to {recipient_email} with {len(high_risk_jobs)} high-risk jobs!")
                    except Exception as e:
                        st.error(f"Failed to send email: {e}. Ensure your credentials are correct and you’re using an app-specific password for Gmail.")

# Footer
st.markdown("---")
st.markdown("*App developed by Interpretive Edge*")
