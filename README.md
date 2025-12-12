# GarmAI — Garment Worker Productivity Predictor

GarmAI is an AI-powered web application that predicts garment worker productivity using a machine learning model trained on real production factors. The system analyzes inputs such as SMV, WIP, overtime, incentives, idle time, department, day, workforce size, and more to generate a productivity score and classify it as **Low**, **Medium**, or **High**. The application features a modern UI with glassmorphism styling, an animated gauge, smooth transitions, and a clean input workflow, making it both functional and visually appealing.

## 🧠 Overview

The core of GarmAI is a **Random Forest Regressor** trained using scikit-learn. Input values are processed using pandas, encoded using one-hot encoding for categorical fields, aligned to the original training feature order, and then fed to the model. The predicted score is displayed with a circular gauge, score bar, clear color-coded category badge, interpretation text, and recommended improvement steps. Optional notes allow additional context for each prediction.

## 🚀 Features

- AI-based productivity prediction (score between 0–1)
- Automatic categorization: Low / Medium / High  
- Modern glass UI with card-based layout  
- Animated circular gauge and progress bar  
- Autofill example button for demo purposes  
- Client-side + server-side validation  
- Helpful hints and tooltips for each field  
- Clean output page with suggestions and interpretation  

## 🏗️ Technologies Used

**Frontend:**  
HTML5, CSS3, JavaScript, Jinja2

**Backend:**  
Python, Flask

**Machine Learning:**  
Scikit-learn, Random Forest, Pandas, NumPy, Joblib

## 📁 Project Structure

GarmAI/

│

├── app.py

├── rf_productivity_model.joblib

├── rf_productivity_features.joblib

│

├── templates/

│ ├── base.html

│ ├── home.html

│ ├── predict.html

│ ├── output.html

│ └── about.html

│

└── static/

├── styles.css

└── images/

markdown
Copy code

## ⚙️ How It Works

1. The user enters production details such as date, SMV, WIP, idle time, workers, department, and day.  
2. The backend validates the inputs, processes them, and encodes categorical values.  
3. The processed features are aligned to match the structure of the training dataset.  
4. The Random Forest model predicts a productivity score.  
5. The result page displays:
   - Score meter  
   - Category badge  
   - Interpretation message  
   - Recommended steps  
   - Optional user-provided notes  

## 📦 Installation

1.Install dependencies:

pip install flask scikit-learn pandas numpy joblib

2.Run the application:

bash

Copy code

python app.py

3.Open in browser:

Copy code

http://127.0.0.1:5000

## 🤖 Machine Learning Details
Model: Random Forest Regressor

Trained using industry-related garment production data

Handles categorical + numeric features efficiently

Robust to noise and non-linear relationships

Features encoded using one-hot encoding

Model + feature layout saved using Joblib for consistent predictions

## 🔮 Future Enhancements
PDF export for prediction reports

Historical analytics dashboard

Worker-specific performance tracking

Batch CSV prediction upload

REST API for enterprise integration

Login + user management

## 📬 Contact

Developer: Sreeshanth Sai

Email: sreeshanthsai10@gmail.com
