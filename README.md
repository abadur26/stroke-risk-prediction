# 🧠 Stroke Risk Prediction System

A machine learning web application that predicts the risk of stroke based on patient health data. The system uses ensemble learning methods to provide real-time risk assessments.

## 📊 Overview

Stroke is a leading cause of disability worldwide. Early prediction can help in preventive healthcare. This application allows healthcare providers and individuals to input health metrics and receive an immediate stroke risk assessment.

### Key Features
- ✅ Real-time stroke risk prediction
- 📈 Probability score with visual indicator
- 🎯 High accuracy using Random Forest algorithm
- 📱 Responsive web interface
- 🔒 Data privacy (no data storage)

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 82.97% |
| ROC-AUC | 76.19% |
| Precision (Stroke) | 5% |
| Recall (Stroke) | 69% |

## 🛠️ Tech Stack

**Backend**
- Python 3.10+
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)

**Frontend**
- HTML5
- CSS3
- JavaScript (Fetch API)

**ML Techniques**
- SMOTE (Handling imbalanced data)
- Feature Scaling
- Hyperparameter Tuning (GridSearchCV)
- Ensemble Methods

## 📁 Project Structure
stroke-prediction-system/
├── app.py # Flask web application
├── train_model.py # Model training script
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore file
├── README.md # Project documentation
├── notebooks/
│ └── model_training.ipynb # Jupyter notebook
├── templates/
│ └── index.html # Web interface
└── static/
└── style.css # Custom styles


## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/stroke-prediction-system.git
cd stroke-prediction-system

Step 2: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Train the model
python train_model.py

Step 5: Run the application
python app.py

Step 6: Open browser
Navigate to http://localhost:5000


📊 Model Training Details
Dataset
Source: Healthcare dataset
Samples: 43,400 patient records
Features: 11 health indicators

Features Used

Feature	                Description
Age	                Patient's age in years
Hypertension	    0 = No, 1 = Yes
Heart Disease	    0 = No, 1 = Yes
Avg Glucose Level	Blood glucose level (mg/dL)
BMI	                Body Mass Index (kg/m²)
Gender	            Male/Female
Marriage Status	    Married/Not Married
Work Type	        Employment category
Residence Type	    Rural/Urban
Smoking Status	    Smoking history



Data Preprocessing

1. Handled missing values (median for BMI, mode for smoking_status)
2. Label encoding for categorical variables
3. Train-test split (80-20 with stratification)
4. SMOTE oversampling for class imbalance
5. StandardScaler for feature normalization


Model Comparison
Model	                Accuracy	            ROC-AUC
Logistic Regression	    78.64%	                75.68%
Decision Tree	        76.30%	                78.36%
Random Forest	        82.97%	                76.19%


🎯 How to Use

1. Fill the form with patient health information
2. Click "Predict Stroke Risk" button
3. View results showing:
    Risk level (High/Low)
    Probability percentage
    Health recommendation


📈 Sample Prediction
Input Example:

Age: 67 years
Hypertension: Yes
Heart Disease: Yes
Glucose Level: 228.69 mg/dL
BMI: 36.6


Output:
Risk Level: High
Probability: 85.3%
Recommendation: Please consult a healthcare provider


🔮 Future Improvements
Add more advanced models (XGBoost, Neural Networks)
Implement API endpoints for third-party integration
Add user authentication for patient history tracking
Deploy to cloud (Heroku/AWS)
Add more visualization charts
Mobile app development


📝 License
MIT License - Free for personal and commercial use


👨‍💻 Author
GitHub: https://github.com/abadur26
LinkedIn: www.linkedin.com/in/teamlancerbd

⭐ Acknowledgments
Dataset source (if public)
Scikit-learn documentation
Flask community

🤝 Contributing
Contributions, issues, and feature requests are welcome!

📧 Contact
For questions or feedback, please open an issue or contact directly.