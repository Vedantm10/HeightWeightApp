Height → Weight Predictor

A Machine Learning-based Streamlit web application that predicts a person's weight based on their height.

📌 Project Overview

The Height → Weight Predictor uses multiple machine learning regression algorithms to learn the relationship between height and weight. The application allows the user to enter their height in feet and inches and predicts the expected weight in kilograms.

The application also compares different regression models using standard evaluation metrics and automatically selects the model with the highest R² score.

🚀 Features
📏 Enter height in feet and inches
⚖️ Predict weight in kilograms
🤖 Uses multiple Machine Learning regression models
📊 Compares model performance
🏆 Automatically selects the best-performing model based on R²
🎨 Interactive and user-friendly Streamlit interface
⚠️ Displays a warning when the entered height is outside the dataset range
📈 Displays model evaluation metrics
🗑️ Clear prediction option
📱 Responsive web interface
🤖 Machine Learning Models

The application uses the following regression models:

Linear Regression
Ridge Regression
Lasso Regression
Decision Tree Regression
Random Forest Regression
Evaluation Metrics

The models are evaluated using:

MAE – Mean Absolute Error
MSE – Mean Squared Error
RMSE – Root Mean Squared Error
R² Score – Coefficient of Determination

The model with the highest R² score is selected for prediction.

📂 Dataset

Dataset used:

SOCR Height Weight Dataset

The dataset contains:

Index
Height(Inches)
Weight(Pounds)

The model is trained using the original height and weight units from the dataset.

🔄 Working Process
User enters height
        ↓
Feet + Inches converted to Inches
        ↓
Machine Learning Model
        ↓
Predicted Weight in Pounds
        ↓
Convert Pounds → Kilograms
        ↓
Display Predicted Weight
🛠️ Technologies Used
Python
Streamlit
Pandas
NumPy
Scikit-learn
HTML
CSS
📁 Project Structure
HeightWeightApp/
│
├── SOCR-HeightWeight.csv
├── streamlit_app.py
├── style.css
├── height_card.html
├── weight_card.html
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Installation
1. Clone the repository
git clone https://github.com/Vedantm10/HeightWeightApp.git
2. Open the project folder
cd HeightWeightApp
3. Install dependencies
py -3 -m pip install -r requirements.txt
4. Run the Streamlit application
py -3 -m streamlit run streamlit_app.py

The application will open in your browser.

📊 Model Training

The dataset is divided into:

80% Training Data
20% Testing Data
train_test_split(test_size=0.2, random_state=42)

Linear, Ridge, and Lasso regression use feature scaling with StandardScaler.

👨‍💻 Author

Vedant More

B.Tech – Computer Science / AI & Data Science

📜 License

This project is created for academic and educational purposes.
