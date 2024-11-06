# 📱 Mobile Price Prediction using Random Forest Regressor 🌲💻

This project uses machine learning to predict mobile phone prices based on various features such as ratings, RAM, ROM, camera quality, battery power, and more. The model utilizes a **Random Forest Regressor** algorithm, which is trained using the features from the mobile dataset to predict the price.

## 📋 **Project Overview**

In this project, we load a dataset containing information about mobile phones and preprocess the data to train a machine learning model to predict the price of a mobile phone. The features considered in the model include:

- **Brand**: 📱 Brand of the mobile phone.
- **Ratings**: ⭐ Average user rating for the mobile.
- **RAM**: 🧠 The amount of RAM (in GB) the mobile has.
- **ROM**: 💾 The amount of internal storage (in GB) of the mobile.
- **Mobile Size**: 📏 The screen size (in inches).
- **Primary Camera**: 📷 Megapixels of the mobile's rear camera.
- **Selfie Camera**: 🤳 Megapixels of the mobile's front camera.
- **Battery Power**: 🔋 Battery power (in mAh).
- **Price**: 💵 The target variable, representing the price of the mobile phone.

## 🚀 **Project Flow**

1. **Data Loading and Exploration**:
   - The dataset is loaded from a CSV file 📂.
   - The dataset is explored to understand its structure and identify any missing values 🧐.
   - Columns are renamed and standardized for easier analysis 🧮.

2. **Data Preprocessing**:
   - Missing values are handled by filling categorical columns with the mode and numerical columns with the mean 💡.
   - Label encoding is applied to the 'brand_me' column to convert categorical values into numeric values 🔢.

3. **Feature Selection**:
   - The relationship between the features and the target variable (Price) is explored using correlation analysis 🔍.
   - **Recursive Feature Elimination (RFE)** is used to select the most important features ✂️.

4. **Model Training**:
   - The data is split into training and testing sets 🧑‍💻.
   - A **Random Forest Regressor** is trained on the selected features 🌲.
   - The model is evaluated using various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score 📊.

5. **Model Saving**:
   - After training, the model is saved using **joblib** for later use in predictions 💾.

## 🔧 **Project Setup**

### **Prerequisites**

You need the following libraries to run this project:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `tensorflow`
- `joblib`
- `scikit-learn`

You can install these dependencies using the following pip command:

```bash
pip install pandas numpy seaborn matplotlib tensorflow joblib scikit-learn
