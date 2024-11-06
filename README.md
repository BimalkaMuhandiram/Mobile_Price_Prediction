📱 Mobile Price Prediction using Random Forest Regressor 🌲💻
This project uses machine learning to predict mobile phone prices based on various features such as ratings, RAM, ROM, camera quality, battery power, and more. The model utilizes a Random Forest Regressor algorithm, which is trained using the features from the mobile dataset to predict the price.

📋 Project Overview
In this project, we load a dataset containing information about mobile phones and preprocess the data to train a machine learning model to predict the price of a mobile phone. The features considered in the model include:

Brand: 📱 Brand of the mobile phone.
Ratings: ⭐ Average user rating for the mobile.
RAM: 🧠 The amount of RAM (in GB) the mobile has.
ROM: 💾 The amount of internal storage (in GB) of the mobile.
Mobile Size: 📏 The screen size (in inches).
Primary Camera: 📷 Megapixels of the mobile's rear camera.
Selfie Camera: 🤳 Megapixels of the mobile's front camera.
Battery Power: 🔋 Battery power (in mAh).
Price: 💵 The target variable, representing the price of the mobile phone.
🚀 Project Flow
Data Loading and Exploration:

The dataset is loaded from a CSV file 📂.
The dataset is explored to understand its structure and identify any missing values 🧐.
Columns are renamed and standardized for easier analysis 🧮.
Data Preprocessing:

Missing values are handled by filling categorical columns with the mode and numerical columns with the mean 💡.
Label encoding is applied to the 'brand_me' column to convert categorical values into numeric values 🔢.
Feature Selection:

The relationship between the features and the target variable (Price) is explored using correlation analysis 🔍.
Recursive Feature Elimination (RFE) is used to select the most important features ✂️.
Model Training:

The data is split into training and testing sets 🧑‍💻.
A Random Forest Regressor is trained on the selected features 🌲.
The model is evaluated using various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score 📊.
Model Saving:

After training, the model is saved using joblib for later use in predictions 💾.
🔧 Project Setup
Prerequisites
You need the following libraries to run this project:

pandas
numpy
seaborn
matplotlib
tensorflow
joblib
scikit-learn
You can install these dependencies using the following pip command:

bash
Copy code
pip install pandas numpy seaborn matplotlib tensorflow joblib scikit-learn
🏃‍♂️ Running the Project
Data Loading: Ensure that the dataset file, Mobile Price Prediction.csv, is available in the same directory 📂.
Execute the script: Run the Python script that includes the data preprocessing, model training, and evaluation code 💻.
🧪 Model Training and Evaluation
The Random Forest model is trained and evaluated on the test dataset. Below are the evaluation results:

Mean Absolute Error (MAE): 3609.47 💥
Mean Squared Error (MSE): 299,956,664.83 📉
Root Mean Squared Error (RMSE): 17,319.26 📊
R² Score: 0.87 (indicating that 87% of the variance in price is explained by the features) 💪
💾 Saving and Loading the Model
The trained model is saved to a file (random_forest_regressor_model_rfe.pkl) using the joblib.dump() method. This model can later be loaded for predictions.

To load the model:

python
Copy code
loaded_rf_regressor = joblib.load('random_forest_regressor_model_rfe.pkl')
⚙️ Features and Target Variable
Features: brand_me, ratings, ram, rom, mobile_size, primary_cam, selfi_cam, battery_power
Target Variable: price
🛠️ Data Preprocessing
🧹 Handling Missing Values
Categorical columns are filled with the mode (most frequent value).
Numerical columns are filled with the mean value.
🔢 Label Encoding
The brand_me column, which contains the brand names, is label-encoded to convert the text values into numerical values for model training.

🎯 Feature Selection
Using Recursive Feature Elimination (RFE), the most important features are selected based on their importance to the model.

📊 Data Visualization
Several visualizations are created to better understand the data distribution:

Distribution of ROM: A histogram and KDE plot for the ROM feature 📉.
RAM Size Distribution: A bar plot for the distribution of RAM sizes across mobile phones 💻.
Boxplot of Brand: A boxplot to visualize the range of brand values 📦.
Ratings Distribution: A histogram to visualize the distribution of ratings ⭐.
🎯 Conclusion
This project successfully trains a Random Forest Regressor to predict the price of mobile phones based on various features. The model's performance is evaluated, and it achieves a high R² score of 0.87, indicating that it can accurately predict the price of mobile phones based on the provided features.
