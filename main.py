# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and display data
st.title("Mobile Price Prediction App")
st.write("This application allows you to explore data, visualize distributions, and predict mobile prices based on various features.")

# Load dataset
data = pd.read_csv('Mobile Price Prediction.csv')
st.subheader("Dataset")
st.write("First few rows of the data:")
st.write(data.head())
st.write("Shape of the data:", data.shape)

# Drop unnecessary columns and handle missing values
data = data.drop(columns=['Unnamed: 0'])
data.fillna(data.mean(), inplace=True)

# Standardize column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)

# Encode categorical variable
label_encoder = LabelEncoder()
data['brand_me'] = label_encoder.fit_transform(data['brand_me'])

# Data Visualizations
st.subheader("Data Visualizations")

# ROM Distribution Plot
st.write("### Distribution of ROM")
fig_rom, ax_rom = plt.subplots()
sns.histplot(data['rom'], kde=True, bins=20, ax=ax_rom)
ax_rom.set_xlabel('ROM')
ax_rom.set_ylabel('Frequency')
st.pyplot(fig_rom)

# RAM Distribution Plot
st.write("### RAM Size Distribution")
ram_counts = data['ram'].value_counts()
fig_ram, ax_ram = plt.subplots()
ax_ram.bar(ram_counts.index, ram_counts.values, color='green')
ax_ram.set_xlabel('RAM Size')
ax_ram.set_ylabel('Count')
st.pyplot(fig_ram)

# Ratings Distribution Plot
st.write("### Distribution of Ratings")
fig_ratings, ax_ratings = plt.subplots()
sns.histplot(data["ratings"], kde=True, color='skyblue', ax=ax_ratings)
ax_ratings.set_xlabel("Ratings")
ax_ratings.set_ylabel("Frequency")
st.pyplot(fig_ratings)

# Feature and Target Selection for Model Training
X = data[['brand_me', 'ratings', 'ram', 'rom', 'mobile_size', 'primary_cam', 'selfi_cam', 'battery_power']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and apply RFE for feature selection
st.subheader("Feature Selection using RFE")
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_regressor, n_features_to_select=3)
rfe.fit(X_train, y_train)

# Display RFE ranking of features
feature_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
st.write("Features ranked by RFE:")
st.write(feature_ranking.sort_values('Ranking'))

# Select the top features and train the model
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Model training
rf_regressor.fit(X_train_rfe, y_train)
rf_predictions = rf_regressor.predict(X_test_rfe)

# Model evaluation metrics
st.subheader("Model Evaluation")
mae = mean_absolute_error(y_test, rf_predictions)
mse = mean_squared_error(y_test, rf_predictions)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, rf_predictions)

st.write("Mean Absolute Error (MAE):", mae)
st.write("Mean Squared Error (MSE):", mse)
st.write("Root Mean Squared Error (RMSE):", rmse)
st.write("R² Score:", r2)

# Save and Load Model
st.subheader("Model Saving and Loading")
joblib.dump(rf_regressor, 'random_forest_regressor_model_rfe.pkl')
st.write("Model saved as 'random_forest_regressor_model_rfe.pkl'")

if st.button("Load and Use Saved Model"):
    loaded_rf_regressor = joblib.load('random_forest_regressor_model_rfe.pkl')
    st.write("Model loaded successfully!")

    # Optional: Predict using user inputs
    st.write("Predict mobile price based on selected features:")
    brand_me = st.number_input("Brand ID (encoded)", min_value=0, max_value=len(label_encoder.classes_)-1)
    ratings = st.slider("Ratings", min_value=0.0, max_value=5.0, step=0.1)
    ram = st.selectbox("RAM (GB)", options=sorted(data['ram'].unique()))
    rom = st.selectbox("ROM (GB)", options=sorted(data['rom'].unique()))
    mobile_size = st.slider("Mobile Size (inches)", min_value=0.0, max_value=10.0, step=0.1)
    primary_cam = st.selectbox("Primary Camera (MP)", options=sorted(data['primary_cam'].unique()))
    selfi_cam = st.selectbox("Selfie Camera (MP)", options=sorted(data['selfi_cam'].unique()))
    battery_power = st.selectbox("Battery Power (mAh)", options=sorted(data['battery_power'].unique()))

    # Make prediction
    if st.button("Predict Price"):
        input_data = [[brand_me, ratings,mobile_size]]
        selected_input_data = rfe.transform(input_data) 
        predicted_price = loaded_rf_regressor.predict(selected_input_data)
        st.write("Predicted Price:", predicted_price[0])

st.write("### End of Application")