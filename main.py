import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
rf_regressor = joblib.load('random_forest_regressor_model_rfe.pkl')

# Streamlit UI for input
st.title('📱 Mobile Price Prediction')

st.write("Please enter the details of the mobile phone to predict its price:")

# Input fields for each feature
brand = st.selectbox('Brand', ['LG V30+ (Black, 128 )', 'I Kall K11', 'Nokia 105 ss', 'Samsung Galaxy A50 (White, 64 )', 'POCO F1 (Steel Blue, 128 )', 'Apple iPhone 11 Pro (Space Grey, 512 )', 'Samsung Galaxy A70s (Prism Crush Red, 128 )', 'Samsung Galaxy S10 Lite (Prism Blue, 512 )', 'OPPO A9 (Marble Green, 128 )', 'POCO F1 (Graphite Black, 256 )', 'Megus Ultra', 'Jmax M40 Coo of Two', 'Wizphone WP (Black, 16 )', 'Easyfone Star', 'OnePlus 7 Pro (Nebula Blue, 256 )', 'OnePlus 7 Pro (Mirror Grey, 128 )', 'Kechaoda A27', 'Samsung Galaxy S20+ (Cloud Blue, 128 )', 'Samsung Galaxy S10 Lite (Prism Black, 512 )', 'OPPO A5s (Red, 32 )', 'InFocus POWER 2', 'Blacear I7 Trio', 'Karbonn K451 Power', 'Micax X742', 'Snexian Guru 332', 'Blacear i7 Trio', 'Karbonn K334', 'Lava Prime X', 'I Kall K23 New Mobile', 'InFocus Selfie C1', 'LG G8X (Black, 128 )', 'Gee A1 (Grey, 64 )', 'Tecno Camon 15 Pro (Opal White, 128 )', 'Lava A7', 'Apple iPhone XR ((PRODUCT)RED, 128 )', 'Micax X749', 'Itel it2161', 'Karbonn K-Pebble', 'Karbonn KX21', 'Inovu A9', 'LG Q Stylus (Black, 32 )', 'OPPO R17 Pro (Emerald Green, 128 )', 'Micax X516', 'Meizu C9 Pro (Black, 32 )', 'Jivi R21Plus', 'Nokia 3310 DS', 'Lava 34 ', 'Easyfone Amico', 'Apple iPhone 11 Pro Max (Gold, 64 )', 'Samsung Galaxy S8 Plus (Midnight Black, 64 )', 'Black Shark 2 (Shadow Black, 128 )', 'Karbonn K24 Plus Pro', 'InFocus Vibe 1', 'Blacear B5 Grip', 'Apple iPhone 11 Pro (Space Grey, 256 )', 'JIVI JV 12M', 'Alcatel 5V (Spectrum Blue, 32 )', 'Kechaoda A32', 'Redmi Note 5 Pro (Gold, 64 )', 'Salora Vishaal', 'Redmi K20 (Flame Red, 128 )', 'Apple iPhone 8 (PRODUCT)RED (Red, 256 )', 'Apple iPhone 11 (Yellow, 64 )', 'Gee F103 Pro (Grey, 16 )', 'Gee F103 Pro (Gold, 16 )', 'Easyfone Elite', 'JIVI 12M', 'Karbonn K2 Boom Box', 'Nexus 6P Special Edit (Gold, 64 )', 'OPPO F15 (Lightening Black, 128 )', 'Apple iPhone 11 Pro (Gold, 64 )', 'Dublin D614', 'Karbonn KX23', 'LG Q6 (Black, 32 )', 'OPPO K1 (Piano Black, 64 )', 'Redmi Note 7 Pro (Neptune Blue, 128 )', 'Nokia 150/150 DS', 'Kechaoda K112', 'Itel it5260', 'OPPO Reno2 (Ocean Blue, 256 )', 'Jivi N9030', 'Inovu A7', 'Easyfone Udaan', 'Gee S11 Lite (Gold, 32 )', 'Nokia 105 Dual Sim 2017', 'Jivi N300 New', 'Micax X940', 'Samsung Guru 1200', 'Samsung Guru FM Plus', 'Ssky S9007 Rainbow', 'Samsung Guru Music 2', 'Q-Tel Q8', 'Ssky S-40 Prime', 'Redmi K20 Pro (Pearl White, 256 )', 'OPPO A3s (Purple, 32 )', 'OPPO A83 (Chaagne, 16 )', 'OPPO A5s (Black, 32 )', 'Vivo Y66 (Crown Gold, 32 )', 'Micax X772', 'Vivo V7 (Gold, 32 )', 'Vivo X21 (Black, 128 )', 'I Kall K3310', 'Redmi K20 (Pearl White, 64 )', 'GAMMA K 28', 'GAMMA K2 no', 'Samsung Galaxy S20+ (Cosmic Black, 128 )', 'Karbonn KX1', 'Redmi K20 (Carbon Black, 128 )', 'Mi A3 (More Than White, 128 )', 'Apple iPhone XR (Yellow, 128 )', 'Kechaoda K9', 'Nokia 216/216 DS', 'JIVI N3720 Power', 'BlackZone Neo-B', 'Micax X707', 'Lava One', 'Muphone MU M360', 'Micax Flash X910', 'I Kall K27 New', 'OPPO A5s (Blue, 32 )', 'Micax X412', 'I Kall K 15', 'Vivo Y81 (Gold, 32 )', 'Karbonn K49', 'MI3 (Metallic Grey, 16 )', 'Karbonn K Pebble', 'Grabo G100', 'Samsung Metro 313 Dual Sim', 'Apple iPhone 11 Pro Max (Silver, 64 )', 'OPPO R17 (Aient Blue, 128 )', 'OPPO R17 (Neon Purple, 128 )', 'Salora Kiano', 'Samsung Guru GT', 'Redmi Note 8 (Cosmic Purple, 128 )', 'MTR MT 310', 'I Kall K18 New Coo of Two Mobiles', 'Karbonn Titanium S4 (Black, 4 )', 'MTR Mt312', 'Redmi Note 6 Pro (Black, 64 )', 'Samsung Galaxy A21s (White, 64 )', 'Nokia 7.2 (Charcoal, 64 )', 'MTR Sima', 'Lava Z61 (Gold, 16 )', 'MTR Ba', 'I Kall K73', 'Itel It 5605n', 'Karbonn K338N', 'MTR Shakti', 'I Kall K3312', 'Honor 8X (Black, 128 )', 'Karbonn K140 Pop', 'GAMMA K1 no', 'Vivo S1 (Diamond Black, 64 )', 'Karbonn K19 Rock', 'I Kall K22 New', 'I Kall K3310 Coo Of Two Mobile', 'Micax X773', 'Tork T13 Ba', 'Honor 9N (Midnight Black, 64 )', 'Lava H1 Hero 600', 'Samsung Galaxy A80 (Phantom Black, 128 )', 'Inovu A7i', 'Alcatel 5V (Spectrum Black, 32 )', 'Mi A2 (Blue/Lake Blue, 64 )', 'Mi A3 (Not just Blue, 128 )', 'Samsung Galaxy S8 Plus (Maple Gold, 64 )', 'InFocus VIBE 2', 'Realme C2 (Diamond Blue, 16 )', 'Honor 8X (Blue, 64 )', 'I Kall K 2180 Coo of Two Mobiles', 'Lava A1 ', 'Itel It 2163', 'Callbar A35S', 'POCO F1 (Rosso Red, 64 )', 'I Kall K55 Coo Of Two Mobile', 'Yuho O1 (Onyx Black, 16 )', 'InFocus Hero Selfie C1', 'Vivo V17Pro (Glacier Ice White, 128 )', 'I Kall K25 New', 'Itel It2171', 'I Kall lk22 new', 'Itel It 5022', 'Apple iPhone XS (Silver, 256 )', 'Honor 9N (Midnight Black, 128 )', 'InFocus F229', 'GreenBerry 312', 'Vivo Y81 (Black, 32 )', 'Mymax A12', 'Intex Eco Selfie 2', 'Samsung Galaxy S10e (Prism Black, 128 )', 'I Kall K71 Coo Of Two Mobile', 'Itel It 2190', 'Trio T6*', 'Itel IT5025', 'Redmi 8A Dual (Midnight Grey, 64 )', 'F-Fook F22', 'Vivo Y17 (Mystic Purple, 128 )', 'Mi A3 (Kind of Grey, 128 )', 'Ecotel E17', 'Honor 9N (Sapphire Blue, 128 )', 'Heemax H10' 
])  
ratings = st.slider('Ratings', 2.8, 4.8, 4.0, 0.1)
mobile_size = st.slider('Mobile Size (Inches)', 2.0, 6.5, 5.0, 0.1)

# Prepare the input data for prediction
input_data = {
    'brand_me': [brand],
    'ratings': [ratings],
    'mobile_size': [mobile_size],
    
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Label encoding for 'brand_me' column as the model expects numeric encoding
label_encoder = LabelEncoder()

brands = ['LG V30+ (Black, 128 )', 'I Kall K11', 'Nokia 105 ss', 'Samsung Galaxy A50 (White, 64 )', 'POCO F1 (Steel Blue, 128 )', 'Apple iPhone 11 Pro (Space Grey, 512 )', 'Samsung Galaxy A70s (Prism Crush Red, 128 )', 'Samsung Galaxy S10 Lite (Prism Blue, 512 )', 'OPPO A9 (Marble Green, 128 )', 'POCO F1 (Graphite Black, 256 )', 'Megus Ultra', 'Jmax M40 Coo of Two', 'Wizphone WP (Black, 16 )', 'Easyfone Star', 'OnePlus 7 Pro (Nebula Blue, 256 )', 'OnePlus 7 Pro (Mirror Grey, 128 )', 'Kechaoda A27', 'Samsung Galaxy S20+ (Cloud Blue, 128 )', 'Samsung Galaxy S10 Lite (Prism Black, 512 )', 'OPPO A5s (Red, 32 )', 'InFocus POWER 2', 'Blacear I7 Trio', 'Karbonn K451 Power', 'Micax X742', 'Snexian Guru 332', 'Blacear i7 Trio', 'Karbonn K334', 'Lava Prime X', 'I Kall K23 New Mobile', 'InFocus Selfie C1', 'LG G8X (Black, 128 )', 'Gee A1 (Grey, 64 )', 'Tecno Camon 15 Pro (Opal White, 128 )', 'Lava A7', 'Apple iPhone XR ((PRODUCT)RED, 128 )', 'Micax X749', 'Itel it2161', 'Karbonn K-Pebble', 'Karbonn KX21', 'Inovu A9', 'LG Q Stylus (Black, 32 )', 'OPPO R17 Pro (Emerald Green, 128 )', 'Micax X516', 'Meizu C9 Pro (Black, 32 )', 'Jivi R21Plus', 'Nokia 3310 DS', 'Lava 34 ', 'Easyfone Amico', 'Apple iPhone 11 Pro Max (Gold, 64 )', 'Samsung Galaxy S8 Plus (Midnight Black, 64 )', 'Black Shark 2 (Shadow Black, 128 )', 'Karbonn K24 Plus Pro', 'InFocus Vibe 1', 'Blacear B5 Grip', 'Apple iPhone 11 Pro (Space Grey, 256 )', 'JIVI JV 12M', 'Alcatel 5V (Spectrum Blue, 32 )', 'Kechaoda A32', 'Redmi Note 5 Pro (Gold, 64 )', 'Salora Vishaal', 'Redmi K20 (Flame Red, 128 )', 'Apple iPhone 8 (PRODUCT)RED (Red, 256 )', 'Apple iPhone 11 (Yellow, 64 )', 'Gee F103 Pro (Grey, 16 )', 'Gee F103 Pro (Gold, 16 )', 'Easyfone Elite', 'JIVI 12M', 'Karbonn K2 Boom Box', 'Nexus 6P Special Edit (Gold, 64 )', 'OPPO F15 (Lightening Black, 128 )', 'Apple iPhone 11 Pro (Gold, 64 )', 'Dublin D614', 'Karbonn KX23', 'LG Q6 (Black, 32 )', 'OPPO K1 (Piano Black, 64 )', 'Redmi Note 7 Pro (Neptune Blue, 128 )', 'Nokia 150/150 DS', 'Kechaoda K112', 'Itel it5260', 'OPPO Reno2 (Ocean Blue, 256 )', 'Jivi N9030', 'Inovu A7', 'Easyfone Udaan', 'Gee S11 Lite (Gold, 32 )', 'Nokia 105 Dual Sim 2017', 'Jivi N300 New', 'Micax X940', 'Samsung Guru 1200', 'Samsung Guru FM Plus', 'Ssky S9007 Rainbow', 'Samsung Guru Music 2', 'Q-Tel Q8', 'Ssky S-40 Prime', 'Redmi K20 Pro (Pearl White, 256 )', 'OPPO A3s (Purple, 32 )', 'OPPO A83 (Chaagne, 16 )', 'OPPO A5s (Black, 32 )', 'Vivo Y66 (Crown Gold, 32 )', 'Micax X772', 'Vivo V7 (Gold, 32 )', 'Vivo X21 (Black, 128 )', 'I Kall K3310', 'Redmi K20 (Pearl White, 64 )', 'GAMMA K 28', 'GAMMA K2 no', 'Samsung Galaxy S20+ (Cosmic Black, 128 )', 'Karbonn KX1', 'Redmi K20 (Carbon Black, 128 )', 'Mi A3 (More Than White, 128 )', 'Apple iPhone XR (Yellow, 128 )', 'Kechaoda K9', 'Nokia 216/216 DS', 'JIVI N3720 Power', 'BlackZone Neo-B', 'Micax X707', 'Lava One', 'Muphone MU M360', 'Micax Flash X910', 'I Kall K27 New', 'OPPO A5s (Blue, 32 )', 'Micax X412', 'I Kall K 15', 'Vivo Y81 (Gold, 32 )', 'Karbonn K49', 'MI3 (Metallic Grey, 16 )', 'Karbonn K Pebble', 'Grabo G100', 'Samsung Metro 313 Dual Sim', 'Apple iPhone 11 Pro Max (Silver, 64 )', 'OPPO R17 (Aient Blue, 128 )', 'OPPO R17 (Neon Purple, 128 )', 'Salora Kiano', 'Samsung Guru GT', 'Redmi Note 8 (Cosmic Purple, 128 )', 'MTR MT 310', 'I Kall K18 New Coo of Two Mobiles', 'Karbonn Titanium S4 (Black, 4 )', 'MTR Mt312', 'Redmi Note 6 Pro (Black, 64 )', 'Samsung Galaxy A21s (White, 64 )', 'Nokia 7.2 (Charcoal, 64 )', 'MTR Sima', 'Lava Z61 (Gold, 16 )', 'MTR Ba', 'I Kall K73', 'Itel It 5605n', 'Karbonn K338N', 'MTR Shakti', 'I Kall K3312', 'Honor 8X (Black, 128 )', 'Karbonn K140 Pop', 'GAMMA K1 no', 'Vivo S1 (Diamond Black, 64 )', 'Karbonn K19 Rock', 'I Kall K22 New', 'I Kall K3310 Coo Of Two Mobile', 'Micax X773', 'Tork T13 Ba', 'Honor 9N (Midnight Black, 64 )', 'Lava H1 Hero 600', 'Samsung Galaxy A80 (Phantom Black, 128 )', 'Inovu A7i', 'Alcatel 5V (Spectrum Black, 32 )', 'Mi A2 (Blue/Lake Blue, 64 )', 'Mi A3 (Not just Blue, 128 )', 'Samsung Galaxy S8 Plus (Maple Gold, 64 )', 'InFocus VIBE 2', 'Realme C2 (Diamond Blue, 16 )', 'Honor 8X (Blue, 64 )', 'I Kall K 2180 Coo of Two Mobiles', 'Lava A1 ', 'Itel It 2163', 'Callbar A35S', 'POCO F1 (Rosso Red, 64 )', 'I Kall K55 Coo Of Two Mobile', 'Yuho O1 (Onyx Black, 16 )', 'InFocus Hero Selfie C1', 'Vivo V17Pro (Glacier Ice White, 128 )', 'I Kall K25 New', 'Itel It2171', 'I Kall lk22 new', 'Itel It 5022', 'Apple iPhone XS (Silver, 256 )', 'Honor 9N (Midnight Black, 128 )', 'InFocus F229', 'GreenBerry 312', 'Vivo Y81 (Black, 32 )', 'Mymax A12', 'Intex Eco Selfie 2', 'Samsung Galaxy S10e (Prism Black, 128 )', 'I Kall K71 Coo Of Two Mobile', 'Itel It 2190', 'Trio T6*', 'Itel IT5025', 'Redmi 8A Dual (Midnight Grey, 64 )', 'F-Fook F22', 'Vivo Y17 (Mystic Purple, 128 )', 'Mi A3 (Kind of Grey, 128 )', 'Ecotel E17', 'Honor 9N (Sapphire Blue, 128 )', 'Heemax H10' 
]  
label_encoder.fit(brands)  # Fit the label encoder
input_df['brand_me'] = label_encoder.transform(input_df['brand_me'])

# Model prediction
if st.button('Predict Price 💰'):
    predicted_price = rf_regressor.predict(input_df)
    st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")

# Optional: Add more visualizations or analysis
st.subheader('📊 Feature Importance Visualization')
# Visualize feature importances from the trained model
importances = rf_regressor.feature_importances_
features = input_df.columns

# Create a bar chart of feature importances
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=features, ax=ax, palette='viridis')
ax.set_title('Feature Importances')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
st.pyplot(fig)
