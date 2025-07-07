# weather_prediction_application
# 🌦 Indian Weather Forecast webApp

This is a Streamlit-powered web application that predicts **today’s and upcoming weather conditions** (Max Temp, Min Temp, Avg Temp, and Precipitation) for major Indian cities using a trained **Random Forest Regressor model**.

The model is trained on historical weather data (1990–2022) and allows users to:

- Select a city from the dropdown
- Predict today's weather and for the next 2 days
- View a 7-day temperature forecast trend with Plotly charts

---

## 🔧 Built With

- [Streamlit](https://streamlit.io/) – For web UI
- [scikit-learn](https://scikit-learn.org/) – For training Random Forest Regressor
- [pandas](https://pandas.pydata.org/) – For data handling
- [plotly](https://plotly.com/) – For interactive data visualizations

---

## 📁 Project Structure
indian-weather-forecast
├── indian_weather.py # Streamlit app interface
├── weather_multi_output_model.pkl # Trained ML model
├── Indian_citites_1990_2022_weather.csv # Historical weather dataset
├── requirements.txt # Dependencies for deployment
└── README.md # Project documentation


## 🚀 How to Run Locally

```bash
# Clone this repo
git clone https://github.com/yourusername/indian-weather-forecast.git
cd indian-weather-forecast

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run indian_weather.py

 Deploying to Streamlit Cloud
Push this folder to GitHub

Go to Streamlit Cloud

Connect your GitHub repo

Deploy your indian_weather.py

Share the app using the public URL generated

