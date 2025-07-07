import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Load model and data
model = pickle.load(open("weather_multi_output_model.pkl", "rb"))
df = pd.read_csv("Indian_citites_1990_2022_weather.csv")
df.dropna(subset=["tmax", "tmin", "tavg", "prcp", "Location"], inplace=True)
df["time"] = pd.to_datetime(df["time"])
df["date_ordinal"] = df["time"].map(pd.Timestamp.toordinal)

st.set_page_config(page_title="Indian Weather Forecast", page_icon="🌤", layout="centered")

# Styling using CSS
st.markdown("""
    <style>
        .main { background-color: #ffffffcc; padding: 2rem; border-radius: 12px; }
        .stButton>button {
            background-color: #007acc;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .result-box {
            background: #f1faff;
            padding: 1rem;
            border-left: 5px solid #007acc;
            margin-top: 1rem;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🌦️ Indian Weather Predictor")
st.write("Forecast real temperatures using trained machine learning model.")

cities = sorted(df["Location"].unique().tolist())
city = st.selectbox("📍 Select City", cities)

city_df = df[df["Location"] == city].sort_values("time", ascending=False)
latest = city_df.iloc[0]
latest_date = latest["time"]
latest_ordinal = latest["date_ordinal"]

# Create input row for selected city and date
dummies = pd.get_dummies(df["Location"])
city_cols = dummies.columns.tolist()

input_row = [0] * (1 + len(city_cols))  # date_ordinal + all city flags
input_row[0] = latest_ordinal  # date ordinal
if city in city_cols:
    input_row[city_cols.index(city) + 1] = 1  # offset by 1 for ordinal

# Prediction
if st.button("🔍 Predict Today's Weather"):
    input_df = pd.DataFrame([input_row], columns=["date_ordinal"] + city_cols)
    prediction = model.predict(input_df)[0]
    tmax, tmin, tavg, prcp = [round(val, 2) for val in prediction]

    st.markdown("### 🔮 Weather Forecast for Today")
    st.markdown(f"<div class='result-box'>🌡️ **Max Temp:** {tmax} °C</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'>❄️ **Min Temp:** {tmin} °C</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'>🌥️ **Avg Temp:** {tavg} °C</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'>🌧️ **Precipitation:** {prcp} mm</div>", unsafe_allow_html=True)

    # Predict next 2 days
    st.markdown("### 📅 Forecast for Next 2 Days")
    for i in range(1, 3):
        future_date = latest_date + timedelta(days=i)
        future_ordinal = future_date.toordinal()
        input_row[0] = future_ordinal
        future_df = pd.DataFrame([input_row], columns=["date_ordinal"] + city_cols)
        future_pred = model.predict(future_df)[0]
        ftmax, ftmin, ftavg, fprcp = [round(val, 2) for val in future_pred]
        st.markdown(f"**{future_date.strftime('%A, %d %b')}:**")
        st.markdown(f"- Max Temp: {ftmax} °C")
        st.markdown(f"- Min Temp: {ftmin} °C")
        st.markdown(f"- Avg Temp: {ftavg} °C")
        st.markdown(f"- Precipitation: {fprcp} mm")

# Plotly 7-day forecast trend
if st.button("📈 Show 7-Day Forecast Trend"):
    temps = []
    dates = []
    for i in range(7):
        day = latest_date + timedelta(days=i)
        input_row[0] = day.toordinal()
        future_df = pd.DataFrame([input_row], columns=["date_ordinal"] + city_cols)
        avg_temp = model.predict(future_df)[0][2]
        dates.append(day.strftime('%a, %d %b'))
        temps.append(round(avg_temp, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=temps,
        mode='lines+markers',
        line=dict(color='royalblue', width=3),
        marker=dict(size=8),
        name='Avg Temp Forecast'
    ))
    fig.update_layout(
        title="📊 7-Day Average Temperature Trend",
        xaxis_title="Day",
        yaxis_title="Temperature (°C)",
        plot_bgcolor='rgba(230, 245, 255, 0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)