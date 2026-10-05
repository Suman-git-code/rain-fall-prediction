import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Rain Predictor", page_icon="🌧️")
st.title("🌧️ Rain Prediction: Will it rain tomorrow?")
st.caption("Live weather from Open-Meteo, predicted by a model trained on Bengaluru data (2019-2024).")

CITIES = {
    "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Mumbai": (19.08, 72.88),
    "Delhi": (28.61, 77.21),
    "Kolkata": (22.57, 88.36),
    "Hyderabad": (17.39, 78.49),
}

DAILY_COLS = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "relative_humidity_2m_mean", "surface_pressure_mean",
    "cloud_cover_mean", "wind_speed_10m_max",
]
LAG_COLS = ["relative_humidity_2m_mean", "surface_pressure_mean",
            "cloud_cover_mean", "precipitation_sum"]


@st.cache_resource
def load_model():
    model = joblib.load("rain_model.pkl")
    features = joblib.load("features.pkl")
    return model, features


def fetch_weather(lat, lon):
    params = {
        "latitude": lat, "longitude": lon, "daily": DAILY_COLS,
        "past_days": 5, "forecast_days": 1, "timezone": "Asia/Kolkata",
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"])
    d["time"] = pd.to_datetime(d["time"])
    d["month"] = d["time"].dt.month
    for col in LAG_COLS:
        d[col + "_lag1"] = d[col].shift(1)
        d[col + "_avg3"] = d[col].rolling(3).mean()
    return d


try:
    model, features = load_model()
except Exception as e:
    st.error("Could not load rain_model.pkl or features.pkl. Keep them in the same folder as app.py.")
    st.exception(e)
    st.stop()

city = st.selectbox("Choose a city", list(CITIES.keys()))
if city != "Bengaluru":
    st.warning("The model was trained on Bengaluru only, so results for other cities are less reliable.")

if st.button("Predict now"):
    try:
        lat, lon = CITIES[city]
        d = fetch_weather(lat, lon)
        today = d.iloc[[-1]]
        X_live = today[features]
        prob = model.predict_proba(X_live)[0][1]

        st.subheader(f"Date: {today['time'].dt.date.values[0]}")
        st.metric("Chance of rain tomorrow", f"{prob:.0%}")
        st.progress(float(prob))

        if prob >= 0.5:
            st.success("Prediction: Rain likely. Carry an umbrella ☔")
        else:
            st.info("Prediction: Rain unlikely 🌤️")

        st.caption("This model leans towards predicting rain, so treat the percentage as a rough signal, not an exact chance.")

        with st.expander("Show today's weather inputs"):
            st.dataframe(today[["time"] + DAILY_COLS].T)
    except Exception as e:
        st.error("Something went wrong while fetching data or predicting.")
        st.exception(e)
