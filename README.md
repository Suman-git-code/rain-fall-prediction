# Rainfall Prediction — Will It Rain Tomorrow?

A machine learning project that predicts whether it will **rain tomorrow** using real-world weather data from **Open-Meteo**.

The project is trained on historical weather data for **Bengaluru, India (2019–2024)** and provides a simple **Streamlit web application** for making live predictions using current weather data.

>  **Note:** The model was trained specifically on Bengaluru weather data. Predictions for other cities are provided in the application but may be less reliable.

---

##  Project Overview

The goal of this project is to build a binary classification model that predicts:

* **1 → Rain tomorrow**
* **0 → No rain tomorrow**

Rain is defined as **1 mm or more of precipitation on the following day**.

The project follows a complete machine learning workflow:

1. Download historical weather data
2. Clean and preprocess the data
3. Create the prediction target
4. Engineer additional features
5. Split data chronologically into training, validation, and test sets
6. Train multiple machine learning models
7. Compare model performance
8. Evaluate the selected model on the test set
9. Save the trained model
10. Use live weather data to make predictions
11. Deploy the prediction system using Streamlit

---

##  Features

*  Real weather data from Open-Meteo
*  Historical data from 2019–2024
*  Bengaluru-focused rainfall prediction
*  Multiple machine learning models
*  Model comparison using classification metrics
*  Time-series-aware train/validation/test split
*  Feature engineering using lag and rolling-average features
*  Trained model saved using Joblib
*  Live weather data for prediction
*  Interactive Streamlit dashboard
*  City selection for Bengaluru, Chennai, Mumbai, Delhi, Kolkata, and Hyderabad
*  Rain probability displayed directly in the application

---

##  Dataset

The historical weather data is obtained from the **Open-Meteo Archive API**.

### Location

**Bengaluru, India**

* Latitude: `12.97`
* Longitude: `77.59`

### Period

**January 1, 2019 → December 31, 2024**

The original dataset contains **2,192 daily observations** before feature engineering.

### Weather Variables

The project uses the following daily weather variables:

| Feature                     | Description               |
| --------------------------- | ------------------------- |
| `temperature_2m_max`        | Maximum temperature       |
| `temperature_2m_min`        | Minimum temperature       |
| `precipitation_sum`         | Total daily precipitation |
| `relative_humidity_2m_mean` | Mean relative humidity    |
| `surface_pressure_mean`     | Mean surface pressure     |
| `cloud_cover_mean`          | Mean cloud coverage       |
| `wind_speed_10m_max`        | Maximum wind speed        |

---

##  Target Variable

The target variable is:

```text
RainTomorrow
```

It is created using the next day's precipitation:

```python
df["RainTomorrow"] = (
    df["precipitation_sum"].shift(-1) >= 1
).astype(int)
```

Therefore:

```text
1 → Rain tomorrow
0 → No rain tomorrow
```

The final day is removed because there is no following day available to calculate its target.

---

##  Feature Engineering

In addition to the original weather variables, the project creates several additional features.

### Month

The month is extracted from the date:

```python
df["month"] = df["time"].dt.month
```

This helps the model capture seasonal patterns.

### Lag Features

For selected weather variables, the previous day's value is included:

```text
relative_humidity_2m_mean_lag1
surface_pressure_mean_lag1
cloud_cover_mean_lag1
precipitation_sum_lag1
```

### 3-Day Rolling Average

Three-day averages are also calculated:

```text
relative_humidity_2m_mean_avg3
surface_pressure_mean_avg3
cloud_cover_mean_avg3
precipitation_sum_avg3
```

The final feature set contains **16 features**.

---

##  Machine Learning Models

The notebook compares several classification algorithms:

### 1. Logistic Regression

A scaled Logistic Regression model is used with balanced class weights.

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

### 2. Decision Tree

A Decision Tree classifier is trained with a maximum depth of 6.

### 3. Random Forest

A Random Forest classifier is trained with:

```text
n_estimators = 200
class_weight = balanced
```

### 4. XGBoost

An XGBoost classifier is also evaluated with class imbalance handling.

---

##  Model Evaluation

The models are compared on a validation dataset using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The data is split chronologically rather than randomly to better preserve the time-based nature of weather prediction.

### Dataset Split

Approximately:

```text
70% → Training
15% → Validation
15% → Testing
```

The notebook then evaluates the selected model on the final test set.

> The notebook currently sets `Logistic Regression` as `best_model` for the final test and model export. The README intentionally does not claim a specific accuracy or F1 score because the project files do not establish a single fixed final metric.

---

##  Project Structure

A typical project directory contains:

```text
Rainfall_project/
│
├── app.py
├── rainfall.ipynb
├── rain_model.pkl
├── features.pkl
├── run_app.bat
└── README.md
```

### File Description

| File             | Purpose                                                                            |
| ---------------- | ---------------------------------------------------------------------------------- |
| `rainfall.ipynb` | Data collection, preprocessing, feature engineering, model training and evaluation |
| `app.py`         | Streamlit application for live predictions                                         |
| `rain_model.pkl` | Saved trained machine learning model                                               |
| `features.pkl`   | Saved list of model features                                                       |
| `run_app.bat`    | Windows batch file for launching the application                                   |
| `README.md`      | Project documentation                                                              |

---

##  Streamlit Application

The Streamlit application provides an easy-to-use interface for live predictions.

The application:

1. Allows the user to select a city
2. Retrieves recent weather information from Open-Meteo
3. Creates the required features
4. Loads the saved machine learning model
5. Calculates the probability of rain
6. Displays the prediction

The app currently supports:

```text
Bengaluru
Chennai
Mumbai
Delhi
Kolkata
Hyderabad
```

The application specifically warns users that the model was trained only on Bengaluru data for other-city predictions.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your GitHub details.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pandas requests streamlit joblib scikit-learn xgboost matplotlib
```

---

##  Running the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
rainfall.ipynb
```

The notebook downloads the historical Open-Meteo data, preprocesses it, creates features, trains the models, evaluates them, and saves the selected model.

---

##  Running the Streamlit App

Make sure these files are in the same directory:

```text
app.py
rain_model.pkl
features.pkl
```

Then run:

```bash
streamlit run app.py
```

The application will open in your browser.

The app loads the saved model and feature list using Joblib.

---

##  How Prediction Works

The application retrieves recent weather data from the Open-Meteo forecast API.

It uses:

* Today's weather variables
* Previous-day weather values
* Three-day rolling averages
* Month/season information

The model then calculates:

```text
Probability of rain tomorrow
```

The application uses a probability threshold of **50%**:

```text
Probability >= 50%
        ↓
Rain likely 

Probability < 50%
        ↓
Rain unlikely 
```

The prediction probability is displayed as a percentage in the Streamlit interface.

---

##  Machine Learning Workflow

```text
                 Open-Meteo
                     │
                     ▼
          Historical Weather Data
              Bengaluru
              2019–2024
                     │
                     ▼
              Data Cleaning
                     │
                     ▼
            Feature Engineering
                     │
                     ├── Weather Features
                     ├── Month
                     ├── Lag Features
                     └── 3-Day Averages
                     │
                     ▼
              Train / Validation
                 / Test Split
                     │
                     ▼
        ┌────────────┬─────────────┐
        │            │             │
        ▼            ▼             ▼
   Logistic      Decision      Random Forest
  Regression       Tree
        │            │             │
        └────────────┼─────────────┘
                     │
                     ▼
                  XGBoost
                     │
                     ▼
             Model Comparison
                     │
                     ▼
              Selected Model
                     │
                     ▼
             rain_model.pkl
                     │
                     ▼
             Streamlit App
                     │
                     ▼
             Live Weather Data
                     │
                     ▼
          Rain Tomorrow Prediction ☔
```

---

##  Limitations

This project has several important limitations:

* The model is trained using **Bengaluru weather data only**.
* Predictions for other cities may not generalize well.
* Weather prediction is inherently uncertain.
* The displayed probability should be treated as a **rough signal**, not an exact probability.
* The model may have a tendency toward predicting rain, as noted directly in the application.
* The application depends on the availability of the Open-Meteo API.
* The saved `.pkl` model and feature files must be present for the Streamlit application to work.

---

## Future Improvements

Possible improvements for future versions include:

*  Train the model using weather data from multiple cities
*  Use a larger historical dataset
*  Add more weather variables
*  Perform hyperparameter tuning
*  Improve class imbalance handling
*  Add feature importance visualizations
*  Experiment with advanced time-series models
*  Add historical prediction charts
*  Add an interactive weather map
*  Deploy the application online
*  Automatically retrain the model with new data
*  Calibrate predicted probabilities

---

##  Technologies Used

* **Python**
* **Pandas** — data manipulation
* **Scikit-learn** — machine learning
* **XGBoost** — gradient boosting
* **Matplotlib** — data visualization
* **Requests** — API requests
* **Joblib** — model serialization
* **Streamlit** — web application
* **Jupyter Notebook** — experimentation and model development
* **Open-Meteo API** — weather data

---

##  Data Source

Weather data is obtained from **Open-Meteo**, including both historical archive data and live forecast data.

The Streamlit application requests daily weather variables from the Open-Meteo forecast API and uses the returned data to construct the model input.

---

##  Author

**Suman**

If you found this project useful, consider giving the repository a ⭐ on GitHub!

---

##  License

This project is intended for educational and research purposes.
