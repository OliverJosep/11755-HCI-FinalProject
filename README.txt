### House Price Predictor

A web application built with streamlit to estimate the price of a house in the Balearic Islands based on its characteristics.

This application has dropdown selectors for city, zone, and house type; and custom inputs for surface area, number of rooms, bathrooms, and amenities (terrace, air conditioning, pool, etc.).
Using this information and a pre-trained xgboost model, the application shows a price prediction and a SHAP-based explanation showing the contribution of each feature to the final prediction.

You can install the required libraries using the `requirements.txt` file (pip install -r requirements.txt).

For running the app, you have to write "streamlit run app.py" in the terminal.
