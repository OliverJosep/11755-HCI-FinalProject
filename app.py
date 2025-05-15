import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

ALLOWED_KEYS = [
    'air_conditioner', 'balcony', 'bath_num', 'chimney', 'garage', 'garden',
    'm2_real', 'reduced_mobility', 'room_num', 'storage_room',
    'swimming_pool', 'terrace',
    # House types
    'house_type_ Casa o chalet', 'house_type_ Casa o chalet independiente', 'house_type_ Casa rural',
    'house_type_ Casa terrera', 'house_type_ Caserón', 'house_type_ Castillo',
    'house_type_ Chalet adosado', 'house_type_ Chalet pareado', 'house_type_ Cortijo',
    'house_type_ Finca rústica', 'house_type_ Masía', 'house_type_ Palacio',
    'house_type_ Torre', 'house_type_Casa rural', 'house_type_Dúplex',
    'house_type_Estudio', 'house_type_Piso', 'house_type_Piso con ascensor',
    'house_type_Ático',
    # Cities
    'loc_city_Alaró', 'loc_city_Ariany', 'loc_city_Bañalbufar', 'loc_city_Binissalem',
    'loc_city_Buger', 'loc_city_Bunyola', 'loc_city_Cala Bona', 'loc_city_Cala Millor',
    'loc_city_Cala Ratjada', 'loc_city_Calvià', 'loc_city_Campanet', 'loc_city_Campos',
    'loc_city_Canyamel', 'loc_city_Capdepera', 'loc_city_Colonia de Sant Jordi',
    'loc_city_Consell', 'loc_city_Costa de los Pinos', 'loc_city_Costitx', 'loc_city_Deya',
    'loc_city_Eivissa', 'loc_city_Es Mercadal', 'loc_city_Es Migjorn Gran', 'loc_city_Escorca',
    'loc_city_Esporles', 'loc_city_Estellenchs', 'loc_city_Felanitx', 'loc_city_Ferreries',
    'loc_city_Formentera', 'loc_city_Fornalutx', 'loc_city_Inca', 'loc_city_Lloret de Vista Alegre',
    'loc_city_Lloseta', 'loc_city_Llubi', 'loc_city_Llucmajor', 'loc_city_Manacor',
    'loc_city_Mancor de la Vall', 'loc_city_Maria de la Salud', 'loc_city_Marratxi',
    'loc_city_Maó/Mahon', 'loc_city_Montuiri', 'loc_city_Muro', 'loc_city_Palma de Mallorca',
    'loc_city_Palmanyola', 'loc_city_Petra', 'loc_city_Pobla (Sa)', 'loc_city_Pollença',
    'loc_city_Porreres', 'loc_city_Portinax', 'loc_city_Puerto de Pollença',
    'loc_city_Puigpunyent', 'loc_city_Sa Coma', 'loc_city_Sa Ràpita', 'loc_city_San Rafael',
    'loc_city_San Vicente', 'loc_city_Sant Antoni de Portmany', 'loc_city_Sant Joan',
    'loc_city_Sant Joan de Labritja', 'loc_city_Sant Josep de Sa Talaia',
    'loc_city_Sant Llorenç Des Cardassar', 'loc_city_Sant Lluis', 'loc_city_Santa Eugenia',
    'loc_city_Santa Eulalia del Río', 'loc_city_Santa Margalida', 'loc_city_Santa Maria del Cami',
    'loc_city_Selva', 'loc_city_Sencelles', 'loc_city_Ses Salines (Mallorca)',
    'loc_city_Sineu', 'loc_city_Soller', 'loc_city_Son Carrio', 'loc_city_Son Servera',
    'loc_city_Valldemossa', 'loc_city_Villafranca de Bonany',
    # Zones
    'loc_zone_Ibiza, Balears (Illes)', 'loc_zone_Mallorca, Balears (Illes)', 'loc_zone_Menorca, Balears (Illes)'
]

cities = sorted([key.replace('loc_city_', '') for key in ALLOWED_KEYS if key.startswith('loc_city_')])
zones = sorted([key.replace('loc_zone_', '') for key in ALLOWED_KEYS if key.startswith('loc_zone_')])
types = sorted([key.replace('house_type_', '') for key in ALLOWED_KEYS if key.startswith('house_type_')])



st.set_page_config(page_title="House Price Predictor", layout="centered")

# Load model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.markdown("""
    <h1 style='text-align: center;'>🏠 House Price Prediction</h1>
    <p style='text-align: center; color: gray;'>Estimate the price of a house based on its characteristics and location</p>
""", unsafe_allow_html=True)

st.markdown("---")

# 🔧 Inputs
st.subheader("📍 Location")
# loc_city = st.selectbox('City:', ['Palma', 'Inca', 'Manacor', 'Llucmajor'])
# loc_zone = st.selectbox('Zone:', ['Ibiza, Balears (Illes)'])  # Exemple
# house_type = st.selectbox('Type of House:', ['Flat', 'House', 'Penthouse', 'Duplex'])
loc_city = st.selectbox('City:', cities)
loc_zone = st.selectbox('Zone:', zones)
house_type = st.selectbox('Type of House:', types)

st.markdown("---")
st.subheader("📏 Basic Information")
m2_real = st.number_input('Real Surface Area (m²):', min_value=0)
room_num = st.number_input('Number of Rooms:', min_value=0)
bath_num = st.number_input('Number of Bathrooms:', min_value=0)

st.markdown("---")
st.subheader("🧰 Amenities and Features")
col1, col2, col3 = st.columns(3)
with col1:
    air_conditioner = st.radio('❄️ Air Conditioner:', ['Yes', 'No'])
    balcony = st.radio('🌇 Balcony:', ['Yes', 'No'])
    chimney = st.radio('🔥 Chimney:', ['Yes', 'No'])

with col2:
    garage = st.radio('🚗 Garage:', ['Yes', 'No'])
    garden = st.radio('🌿 Garden:', ['Yes', 'No'])
    storage_room = st.radio('📦 Storage Room:', ['Yes', 'No'])

with col3:
    reduced_mobility = st.radio('♿ Reduced Mobility:', ['Yes', 'No'])
    swimming_pool = st.radio('🏊 Swimming Pool:', ['Yes', 'No'])
    terrace = st.radio('🌤️ Terrace:', ['Yes', 'No'])

st.markdown("---")


# Function to convert categorical variables to binary
def to_binary(x): return 1 if x == 'Yes' else 0

# Make prediction
if st.button('💡 Predict Price', use_container_width=True):

    input_dict = {
        'air_conditioner': to_binary(air_conditioner),
        'balcony': to_binary(balcony),
        'bath_num': bath_num,
        'chimney': to_binary(chimney),
        'garage': to_binary(garage),
        'garden': to_binary(garden),
        'm2_real': m2_real,
        'reduced_mobility': to_binary(reduced_mobility),
        'room_num': room_num,
        'storage_room': to_binary(storage_room),
        'swimming_pool': to_binary(swimming_pool),
        'terrace': to_binary(terrace),
        f'house_type_{house_type}': 1,
        f'loc_city_{loc_city}': 1,
        f'loc_zone_{loc_zone}': 1,
    }

    # Set default values for missing keys
    for key in ALLOWED_KEYS:
        if key not in input_dict:
            input_dict[key] = 0

    # Create DataFrame for input
    input_df = pd.DataFrame([input_dict])[ALLOWED_KEYS]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏷️ Estimated Price: {prediction:.2f} €")

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(pd.DataFrame(input_scaled, columns=columns))

    st.subheader("🔍 Feature Impact")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())

# Footer
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: var(--footer-bg-color, #f0f2f6);
    padding: 10px;
    text-align: center;
    font-size: 0.9em;
    color: var(--text-color, #888);
    border-top: 1px solid #e6e6e6;
}
@media (prefers-color-scheme: dark) {
    .footer {
        background-color: #0e1117;
        color: #aaa;
        border-top: 1px solid #444;
    }
}
</style>
<div class="footer">
    <p>&copy; 2025 - Made with ❤️ using Streamlit</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
