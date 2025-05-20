import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
pipeline = joblib.load('hotel_model.pkl')

# Daftar fitur yang digunakan untuk prediksi
feature_names = [
    'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes', 'agent', 'company',
    'required_car_parking_spaces', 'total_of_special_requests', 'adr',
    'market_segment', 'distribution_channel', 'customer_type', 'country',
    'total_stay_duration', 'total_expense', 'total_stay', 'total_spending'
]

# Sidebar untuk input user
st.sidebar.header('Input Fitur Booking')
def user_input_features():
    data = {}
    data['lead_time'] = st.sidebar.number_input('Lead Time', 0, 1000, 100)
    data['stays_in_weekend_nights'] = st.sidebar.number_input('Stays in Weekend Nights', 0, 20, 2)
    data['stays_in_week_nights'] = st.sidebar.number_input('Stays in Week Nights', 0, 50, 2)
    data['adults'] = st.sidebar.number_input('Total Adults', 1, 10, 2)
    data['children'] = st.sidebar.number_input('Total Children', 0, 10, 0)
    data['is_repeated_guest'] = st.sidebar.selectbox('Is Repeated Guest?', [0, 1])
    data['previous_cancellations'] = st.sidebar.number_input('Previous Cancellations', 0, 10, 0)
    data['previous_bookings_not_canceled'] = st.sidebar.number_input('Previous Bookings Not Canceled', 0, 10, 0)
    data['booking_changes'] = st.sidebar.number_input('Booking Changes', 0, 10, 0)
    data['agent'] = st.sidebar.number_input('Agent Code', 0, 600, 0)
    data['company'] = st.sidebar.number_input('Company Code', 0, 600, 0)
    data['required_car_parking_spaces'] = st.sidebar.number_input('Car Parking Spaces', 0, 5, 0)
    data['total_of_special_requests'] = st.sidebar.number_input('Total Special Requests', 0, 5, 0)
    data['total_stay_duration'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
    data['adr'] = st.sidebar.number_input('Average Daily Rate', 0.0, 10000.0, 100.0)
    data['total_expense'] = data['adr'] * data['total_stay_duration']

    # Kategorikal
    data['market_segment'] = st.sidebar.selectbox('Market Segment', ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate'])
    data['distribution_channel'] = st.sidebar.selectbox('Distribution Channel', ['TA/TO', 'Direct', 'Corporate'])
    data['customer_type'] = st.sidebar.selectbox('Customer Type', ['Transient', 'Contract', 'Transient-Party', 'Group'])
    data['country'] = st.sidebar.selectbox(
        'Country',
        ['PRT', 'GBR', 'USA', 'ESP', 'FRA', 'DEU', 'ITA', 'IRL', 'NLD', 'BEL', 'BRA', 'Other']  # Ganti sesuai daftar country pada datamu
    )
    data['total_stay'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
    data['total_spending'] = data['adr'] * data['total_stay']

    return pd.DataFrame([data])

input_df = user_input_features()


st.title("Prediksi Pembatalan Booking Hotel")
st.write("Aplikasi ini digunakan untuk memprediksi apakah booking hotel akan dibatalkan atau tidak")
st.write("Cara menggunakan aplikasi ini:")
st.write("1. Masukkan detail booking hotel pada bagian **Input Fitur Booking** di bagian sebelah kiri")
st.write("2. Silahkan isi semua field yang ada di bagian **Input Fitur Booking**")
st.write("3. Klik tombol **Prediksi** untuk melihat hasil prediksi apakah booking akan dibatalkan atau tidak")

st.write("#### Data Input")
st.write(input_df)

if st.button('Prediksi'):
    input_df = input_df[feature_names]
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0, 1]

    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.error(f"Booking **DIBATALKAN** ❌ (Probabilitas: {proba:.2f})")
    else:
        st.success(f"Booking **TIDAK DIBATALKAN** ✅ (Probabilitas: {proba:.2f})")

    st.caption('Jika hasil **Prediksi** mendekati 1 = kemungkinan Booking Hotel tersebut akan dibatalkan)')

st.info("""
Info:
- Model Machine Learning yang digunakan adalah LGBMClassifier
- Model ini dilatih menggunakan dataset Hotel Booking Demand
""")