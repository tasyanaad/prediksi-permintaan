import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.title("Prediksi Permintaan Barang Retail")

# Upload data CSV
uploaded_file = st.file_uploader("Upload file CSV bersih", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Tanggal'] = pd.to_datetime(data['Tanggal'])
    data['Hari_ke'] = (data['Tanggal'] - data['Tanggal'].min()).dt.days

    # Split fitur dan target
    X = data[['Hari_ke']]
    y = data['Total_Order_Harian']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"📉 Mean Absolute Error: ±{mae:.2f} unit/hari")

    # Prediksi 7 hari ke depan
    hari_terakhir = data['Hari_ke'].max()
    hari_baru = pd.DataFrame({'Hari_ke': range(hari_terakhir + 1, hari_terakhir + 8)})
    prediksi = model.predict(hari_baru)

    hasil = pd.DataFrame({
        'Hari_ke': hari_baru['Hari_ke'],
        'Prediksi_Order': prediksi.astype(int)
    })

    st.subheader("📆 Prediksi Permintaan 7 Hari ke Depan")
    st.dataframe(hasil)

    st.line_chart(hasil.set_index('Hari_ke'))
