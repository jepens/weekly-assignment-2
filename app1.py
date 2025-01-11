import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

# Set page configuration
st.set_page_config(
    page_title="Prediksi Risiko Diabetes - Pima Indians",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS untuk meningkatkan tampilan
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .title-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .info-box {
        background-color: #e1ecf4;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .footer {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header section dengan informasi dataset
st.markdown("""
    <div class="title-container">
        <h1>🏥 Sistem Prediksi Risiko Diabetes</h1>
        <p>Dataset Pima Indians - National Institute of Diabetes and Digestive and Kidney Diseases</p>
    </div>
""", unsafe_allow_html=True)

# Informasi tentang dataset
st.markdown("""
    <div class="info-box">
        <h3>📊 Tentang Dataset</h3>
        <p>Sistem cerdas ini dikembangkan menggunakan dataset khusus yang berasal dari populasi wanita Pima Indians 
        di Amerika Utara. Karakteristik dataset:</p>
        <ul>
            <li>Sumber data: National Institute of Diabetes and Digestive and Kidney Diseases</li>
            <li>Populasi: Wanita keturunan Pima Indians</li>
            <li>Usia: Minimal 21 tahun</li>
            <li>Tujuan: Prediksi diagnostik diabetes berdasarkan pengukuran medis</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Function to create model architecture
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(8,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to load the model and scaler
@st.cache_resource
def load_prediction_model():
    try:
        # Create model with same architecture
        model = create_model()
        
        # Load weights from the saved model
        model.load_weights('diabetes_prediction_model.h5')
        
        # Load scaler
        scaler = joblib.load('diabetes_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.error("Please ensure model files are in the correct location.")
        st.stop()

# Try to load the model and scaler
model, scaler = load_prediction_model()

# Form input dengan desain yang lebih baik
st.markdown("""
    <div style='padding: 1rem; background-color: #ffffff; border-radius: 10px; margin: 1rem 0;'>
        <h3>📋 Form Pengisian Data Pasien</h3>
        <p>Silakan masukkan data pasien sesuai dengan parameter yang diminta:</p>
    </div>
""", unsafe_allow_html=True)

# Membuat layout kolom yang lebih rapi
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div style='padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>
            <h4>Data Fisik</h4>
        </div>
    """, unsafe_allow_html=True)
    
    pregnancies = st.number_input("Jumlah Kehamilan", 
                                min_value=0, 
                                max_value=20, 
                                value=0,
                                help="Jumlah kehamilan yang pernah dialami")
    
    age = st.number_input("Usia (tahun)", 
                         min_value=21, 
                         max_value=120, 
                         value=30,
                         help="Usia minimal 21 tahun sesuai dengan dataset")
    
    bmi = st.number_input("Indeks Massa Tubuh (IMT/BMI)", 
                         min_value=0.0, 
                         max_value=100.0, 
                         value=25.0,
                         help="Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)²)")
    
    skin_thickness = st.number_input("Ketebalan Lipatan Kulit (mm)", 
                                   min_value=0, 
                                   max_value=100, 
                                   value=20,
                                   help="Ketebalan lipatan kulit trisep")

with col2:
    st.markdown("""
        <div style='padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>
            <h4>Data Klinis</h4>
        </div>
    """, unsafe_allow_html=True)
    
    glucose = st.number_input("Kadar Glukosa (mg/dL)", 
                            min_value=0, 
                            max_value=500, 
                            value=120,
                            help="Kadar glukosa plasma setelah 2 jam dalam tes toleransi glukosa oral")
    
    blood_pressure = st.number_input("Tekanan Darah (mmHg)", 
                                   min_value=0, 
                                   max_value=300, 
                                   value=70,
                                   help="Tekanan darah diastolik")
    
    insulin = st.number_input("Kadar Insulin (µU/ml)", 
                            min_value=0, 
                            max_value=1000, 
                            value=80,
                            help="Kadar insulin serum dalam 2 jam")
    
    diabetes_pedigree = st.number_input("Riwayat Diabetes Keluarga", 
                                      min_value=0.0, 
                                      max_value=5.0, 
                                      value=0.5,
                                      help="Fungsi riwayat diabetes keluarga (skor yang menunjukkan kemungkinan diabetes berdasarkan riwayat keluarga)")

# Tombol prediksi dengan desain yang lebih menarik
st.markdown("<br>", unsafe_allow_html=True)

# Function to make prediction
def predict_diabetes(input_data):
    # Create DataFrame with the input data
    input_df = pd.DataFrame([input_data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = prediction_prob > 0.5
    
    return prediction, prediction_prob

predict_button = st.button("🔍 Analisis Data", type="primary", use_container_width=True)

if predict_button:
    with st.spinner('Sedang memproses data...'):
        # Progress bar animation
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        
        # Prediksi
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, diabetes_pedigree, age]
        
        prediction, probability = predict_diabetes(input_data)
        
        # Hasil dengan desain yang lebih menarik
        st.markdown("""
            <div style='padding: 2rem; background-color: #ffffff; border-radius: 10px; margin: 2rem 0;'>
                <h3>🔬 Hasil Analisis</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Tampilan metrik yang lebih menarik
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
                <div class="metric-card">
                    <h4>Status Prediksi</h4>
            """, unsafe_allow_html=True)
            status = "🔴 Berisiko Diabetes" if prediction else "🟢 Risiko Rendah"
            st.markdown(f"<h2>{status}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("""
                <div class="metric-card">
                    <h4>Probabilitas</h4>
            """, unsafe_allow_html=True)
            st.markdown(f"<h2>{probability:.2%}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown("""
                <div class="metric-card">
                    <h4>Tingkat Kepercayaan</h4>
            """, unsafe_allow_html=True)
            confidence = "Tinggi" if abs(probability - 0.5) > 0.3 else "Sedang" if abs(probability - 0.5) > 0.15 else "Rendah"
            st.markdown(f"<h2>{confidence}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Visualisasi risiko
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Penilaian Risiko")
        st.progress(float(probability))
        risk_color = "red" if probability > 0.75 else "orange" if probability > 0.25 else "green"
        st.markdown(f"<p style='color: {risk_color}; font-size: 1.2em;'>Tingkat Risiko: {probability:.2%}</p>", unsafe_allow_html=True)

        # Interpretasi hasil
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction:
            st.error("""
                ### 📋 Interpretasi Hasil
                Model mendeteksi risiko diabetes yang tinggi. Rekomendasi:
                
                1. ⚕️ **Konsultasi Medis**: Segera konsultasikan hasil ini dengan tenaga medis profesional
                2. 🩺 **Pemeriksaan Lanjutan**: Lakukan tes gula darah lengkap
                3. 📝 **Dokumentasi**: Catat pola makan dan aktivitas fisik Anda
                4. 🔍 **Pemantauan**: Ukur gula darah secara teratur
            """)
        else:
            st.success("""
                ### 📋 Interpretasi Hasil
                Model mendeteksi risiko diabetes yang rendah. Rekomendasi:
                
                1. 🏃‍♀️ **Gaya Hidup Sehat**: Pertahankan pola hidup aktif
                2. 🥗 **Pola Makan**: Jaga pola makan seimbang
                3. ⏰ **Pemeriksaan Rutin**: Tetap lakukan pemeriksaan kesehatan berkala
                4. 📊 **Monitor**: Pantau perubahan kondisi kesehatan
            """)

        # Analisis faktor risiko
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔍 Analisis Faktor Risiko")
        
        risk_factors = []
        if glucose > 140: risk_factors.append("Kadar glukosa tinggi (> 140 mg/dL)")
        if bmi > 30: risk_factors.append("IMT tinggi (> 30)")
        if blood_pressure > 90: risk_factors.append("Tekanan darah tinggi (> 90 mmHg)")
        if age > 40: risk_factors.append("Usia di atas 40 tahun")
        if diabetes_pedigree > 0.8: risk_factors.append("Riwayat diabetes keluarga signifikan")

        if risk_factors:
            for factor in risk_factors:
                st.warning(f"⚠️ {factor}")
        else:
            st.info("✅ Tidak ada faktor risiko mayor yang teridentifikasi")

# Footer dengan disclaimer
st.markdown("""
    <div class="footer">
        <p><strong>Tentang Sistem Prediksi Ini</strong></p>
        <p>Sistem prediksi ini dikembangkan berdasarkan data dari populasi khusus (wanita Pima Indians) 
        dan mungkin tidak sepenuhnya representatif untuk populasi lain.</p>
        <hr>
        <p><small>Disclaimer: Aplikasi ini hanya untuk tujuan tugas, edukasi dan penelitian. 
        Hasil prediksi tidak menggantikan diagnosis medis profesional.</small></p>
    </div>
""", unsafe_allow_html=True)