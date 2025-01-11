import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Prediksi Risiko Diabetes - Pima Indians",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS umum
st.markdown("""
    <style>
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 20px;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .card {
        background-color: var(--card-background);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .card:hover {
        background-color: var(--hover-color);
        transform: translateY(-2px);
    }
    .info-tooltip {
        color: #7c8db0;
        font-size: 0.9em;
        margin-top: 0.5rem;
    }
    .step-container {
        padding: 20px;
        background-color: var(--card-background);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: var(--card-background);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function untuk validasi input
def validate_input(value, min_val, max_val, name):
    if value < min_val or value > max_val:
        st.error(f"{name} harus berada di antara {min_val} dan {max_val}")
        return False
    return True

# Function to create model
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
        model = create_model()
        model.load_weights('diabetes_prediction_model.h5')
        scaler = joblib.load('diabetes_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.error("Please ensure model files are in the correct location.")
        st.stop()

# Load model and scaler
model, scaler = load_prediction_model()

# Load sample data untuk visualisasi
@st.cache_data
def load_sample_data():
    # Simulasi data sample - ganti dengan data training asli jika tersedia
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame({
        'Pregnancies': np.random.randint(0, 15, 100),
        'Glucose': np.random.normal(120, 30, 100),
        'BloodPressure': np.random.normal(70, 10, 100),
        'SkinThickness': np.random.normal(20, 10, 100),
        'Insulin': np.random.normal(80, 20, 100),
        'BMI': np.random.normal(25, 5, 100),
        'DiabetesPedigreeFunction': np.random.normal(0.5, 0.3, 100),
        'Age': np.random.normal(30, 10, 100)
    })

sample_data = load_sample_data()

# Fungsi untuk membuat radar chart
def create_radar_chart(input_data, sample_data):
    categories = sample_data.columns
    input_normalized = []
    avg_normalized = []
    
    for i, cat in enumerate(categories):
        min_val = sample_data[cat].min()
        max_val = sample_data[cat].max()
        input_normalized.append((input_data[i] - min_val) / (max_val - min_val))
        avg_normalized.append((sample_data[cat].mean() - min_val) / (max_val - min_val))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=input_normalized,
        theta=categories,
        fill='toself',
        name='Data Pasien',
        line_color='#FF4B4B'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_normalized,
        theta=categories,
        fill='toself',
        name='Rata-rata Populasi',
        line_color='#1F77B4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Perbandingan dengan Rata-rata Populasi"
    )
    
    return fig

# Fungsi untuk membuat histogram
def create_parameter_histograms(sample_data, input_data):
    fig = make_subplots(rows=4, cols=2, 
                       subplot_titles=sample_data.columns,
                       vertical_spacing=0.12)
    
    for idx, col in enumerate(sample_data.columns, 1):
        row = (idx-1) // 2 + 1
        col_num = 2 if idx % 2 == 0 else 1
        
        fig.add_trace(
            go.Histogram(
                x=sample_data[col],
                name=col,
                nbinsx=30,
                marker_color='#1F77B4'
            ),
            row=row, 
            col=col_num
        )
        
        fig.add_vline(
            x=input_data[idx-1],
            line_color="#FF4B4B",
            line_width=2,
            line_dash="dash",
            row=row,
            col=col_num
        )
    
    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="Distribusi Parameter Dataset",
        title_x=0.5
    )
    
    return fig

# Function untuk prediksi
def predict_diabetes(input_data):
    input_df = pd.DataFrame([input_data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    input_scaled = scaler.transform(input_df)
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = prediction_prob > 0.5
    
    return prediction, prediction_prob

# Sidebar content
with st.sidebar:
    st.header("ℹ️ Informasi Dataset")
    st.info("""
    Dataset Pima Indians dari National Institute of Diabetes and 
    Digestive and Kidney Diseases.
    
    **Karakteristik Dataset:**
    - Populasi: Wanita Pima Indians
    - Usia: ≥ 21 tahun
    - Region: Amerika Utara
    """)
    
    st.header("📚 Referensi")
    st.markdown("""
    1. Smith, J. W., et al. (1988) "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus"
    2. Diabetes Atlas, 9th Edition (2019)
    """)
    
    st.header("🔍 Tentang Parameter")
    st.markdown("""
    **Glukosa:** Kadar glukosa 2 jam setelah tes toleransi glukosa oral
    
    **Tekanan Darah:** Tekanan darah diastolik (mm Hg)
    
    **BMI:** Indeks Massa Tubuh = berat (kg) / (tinggi (m))²
    
    **Diabetes Pedigree:** Fungsi yang menyatakan risiko diabetes berdasarkan riwayat keluarga
    """)

# Main content
st.title("🏥 Sistem Prediksi Risiko Diabetes")
st.markdown("*Dataset: Pima Indians Diabetes Database*")

# Initialize session state for wizard
if 'step' not in st.session_state:
    st.session_state.step = 1

# Progress bar untuk wizard
step = st.session_state.step
total_steps = 3
st.progress(step/total_steps)
st.write(f"Langkah {step} dari {total_steps}")

# Step 1: Data Dasar
if step == 1:
    with st.container():
        st.markdown("""
            <div class="step-container">
                <h3>🧑‍⚕️ Data Dasar Pasien</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Usia (tahun)", 
                min_value=21, 
                max_value=100, 
                value=21,
                help="Usia minimal 21 tahun sesuai dataset"
            )
            
            pregnancies = st.number_input(
                "Jumlah Kehamilan",
                min_value=0,
                max_value=20,
                value=0,
                help="Jumlah kehamilan yang pernah dialami"
            )
        
        with col2:
            bmi = st.number_input(
                "Indeks Massa Tubuh (BMI)",
                min_value=10.0,
                max_value=70.0,
                value=25.0,
                help="BMI = berat (kg) / (tinggi (m))²"
            )
            
            skin_thickness = st.number_input(
                "Ketebalan Lipatan Kulit (mm)",
                min_value=0,
                max_value=100,
                value=20,
                help="Ketebalan lipatan kulit trisep"
            )

        if st.button("Lanjut ➡️"):
            if all([
                validate_input(age, 21, 100, "Usia"),
                validate_input(pregnancies, 0, 20, "Jumlah Kehamilan"),
                validate_input(bmi, 10.0, 70.0, "BMI"),
                validate_input(skin_thickness, 0, 100, "Ketebalan Kulit")
            ]):
                st.session_state.age = age
                st.session_state.pregnancies = pregnancies
                st.session_state.bmi = bmi
                st.session_state.skin_thickness = skin_thickness
                st.session_state.step = 2
                st.rerun()

# Step 2: Data Klinis
elif step == 2:
    with st.container():
        st.markdown("""
            <div class="step-container">
                <h3>🩺 Data Klinis</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            glucose = st.number_input(
                "Kadar Glukosa (mg/dL)",
                min_value=0,
                max_value=500,
                value=120,
                help="Kadar glukosa setelah 2 jam tes toleransi glukosa oral"
            )
            
            blood_pressure = st.number_input(
                "Tekanan Darah (mmHg)",
                min_value=0,
                max_value=300,
                value=70,
                help="Tekanan darah diastolik"
            )
        
        with col2:
            insulin = st.number_input(
                "Kadar Insulin (µU/ml)",
                min_value=0,
                max_value=1000,
                value=80,
                help="Kadar insulin serum dalam 2 jam"
            )
            
            diabetes_pedigree = st.number_input(
                "Riwayat Diabetes Keluarga",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                help="Skor riwayat diabetes dalam keluarga"
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Kembali"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Lanjut ➡️"):
                if all([
                    validate_input(glucose, 0, 500, "Glukosa"),
                    validate_input(blood_pressure, 0, 300, "Tekanan Darah"),
                    validate_input(insulin, 0, 1000, "Insulin"),
                    validate_input(diabetes_pedigree, 0.0, 5.0, "Riwayat Diabetes")
                ]):
                    st.session_state.glucose = glucose
                    st.session_state.blood_pressure = blood_pressure
                    st.session_state.insulin = insulin
                    st.session_state.diabetes_pedigree = diabetes_pedigree
                    st.session_state.step = 3
                    st.rerun()

# Step 3: Analisis dan Hasil
else:
    with st.container():
        st.markdown("""
            <div class="step-container">
                <h3>📊 Analisis dan Hasil</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Collect all input data
        input_data = [
            st.session_state.pregnancies,
            st.session_state.glucose,
            st.session_state.blood_pressure,
            st.session_state.skin_thickness,
            st.session_state.insulin,
            st.session_state.bmi,
            st.session_state.diabetes_pedigree,
            st.session_state.age
        ]

        # Tabs for different visualizations
        tab1, tab2 = st.tabs(["Perbandingan dengan Populasi", "Distribusi Parameter"])
        
        with tab1:
            st.plotly_chart(create_radar_chart(input_data, sample_data), use_container_width=True)
            
            st.info("""
            **Interpretasi Grafik Radar:**
            - Area merah menunjukkan nilai parameter pasien
            - Area biru menunjukkan rata-rata populasi dalam dataset
            - Semakin besar area, semakin tinggi nilainya relatif terhadap populasi
            """)
        
        with tab2:
            st.plotly_chart(create_parameter_histograms(sample_data, input_data), use_container_width=True)
            
            st.info("""
            **Interpretasi Histogram:**
            - Batang biru menunjukkan distribusi nilai dalam dataset
            - Garis merah putus-putus menunjukkan posisi nilai pasien
            - Posisi nilai relatif terhadap distribusi dapat mengindikasikan tingkat keparahan
            """)

        # Prediction section
        st.markdown("### 🔍 Hasil Prediksi")
        
        with st.spinner('Menganalisis data...'):
            prediction, probability = predict_diabetes(input_data)
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Status Prediksi",
                    value="Berisiko Diabetes" if prediction else "Risiko Rendah",
                    delta="Perlu Perhatian" if prediction else "Terkendali"
                )
            
            with col2:
                st.metric(
                    label="Probabilitas",
                    value=f"{probability:.2%}"
                )
            
            with col3:
                confidence = "Tinggi" if abs(probability - 0.5) > 0.3 else "Sedang" if abs(probability - 0.5) > 0.15 else "Rendah"
                st.metric(
                    label="Tingkat Kepercayaan",
                    value=confidence
                )

            # Risk visualization
            st.subheader("📊 Tingkat Risiko")
            st.progress(float(probability))
            risk_color = "red" if probability > 0.75 else "orange" if probability > 0.25 else "green"
            st.markdown(f"<p style='color: {risk_color}; font-size: 1.2em; text-align: center;'>Tingkat Risiko: {probability:.2%}</p>", unsafe_allow_html=True)

            # Risk factors analysis
            st.subheader("🔍 Analisis Faktor Risiko")
            risk_factors = []
            
            # Get values from session state
            current_glucose = st.session_state.glucose
            current_bmi = st.session_state.bmi
            current_blood_pressure = st.session_state.blood_pressure
            current_age = st.session_state.age
            current_diabetes_pedigree = st.session_state.diabetes_pedigree
            
            # Analisis setiap parameter
            if current_glucose > 140:
                risk_factors.append({
                    "factor": "Kadar glukosa tinggi",
                    "value": f"{current_glucose} mg/dL",
                    "recommendation": "Konsultasikan dengan dokter untuk pemeriksaan diabetes lebih lanjut"
                })
            
            if current_bmi > 30:
                risk_factors.append({
                    "factor": "IMT tinggi (obesitas)",
                    "value": f"BMI: {current_bmi:.1f}",
                    "recommendation": "Pertimbangkan program penurunan berat badan dan konsultasi dengan ahli gizi"
                })
            
            if current_blood_pressure > 90:
                risk_factors.append({
                    "factor": "Tekanan darah tinggi",
                    "value": f"{current_blood_pressure} mmHg",
                    "recommendation": "Pantau tekanan darah secara teratur dan konsultasikan dengan dokter"
                })
            
            if current_age > 40:
                risk_factors.append({
                    "factor": "Faktor usia",
                    "value": f"{current_age} tahun",
                    "recommendation": "Lakukan pemeriksaan kesehatan rutin setiap tahun"
                })
            
            if current_diabetes_pedigree > 0.8:
                risk_factors.append({
                    "factor": "Riwayat diabetes keluarga signifikan",
                    "value": f"Skor: {current_diabetes_pedigree:.2f}",
                    "recommendation": "Pertimbangkan skrining diabetes secara berkala"
                })

            # Display risk factors
            if risk_factors:
                for rf in risk_factors:
                    with st.expander(f"⚠️ {rf['factor']} ({rf['value']})"):
                        st.markdown(f"**Rekomendasi:** {rf['recommendation']}")
            else:
                st.success("✅ Tidak ada faktor risiko mayor yang teridentifikasi")

            # Recommendations section
            st.subheader("💡 Rekomendasi Umum")
            if prediction:
                st.warning("""
                    **Berdasarkan hasil analisis, disarankan untuk:**
                    1. Segera konsultasikan hasil ini dengan dokter
                    2. Lakukan pemeriksaan gula darah lengkap
                    3. Catat pola makan dan aktivitas fisik
                    4. Pantau gula darah secara teratur
                    
                    **Penting:**
                    - Hasil ini adalah prediksi awal dan bukan diagnosis final
                    - Hanya dokter yang dapat memberikan diagnosis resmi
                    """)
            else:
                st.success("""
                    **Meski risiko rendah, tetap disarankan untuk:**
                    1. Pertahankan pola hidup sehat
                    2. Lakukan aktivitas fisik teratur
                    3. Jaga pola makan seimbang
                    4. Lakukan pemeriksaan kesehatan rutin
                    
                    **Tips Pencegahan:**
                    - Olahraga minimal 30 menit per hari
                    - Kontrol berat badan
                    - Kurangi konsumsi gula dan karbohidrat sederhana
                    """)

        if st.button("⬅️ Kembali"):
            st.session_state.step = 2
            st.rerun()

        if st.button("🔄 Mulai Baru"):
            st.session_state.step = 1
            st.rerun()

# Footer with disclaimer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: var(--card-background); border-radius: 10px; margin-top: 2rem;'>
        <p><strong>Tentang Sistem Prediksi Ini</strong></p>
        <p>Sistem ini dikembangkan oleh Tim NLP Alan Turing menggunakan dataset Pima Indians dan mungkin tidak sepenuhnya representatif 
        untuk populasi lain.</p>
        <hr>
        <p><small>Disclaimer: Aplikasi ini hanya untuk tujuan tugas, edukasi dan penelitian. 
        Hasil prediksi tidak menggantikan diagnosis medis profesional.</small></p>
    </div>
""", unsafe_allow_html=True)