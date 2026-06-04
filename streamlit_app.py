import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Retak Beton", page_icon="🏗️", layout="centered")

# 2. Fungsi untuk Memuat Model dengan Caching
# Menggunakan st.cache_resource agar model tidak di-load ulang setiap kali ada interaksi di UI
@st.cache_resource
def load_model():
    # Pastikan file 'model_crack_beton.h5' berada di folder yang sama dengan script ini
    model_path = 'model_crack_beton.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# 3. Setup UI Aplikasi
st.title("🏗️ Aplikasi Klasifikasi Permukaan Beton")
st.write("Unggah gambar permukaan beton untuk menganalisis dan mendeteksi apakah terdapat retakan.")

# Mencoba memuat model
try:
    model = load_model()
    st.sidebar.success("Model berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model. Pastikan file 'model_crack_beton.h5' ada di folder yang sama. Error: {e}")
    st.stop()

# Daftar Kelas (Sesuai dengan urutan saat training)
class_names = ['Retak', 'Tidak_Retak']

# 4. Fitur Upload Gambar
uploaded_file = st.file_uploader("Pilih file gambar beton...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar menggunakan PIL
    image = Image.open(uploaded_file)
    
    # Konversi ke format RGB untuk menghindari isu channel jika gambar transparan (RGBA)
    image = image.convert('RGB')
    
    # Menampilkan gambar yang diunggah
    st.image(image, caption='Gambar Beton yang Diunggah', use_container_width=True)
    
    with st.spinner("Sedang memproses dan menganalisis gambar..."):
        # 5. Pre-processing Gambar agar sesuai input model (150x150)
        img_height = 150
        img_width = 150
        
        # Mengubah ukuran gambar
        img_resized = image.resize((img_width, img_height))
        
        # Mengubah menjadi array numpy
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        
        # Menambahkan dimensi batch (dari [150, 150, 3] menjadi [1, 150, 150, 3])
        img_array = tf.expand_dims(img_array, 0)
        
        # 6. Melakukan Prediksi
        predictions = model.predict(img_array)
        
        # Mengambil skor probabilitas tertinggi (Logika yang sama dengan kode Colab Anda)
        if len(class_names) == 2 and predictions.shape[-1] == 1:
            # Jika model menggunakan klasifikasi biner dengan sigmoid (output 1 neuron)
            predicted_class_idx = 1 if predictions[0][0] >= 0.5 else 0
            konfidensi = predictions[0][0] if predicted_class_idx == 1 else 1 - predictions[0][0]
            konfidensi = konfidensi * 100
        else:
            # Jika menggunakan softmax atau output multi-kolom
            score = tf.nn.softmax(predictions[0]) if len(class_names) > 2 else predictions[0]
            predicted_class_idx = np.argmax(predictions[0])
            konfidensi = np.max(score) * 100
            
        hasil_prediksi = class_names[predicted_class_idx]
        
    # 7. Menampilkan Hasil Prediksi ke Layar
    st.divider()
    st.subheader("Hasil Analisis:")
    
    if hasil_prediksi == 'Retak':
        st.error(f"**Status Permukaan: {hasil_prediksi}**")
    else:
        st.success(f"**Status Permukaan: {hasil_prediksi}**")
        
    st.metric(label="Tingkat Kepercayaan (Confidence Score)", value=f"{konfidensi:.2f}%")
