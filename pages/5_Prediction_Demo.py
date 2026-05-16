import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import Feature Extraction
from skimage.feature import hog, local_binary_pattern

# Import utils 
from utils.cv_helpers import build_gabor_filter, calculate_zernike_heatmap

# Configure page
st.set_page_config(page_title="Medical OCR - Prediction", layout="wide")

st.title("Real-Time Prediction Demo")
st.markdown("### **End-to-End Medical OCR Inference**")
st.divider()


# Mengecek apakah user sudah melatih model di Halaman 4
if 'trained_pipeline' not in st.session_state:
    st.error("**Akses Ditolak:** Belum ada model yang aktif di memori RAM.")
    st.warning("Silakan kembali ke halaman **4. Classification**, lakukan ekstraksi, dan klik tombol **Ekstrak & Latih Model!** terlebih dahulu.")
    st.stop() # Menghentikan eksekusi halaman di bawahnya

# Mengambil objek dari memori
ml_pipeline = st.session_state['trained_pipeline']
kombinasi_aktif = st.session_state['kombinasi_aktif']
fitur_aktif = st.session_state['fitur_aktif']

col_info1, col_info2 = st.columns(2)
col_info1.info(f"**Model Aktif:** `{kombinasi_aktif}`")
col_info2.info(f"**Metode Ekstraksi:** `{fitur_aktif}`")

st.write("---")


st.markdown("#### Unggah Gambar Resep (Tulisan Tangan)")
uploaded_file = st.file_uploader("Pilih gambar potongan kata resep dokter (JPG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mengkonversi file upload menjadi array OpenCV
    image_pil = Image.open(uploaded_file).convert('L') # Convert ke Grayscale
    img_raw = np.array(image_pil)
    
    col_img1, col_img2, col_result = st.columns([1, 1, 2])
    
    with col_img1:
        st.write("**1. Gambar Asli (Raw Upload):**")
        st.image(image_pil, use_container_width=True)
        
    with col_img2:
        # Preprocessing persis seperti di Halaman 4
        _, binarized = cv2.threshold(img_raw, 127, 255, cv2.THRESH_BINARY_INV)
        img_clean = cv2.resize(binarized, (64, 64))
        img_norm = img_clean.astype(np.float64) / 255.0
        
        st.write("**2. Hasil Preprocessing (64x64):**")
        st.image(img_clean, use_container_width=True, clamp=True)
        

    with col_result:
        st.write("**3. Panel Eksekusi & Hasil Prediksi**")
        
        if st.button("Prediksi Teks Obat!", type="primary", use_container_width=True):
            with st.spinner("Mengekstrak fitur dan menjalankan model..."):
                
                if "HOG" in fitur_aktif or "FUSION" in fitur_aktif:
                    f_hog = hog(img_clean, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                
                if "LBP" in fitur_aktif or "FUSION" in fitur_aktif:
                    lbp_img = local_binary_pattern(img_clean, 8, 1, method='uniform')
                    n_bins = int(lbp_img.max() + 1)
                    f_lbp, _ = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                    
                if "Gabor" in fitur_aktif or "FUSION" in fitur_aktif:
                    gabor_img = cv2.filter2D(img_clean, cv2.CV_8UC3, build_gabor_filter())
                    f_gabor = cv2.resize(gabor_img, (8, 8)).flatten() / 255.0
                    
                if "Zernike" in fitur_aktif or "FUSION" in fitur_aktif:
                    f_zer = calculate_zernike_heatmap(img_norm).flatten()
                
                # Menggabungkan fitur sesuai pilihan
                if "HOG" in fitur_aktif: feat_vector = f_hog
                elif "LBP" in fitur_aktif: feat_vector = f_lbp
                elif "Gabor" in fitur_aktif: feat_vector = f_gabor
                elif "Zernike" in fitur_aktif: feat_vector = f_zer
                else: feat_vector = np.concatenate([f_hog, f_lbp, f_gabor, f_zer])
                
                # Reshape array 1D menjadi 2D untuk input Scikit-Learn (1 baris, N kolom)
                X_input = feat_vector.reshape(1, -1)
                
                # Ajaibnya: ml_pipeline otomatis melakukan Standarisasi (Scaler) 
                # dan mereduksi dimensi (PCA) pada X_input sebelum masuk ke SVM/KNN/RF!
                prediction = ml_pipeline.predict(X_input)
                label_prediksi = prediction[0]
                
            # Tampilkan Hasil
            st.success("Prediksi Selesai!")
            st.markdown(f"""
            <div style="background-color:#1E293B; padding:20px; border-radius:10px; border: 2px solid #38BDF8; text-align:center;">
                <p style="color:#94A3B8; margin:0; font-size:16px; text-transform:uppercase;">Model Menebak Kata:</p>
                <h1 style="color:#38BDF8; margin:10px 0 0 0; font-size:54px; letter-spacing: 2px;">{label_prediksi}</h1>
            </div>
            """, unsafe_allow_html=True)
        