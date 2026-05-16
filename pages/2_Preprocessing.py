import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import random

# Configure page
st.set_page_config(page_title="Medical OCR - Preprocessing", layout="wide")

st.title("Interactive Data Preprocessing")
st.markdown("### **Standarisasi & Pembersihan Citra (Image Cleaning)**")
st.divider()

st.write("""
Halaman ini adalah laboratorium simulasi *pipeline* Preprocessing. Di sini, kita bereksperimen dengan parameter Computer Vision untuk mengubah gambar `raw` yang memiliki ukuran dan kontras bervariasi menjadi gambar `processed` yang bersih, terstandarisasi, dan siap untuk tahap ekstraksi fitur.
""")

BASE_RAW_DIR = os.path.join("dataset", "raw")

splits_map = {
    "Training": "training_words",
    "Testing": "testing_words",
    "Validation": "validation_words"
}

st.markdown("#### 1. Pilih Sumber Gambar")
col_split, col_btn = st.columns([1, 2])

with col_split:
    selected_split = st.selectbox("Pilih Split Dataset:", list(splits_map.keys()))  
    
target_dir = os.path.join(BASE_RAW_DIR, selected_split, splits_map[selected_split])

with col_btn:
    st.write("") 
    st.write("")
    if st.button("Ambil Gambar Acak dari Folder Ini"):
        if os.path.exists(target_dir):
            valid_images = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if valid_images:
                st.session_state['prep_img_name'] = random.choice(valid_images)
                st.session_state['prep_img_path'] = os.path.join(target_dir, st.session_state['prep_img_name'])
            else:
                st.warning(f"Tidak ada gambar di dalam folder {target_dir}")
        else:
            st.error(f"Folder tidak ditemukan: {target_dir}")

st.write("---")

if 'prep_img_path' in st.session_state and os.path.exists(st.session_state['prep_img_path']):
    img_pil = Image.open(st.session_state['prep_img_path']).convert('L')
    img_array = np.array(img_pil)
    img_name = st.session_state['prep_img_name']
    
    st.markdown("#### 2. Tuning Parameter Computer Vision")
    
    col_orig, col_controls, col_proc = st.columns([1.2, 1, 1.2])
    
    with col_orig:
        st.write("**Original (Raw Grayscale)**")
        st.image(img_pil, caption=f"File: {img_name}", use_container_width=True)
        st.write(f"Dimensi Asli: {img_array.shape[1]} x {img_array.shape[0]} px")
        
    with col_controls:
        st.write("**Parameter Settings**")
        
        st.write("""
        **Optimal Experiment Parameters:**
        * **Threshold:** 127
        * **Target Size:** 64 x 64
        * **Inverse:** Active (True)
        """)
        
        # Slider interaktif
        thresh_val = st.slider("Binarization Threshold", min_value=0, max_value=255, value=127, step=1)
        resize_val = st.slider("Target Size (N x N)", min_value=32, max_value=128, value=64, step=16)
        
        # Toggle untuk invert
        invert_thresh = st.checkbox("Inverse Threshold", value=True, help="Centang agar teks menjadi putih dan latar menjadi hitam (penting untuk ekstraksi fitur).")
        
    with col_proc:
        st.write("**Processed Output (Clean)**")
        
        # 1. Binarization (Thresholding)
        thresh_type = cv2.THRESH_BINARY_INV if invert_thresh else cv2.THRESH_BINARY
        _, binarized = cv2.threshold(img_array, thresh_val, 255, thresh_type)
        
        # 2. Resize
        resized = cv2.resize(binarized, (resize_val, resize_val))
        
        # Tampilkan Hasil
        st.image(resized, caption=f"Size: {resize_val}x{resize_val} | Thresh: {thresh_val}", use_container_width=True)
        st.write(f"Dimensi Akhir: {resize_val} x {resize_val} px")
    
    st.divider()
    col_save1, col_save2 = st.columns([1, 2])
    
    with col_save1:
        # Mengonversi gambar matriks OpenCV menjadi format bytes agar bisa didownload lewat browser
        _, img_encoded = cv2.imencode('.jpg', resized)
        img_bytes = img_encoded.tobytes()
        
        # Tombol download langsung ke perangkat lokal user
        st.download_button(
            label="Download Gambar Clean",
            data=img_bytes,
            file_name=f"clean_{img_name}",
            mime="image/jpeg",
            type="primary"
        )
        
    with col_save2:
        st.info("**Catatan untuk Evaluator:** Aplikasi ini berfungsi murni sebagai simulator interaktif untuk kalibrasi visual. Menekan tombol unduh akan mengunduh gambar hasil pembersihan langsung ke komputer lokal Anda tanpa mengubah isi direktori repositori sistem.")

else:
    st.info("Silakan pilih split dataset dan klik tombol 'Ambil Gambar Acak' untuk memulai proses *tuning*.")