import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from PIL import Image
import os
import random

from utils.cv_helpers import build_gabor_filter, calculate_zernike_heatmap, make_gradcam_style

# Configure page
st.set_page_config(page_title="Medical OCR - Feature Extraction", layout="wide")

st.title("Feature Extraction & XAI Visualizer")
st.markdown("### **Melihat Seperti Algoritma Membaca (*Explainable AI*)**")
st.divider()

st.write("""
Tahap **Feature Extraction** adalah inti dari *Computer Vision*. Di halaman ini, kita menggunakan teknik *Explainable AI* (XAI) untuk memvisualisasikan peta aktivasi (*Activation Heatmaps*) bergaya Grad-CAM. Visualisasi ini membuktikan bahwa setiap algoritma mengekstrak karakteristik tulisan tangan yang berbeda-beda, dan mengapa **Super Feature Fusion** menutupi kelemahan fitur tunggal.
""")

BASE_PROCESSED_DIR = os.path.join("dataset", "processed")

splits_map = {
    "Training": "training_words",
    "Testing": "testing_words",
    "Validation": "validation_words"
}

st.markdown("#### 1. Pilih Gambar Input (Clean Data)")
input_method = st.radio("Pilih sumber gambar:", ["Ambil Acak dari Folder Processed", "Upload Gambar Sendiri"], horizontal=True)

img_pil = None
img_name = "custom_upload.jpg"

if input_method == "Ambil Acak dari Folder Processed":
    col_split, col_btn = st.columns([1, 2])
    with col_split:
        selected_split = st.selectbox("Pilih Split Dataset:", list(splits_map.keys()))
        target_dir = os.path.join(BASE_PROCESSED_DIR, selected_split, splits_map[selected_split])
    
    with col_btn:
        st.write("") 
        st.write("")
        if st.button("Ambil Gambar Acak"):
            if os.path.exists(target_dir):
                valid_images = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if valid_images:
                    st.session_state['feat_img_name'] = random.choice(valid_images)
                    st.session_state['feat_img_path'] = os.path.join(target_dir, st.session_state['feat_img_name'])
                else:
                    st.warning(f"Folder {target_dir} kosong.")
            else:
                st.error(f"Folder tidak ditemukan: {target_dir}. Pastikan data hasil preprocessing sudah dimasukkan ke sini.")

    if 'feat_img_path' in st.session_state and os.path.exists(st.session_state['feat_img_path']):
        img_pil = Image.open(st.session_state['feat_img_path']).convert('L')
        img_name = st.session_state['feat_img_name']
else:
    uploaded_file = st.file_uploader("Upload Processed Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert('L')
        img_name = uploaded_file.name


if img_pil is not None:
    img_array = np.array(img_pil)
    img_resized = cv2.resize(img_array, (64, 64))
    img_norm = img_resized.astype(np.float64) / 255.0
    
    st.divider()
    st.markdown("#### 2. Ekstraksi Fitur & Visualisasi XAI")
    
    col_img, col_extract = st.columns([1, 2])
    
    with col_img:
        st.write("**Gambar Input (64x64)**")
        st.image(img_resized, caption=f"File: {img_name}", width=250)
        
        feature_choice = st.selectbox(
            "Pilih Algoritma Ekstraksi:", 
            ["Select Method", "HOG (Edge Activations)", "LBP (Texture Activations)", "Gabor (Frequency Activations)", "Zernike (Global Shape)", "FUSION (Super Combined Map)"]
        )
    
    with col_extract:
        if feature_choice != "Select Method":
            with st.spinner(f"Menghitung Spatial Activations untuk {feature_choice}..."):
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                fig.tight_layout(pad=0)
                
                # Gambar background (grayscale)
                ax.imshow(img_resized, cmap='gray')
                
                heatmap = None
                
                if feature_choice == "HOG (Edge Activations)":
                    from skimage import exposure
                    _, hog_raw = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    hog_rescaled = exposure.rescale_intensity(hog_raw, in_range=(0, 10))
                    heatmap = make_gradcam_style(hog_rescaled, ksize=11)
                    st.info("**HOG** fokus menyala pada kerangka tepi tulisan (garis luar huruf).")
                    
                elif feature_choice == "LBP (Texture Activations)":
                    lbp_raw = local_binary_pattern(img_resized, 8, 1, method='uniform')
                    heatmap = make_gradcam_style(lbp_raw, ksize=15)
                    st.info("**LBP** fokus menyala pada tekstur guratan tinta tebal di dalam huruf.")
                    
                elif feature_choice == "Gabor (Frequency Activations)":
                    gabor_raw = cv2.filter2D(img_resized, cv2.CV_8UC3, build_gabor_filter())
                    heatmap = make_gradcam_style(gabor_raw, ksize=15)
                    st.info("**Gabor** mendeteksi frekuensi spasial yang miring (diagonal).")
                    
                elif feature_choice == "Zernike (Global Shape)":
                    zernike_raw = calculate_zernike_heatmap(img_norm, radius=32, degree=8)
                    heatmap = make_gradcam_style(zernike_raw, ksize=21)
                    st.info("**Zernike** membentuk gumpalan besar yang menandakan ia mengenali siluet keseluruhan kata tanpa peduli detail mikronya.")
                    
                elif feature_choice == "FUSION (Super Combined Map)":
                    from skimage import exposure
                    _, hog_raw = hog(img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
                    hog_rescaled = exposure.rescale_intensity(hog_raw, in_range=(0, 10))
                    lbp_raw = local_binary_pattern(img_resized, 8, 1, method='uniform')
                    gabor_raw = cv2.filter2D(img_resized, cv2.CV_8UC3, build_gabor_filter())
                    zernike_raw = calculate_zernike_heatmap(img_norm, radius=32, degree=8)
                    
                    fusion_raw = (make_gradcam_style(hog_rescaled, 11)*1.5) + (make_gradcam_style(lbp_raw, 15)*0.5) + \
                                (make_gradcam_style(gabor_raw, 15)*1.0) + (make_gradcam_style(zernike_raw, 21)*1.2)
                    heatmap = make_gradcam_style(fusion_raw, ksize=9)
                    st.info("**Feature Fusion:** Perhatikan bagaimana 'titik buta' dari algoritma lain kini tertutupi. Peta aktivasi menjadi padat dan komprehensif, inilah rahasia di balik akurasi tertinggi model kita!")

                if heatmap is not None:
                    ax.imshow(heatmap, cmap='jet', alpha=0.6, interpolation='bicubic')
                    ax.axis('off')
                    
                    st.pyplot(fig, use_container_width=False)
else:
    st.info("Silakan pilih/upload gambar untuk memulai proses ekstraksi fitur.")