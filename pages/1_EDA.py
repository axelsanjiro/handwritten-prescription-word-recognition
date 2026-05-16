import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import random

# Configure page
st.set_page_config(page_title="Medical OCR - EDA", layout="wide")

st.title("Exploratory Data Analysis (EDA)")
st.markdown("### **Membedah Karakteristik Dataset Resep Tulisan Tangan**")
st.divider()

st.write("""
Tahap *Exploratory Data Analysis* (EDA) sangat krusial dalam pipeline Computer Vision. Pada halaman ini, kita mengeksplorasi struktur **Doctors' Handwritten Prescription BD Dataset**, memeriksa keseimbangan distribusi kelas obat, serta menganalisis karakteristik visual dari citra tulisan tangan itu sendiri.
""")

# Dataset Path
BASE_DATASET_PATH = os.path.join("dataset", "raw")

@st.cache_data
def load_and_combine_datasets(base_path):
    all_records = []
    
    splits = [
        ("Training", "training_words", "training_labels.csv"),
        ("Testing", "testing_words", "testing_labels.csv"),
        ("Validation", "validation_words", "validation_labels.csv")
    ]
    
    for split_folder, img_folder, csv_filename in splits:
        csv_path = os.path.join(base_path, split_folder, csv_filename)
        img_dir = os.path.join(base_path, split_folder, img_folder)
        
        if os.path.exists(csv_path) and os.path.exists(img_dir):
            try:
                df = pd.read_csv(csv_path)
                
                img_col = df.columns[0]
                lbl_col = df.columns[1]
                
                for _, row in df.iterrows():
                    img_name = str(row[img_col]).strip()
                    label = str(row[lbl_col]).strip()
                    
                    img_path = os.path.join(img_dir, img_name)
                    
                    if not any(img_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        img_path += '.jpg'
                        
                    if os.path.exists(img_path):
                        all_records.append({"label": label, "path": img_path})
            except Exception as e:
                st.error(f"Gagal membaca {csv_filename}: {e}")

    if not all_records:
        return [], [], {}
        
    df_all = pd.DataFrame(all_records)
    
    counts_df = df_all['label'].value_counts().reset_index()
    counts_df.columns = ['label', 'count']
    
    classes = counts_df['label'].tolist()
    counts = counts_df['count'].tolist()
    
    img_dict = df_all.groupby('label')['path'].apply(list).to_dict()
    
    return classes, counts, img_dict

classes, counts, img_dict = load_and_combine_datasets(BASE_DATASET_PATH)


st.markdown("### 1. Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

total_images = sum(counts) if counts else 0
total_classes = len(classes) if classes else 0

col1.metric(label="Total Sampel Data (Gabungan)", value=f"{total_images:,}", delta="Train + Test + Val")
col2.metric(label="Total Kelas (Label)", value=f"{total_classes}", delta="Pharmaceutical Names")
col3.metric(label="Resolusi Asli Rata-rata", value="Bervariasi", delta="Unconstrained")
col4.metric(label="Target Resolusi", value="64 x 64", delta="Standarisasi Model")

st.write("") # Spacing

st.markdown("### 2. Distribusi Kelas Obat (Class Balance)")
st.write("Grafik di bawah ini memvisualisasikan jumlah sampel citra gabungan untuk setiap nama obat aktual. Kamu dapat menyorot (*hover*) pada batang grafik untuk melihat detailnya.")

if counts:
    df_dist = pd.DataFrame({
        "Nama Obat": classes,
        "Jumlah Sampel": counts
    })
    
    # Plotly Bar Chart
    fig_bar = px.bar(
        df_dist, 
        x="Nama Obat", 
        y="Jumlah Sampel", 
        title="Distribusi Jumlah Sampel per Kelas Farmasi (Dataset Gabungan)",
        labels={"Jumlah Sampel": "Total Citra", "Nama Obat": "Kelas Farmasi"},
        color="Jumlah Sampel",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(b=100))
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning(f"Dataset tidak ditemukan! Pastikan struktur foldermu persis seperti ini:\n`cv_final_project/dataset/raw/Training/training_labels.csv`")

with st.expander("Mengapa Distribusi Kelas Penting?"):
    st.write("""
    Dalam Machine Learning (seperti model SVM atau Random Forest), ketidakseimbangan kelas (*Class Imbalance*) dapat menyebabkan model menjadi bias terhadap kelas mayoritas. Penggunaan metrik **F1-Macro** pada tahap klasifikasi dirancang khusus untuk mengatasi variasi jumlah sampel ini, memastikan bahwa setiap kelas dievaluasi secara berimbang.
    """)

st.divider()

st.markdown("### 3. Visualisasi Sampel Citra & Analisis Piksel")
st.write("Eksplorasi karakteristik tulisan guratan miring (*cursive*) dokter dan sebaran intensitas pikselnya dalam format *grayscale* secara interaktif.")

if counts:
    selected_class = st.selectbox("Pilih Kelas Obat untuk dianalisis:", sorted(classes))
    
    if st.button("Ambil Sampel Acak dari Kelas Ini"):
        random_img_path = random.choice(img_dict[selected_class])
        
        col_img, col_hist = st.columns([1, 1.5])

        img_pil = Image.open(random_img_path).convert('L')
        img_array = np.array(img_pil)
        
        with col_img:
            st.markdown(f"#### Sampel Gambar: **{selected_class}**")
            st.image(img_pil, caption=f"File: {os.path.basename(random_img_path)}", width=300)
            
            asal_folder = random_img_path.split(os.sep)[-3]
            st.write(f"**Asal Data:** {asal_folder}")
            st.write(f"**Dimensi Asli:** {img_array.shape[1]} x {img_array.shape[0]} piksel")
            
        with col_hist:
            st.markdown("#### Histogram Intensitas Piksel")
            pixel_values = img_array.flatten()
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=pixel_values,
                nbinsx=256,
                marker_color='#FF7F0E',
                opacity=0.75
            ))
            
            fig_hist.update_layout(
                title_text='Distribusi Nilai Piksel (Grayscale)',
                xaxis_title_text='Intensitas Piksel (0 = Hitam, 255 = Putih)',
                yaxis_title_text='Frekuensi Piksel',
                bargap=0.2,
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("Visualisasi sampel citra akan muncul setelah dataset tersedia di dalam folder.")

with st.expander("Apa yang bisa kita pelajari dari Histogram Piksel?"):
    st.write("""
    Citra resep tulisan tangan idealnya memiliki gaya **bimodal** (dua puncak histogram): 
    1. Puncak tinggi di area nilai piksel terang (mendekati 255), merepresentasikan kertas latar belakang. 
    2. Puncak di area nilai piksel gelap (mendekati 0), merepresentasikan guratan tinta pulpen. 
    
    Melalui analisis ini, kita dapat menentukan parameter **Threshold Binarisasi** yang optimal untuk memisahkan teks dari *background* secara sempurna pada tahapan Preprocessing.
    """)