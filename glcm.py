import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from skimage import color
from PIL import Image

st.set_page_config(
    page_title="Analisis GLCM",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
FITUR = {
    "Contrast":      ("contrast",      "Seberapa besar perbedaan intensitas antar piksel. Nilai tinggi = tekstur kasar."),
    "Dissimilarity": ("dissimilarity", "Perbedaan lokal antar piksel. Mirip Contrast tapi lebih sensitif."),
    "Homogeneity":   ("homogeneity",   "Seberapa seragam tekstur. Nilai tinggi = tekstur halus."),
    "Energy":        ("energy",        "Seberapa berulang pola tekstur. Nilai tinggi = pola teratur."),
    "Correlation":   ("correlation",   "Keterkaitan antar piksel. Nilai ±1 = ada pola kuat."),
}

ANGLE_LABELS = ["0°", "45°", "90°", "135°"]
ANGLE_VALS   = [0, np.pi/4, np.pi/2, 3*np.pi/4]

def hitung_glcm(gray_img, jarak, levels):
    gray_q = (gray_img / 256 * levels).astype(np.uint8)
    return graycomatrix(
        gray_q,
        distances=[jarak],
        angles=ANGLE_VALS,
        levels=levels,
        symmetric=True,
        normed=True,
    ), gray_q

# ── Judul ─────────────────────────────────────────────────────────────────────
st.title(" Analisis Tekstur GLCM")
st.caption("Gray-Level Co-occurrence Matrix")
st.divider()

# ── Upload + Parameter (satu baris) ──────────────────────────────────────────
col_up, col_par = st.columns([2, 1])

with col_up:
    uploaded = st.file_uploader("Upload Gambar (JPG / PNG)", type=["jpg", "jpeg", "png"])

with col_par:
    st.markdown("**⚙️ Parameter GLCM**")
    jarak  = st.slider("Jarak antar piksel (d)", 1, 5, 1,
                    help="Seberapa jauh pasangan piksel yang dibandingkan")
    levels = st.select_slider("Jumlah level abu-abu", options=[8, 16, 32, 64, 128, 256], value=64,
                            help="Semakin kecil = proses lebih cepat")

# ── Stop jika belum upload ────────────────────────────────────────────────────
if uploaded is None:
    st.info("👆 Silakan upload gambar terlebih dahulu untuk memulai analisis.")
    st.stop()

# ── Proses gambar ─────────────────────────────────────────────────────────────
try:
    pil_img = Image.open(uploaded).convert("RGB")
except Exception as e:
    st.error(f"Gagal membaca gambar: {e}")
    st.stop()

np_img   = np.array(pil_img)
gray_f   = color.rgb2gray(np_img)
gray_256 = (gray_f * 255).astype(np.uint8)

glcm, gray_q = hitung_glcm(gray_256, jarak, levels)


# BAGIAN 1 — Pratinjau Gambar
st.subheader("Pratinjau Gambar")
st.caption(f"Ukuran: {pil_img.width} × {pil_img.height} px  |  Level abu-abu: {levels}  |  Jarak: d = {jarak}")

c1, c2, c3 = st.columns(3)

with c1:
    st.image(pil_img, caption="Gambar Asli (RGB)", use_container_width=True)

with c2:
    st.image(gray_256, caption="Grayscale", use_container_width=True, clamp=True)

with c3:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(gray_256.ravel(), bins=64, color="#3b82f6", edgecolor="none", alpha=0.85)
    ax.set_title("Histogram Intensitas", fontsize=10)
    ax.set_xlabel("Nilai Piksel (0–255)", fontsize=8)
    ax.set_ylabel("Jumlah Piksel", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()

# BAGIAN 2 — Fitur Haralick
st.subheader("2️⃣ Fitur Haralick (rata-rata dari 4 sudut)")
cols = st.columns(5)
for i, (nama, (prop, desc)) in enumerate(FITUR.items()):
    val = graycoprops(glcm, prop).mean()
    with cols[i]:
        st.metric(label=nama, value=f"{val:.4f}")
        st.caption(desc)

st.divider()

# BAGIAN 3 — Heatmap GLCM
st.subheader("3️⃣ Visualisasi Matriks GLCM (Heatmap)")

col_ctrl, _ = st.columns([1, 2])
with col_ctrl:
    sudut_pilih = st.radio("Pilih sudut:", ANGLE_LABELS, horizontal=True)

angle_idx = ANGLE_LABELS.index(sudut_pilih)
glcm_2d   = glcm[:, :, 0, angle_idx]

# Downsample untuk tampilan
disp_lv = min(levels, 64)
if levels > 64:
    f = levels // 64
    glcm_vis = glcm_2d.reshape(disp_lv, f, disp_lv, f).sum(axis=(1, 3))
else:
    glcm_vis = glcm_2d

col_hm, col_info = st.columns([2, 1])

with col_hm:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(glcm_vis, ax=ax2, cmap="Blues",
                cbar=True, xticklabels=False, yticklabels=False)
    ax2.set_title(f"GLCM Matrix — Sudut {sudut_pilih}, d={jarak}", fontsize=10)
    ax2.set_xlabel("Intensitas piksel j", fontsize=8)
    ax2.set_ylabel("Intensitas piksel i", fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with col_info:
    st.markdown("**💡 Cara membaca heatmap:**")
    st.markdown(f"""
Setiap kotak pada heatmap menunjukkan **seberapa sering** pasangan piksel dengan intensitas tertentu muncul berdampingan pada sudut **{sudut_pilih}**.

- Warna **biru gelap** = jarang muncul
- Warna **biru terang** = sering muncul

**Pola diagonal** → piksel cenderung punya intensitas yang sama → tekstur **halus/seragam**.

**Pola menyebar** → banyak variasi intensitas → tekstur **kasar/kompleks**.
""")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# BAGIAN 4 — Grafik per Sudut
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("4️⃣ Perbandingan Nilai Fitur per Sudut")
st.caption("Jika nilai antar sudut hampir sama → tekstur tidak bergantung arah (isotropik).")

per_angle = {}
for nama, (prop, _) in FITUR.items():
    per_angle[nama] = graycoprops(glcm, prop)[0].tolist()

fig3, axes = plt.subplots(1, 5, figsize=(14, 3.5))
colors = ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]

for ax, (nama, vals) in zip(axes, per_angle.items()):
    bars = ax.bar(ANGLE_LABELS, vals, color=colors, edgecolor="none", width=0.55)
    ax.set_title(nama, fontsize=9, fontweight="bold")
    ax.set_xlabel("Sudut", fontsize=7)
    ax.tick_params(labelsize=7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

fig3.tight_layout(pad=1.5)
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

st.divider()