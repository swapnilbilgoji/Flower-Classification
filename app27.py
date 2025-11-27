# app.py
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time
from io import BytesIO
import matplotlib.pyplot as plt
import json
import requests

# Keras helpers for fallback model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

st.set_page_config(page_title="Fruit Ripeness Classifier", layout="wide")

# ---------------------------
# Config / model path
# ---------------------------
MODEL_PATH = "Fruit_Classification_Model.h5"   # local filename (or use Drive)
TARGET_SIZE = (224, 224)  # from notebook

FRUITS = ["Apple", "Banana", "Mango", "Orange", "Tomato"]

# IMPORTANT: this is the REAL class index order used by the model
# (very likely alphabetical by folder name: Overripe, Ripe, Unripe)
# If your train_generator.class_indices is different, EDIT this mapping.
CLASS_INDICES = {
    "overripe": 0,
    "ripe": 1,
    "unripe": 2
}
IDX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}

# For charts / display we can use this order:
RIPENESS_CLASSES_DISPLAY = ["unripe", "ripe", "overripe"]

DEFAULT_DAYS_TO_RIPE = {
    "Apple": 5,
    "Banana": 3,
    "Mango": 4,
    "Orange": 7,
    "Tomato": 4
}

# ---------------------------
# Google Drive download helpers
# ---------------------------
def _get_gdrive_file_id_from_link(link: str):
    if "/file/d/" in link:
        parts = link.split("/file/d/")[1].split("/")
        return parts[0]
    if "id=" in link:
        parts = link.split("id=")[1].split("&")
        return parts[0]
    return None

def download_from_google_drive_requests(file_id: str, destination: str, progress_placeholder=None):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for key, value in session.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    CHUNK_SIZE = 32768
    total = response.headers.get('content-length')
    if total is not None:
        total = int(total)

    with open(destination, "wb") as f:
        downloaded = 0
        if total and progress_placeholder:
            prog = progress_placeholder.progress(0)
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total and progress_placeholder:
                    prog.progress(min(int(downloaded / total * 100), 100))
    if progress_placeholder:
        progress_placeholder.empty()

def try_download_from_drive(destination: str):
    file_id = None
    share_link = None
    try:
        secrets = st.secrets
        file_id = secrets.get("GDRIVE_FILE_ID") if secrets and "GDRIVE_FILE_ID" in secrets else None
        share_link = secrets.get("GDRIVE_SHAREABLE_LINK") if secrets and "GDRIVE_SHAREABLE_LINK" in secrets else None
    except Exception:
        file_id = None
        share_link = None

    if not file_id:
        file_id = os.environ.get("GDRIVE_FILE_ID")
    if not share_link:
        share_link = os.environ.get("GDRIVE_SHAREABLE_LINK")

    if share_link and not file_id:
        parsed = _get_gdrive_file_id_from_link(share_link)
        if parsed:
            file_id = parsed

    if not file_id:
        raise ValueError("Google Drive file id not provided. Set st.secrets['GDRIVE_FILE_ID'] or st.secrets['GDRIVE_SHAREABLE_LINK'].")

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        progress = st.empty()
        progress.info("Downloading model from Google Drive via gdown (this may take a while)...")
        gdown.download(url, destination, quiet=False)
        progress.empty()
    except Exception:
        progress = st.empty()
        progress.info("Downloading model from Google Drive (requests fallback; this may take a while)...")
        download_from_google_drive_requests(file_id, destination, progress_placeholder=progress)

# ---------------------------
# Helpers (image/predict/plots) - matches training: resize 224x224 + /255
# ---------------------------
def preprocess_pil_exact_rescale(img: Image.Image, target_size=(224,224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_preproc(image_pil, model, target_size=(224,224)):
    arr = preprocess_pil_exact_rescale(image_pil, target_size)
    preds = model.predict(arr)
    return preds

def plot_pie(counts, classes):
    total = sum(counts.values())
    labels = []
    sizes = []
    for k in classes:
        labels.append(f"{k} ({counts.get(k,0)})")
        sizes.append(counts.get(k,0))
    if total == 0:
        labels = [f"{c} (0)" for c in classes]
        sizes = [1 for _ in classes]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=classes, autopct='')
        ax.set_title("No predictions yet")
        return fig
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct=lambda pct: f"{pct:.1f}%")
    ax.axis('equal')
    ax.set_title("Ripeness distribution")
    return fig

def plot_days_to_ripen(unripe_records, days_map):
    if not unripe_records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No unripe items in history", ha='center', va='center')
        ax.axis('off')
        return fig
    counts = {}
    for f in unripe_records:
        counts[f] = counts.get(f, 0) + 1
    fruits = sorted(counts.keys())
    days = [days_map.get(f, DEFAULT_DAYS_TO_RIPE.get(f, 4)) for f in fruits]
    fig, ax = plt.subplots()
    ax.bar(fruits, days)
    ax.set_ylabel("Estimated days to ripen (avg)")
    ax.set_title("Estimated days to ripen for unripe fruits in session")
    return fig

# ---------------------------
# Model loader with Drive fallback + weight-by-name fallback (quiet)
# ---------------------------
@st.cache_resource
def load_model_with_fallback(local_path, image_size=(224,224), num_classes=3, debug=False):
    def _sidebar(msg):
        if debug:
            st.sidebar.info(msg)

    # 1) try local full model load first
    if os.path.exists(local_path):
        try:
            model = tf.keras.models.load_model(local_path)
            _sidebar("Loaded full model file successfully (local).")
            return model, image_size
        except Exception:
            _sidebar("Full model load failed (local).")

    # 2) if missing, download from Drive
    if not os.path.exists(local_path):
        try:
            try_download_from_drive(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download model from Google Drive: {e}")

    # 3) try full load again after download
    try:
        model = tf.keras.models.load_model(local_path)
        _sidebar("Loaded full model file successfully (after download).")
        return model, image_size
    except Exception:
        _sidebar("Full model load still failed (after download).")

    # 4) fallback: build single-input model and load weights by_name
    try:
        inputs = Input(shape=(image_size[0], image_size[1], 3), name="input_image")
        base = InceptionV3(weights='imagenet', include_top=False, input_tensor=inputs)

        x = base.output
        x = GlobalAveragePooling2D(name="gap")(x)
        x = Dense(256, activation='relu', name='new_dense_1')(x)
        x = Dropout(0.2, name='dropout_1')(x)
        x = Dense(128, activation='relu', name='new_dense_2')(x)
        outputs = Dense(num_classes, activation='softmax', name='new_output')(x)

        fallback_model = Model(inputs=inputs, outputs=outputs, name="fallback_inception_model")
        fallback_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                               loss=tf.keras.losses.CategoricalCrossentropy(),
                               metrics=['accuracy'])

        fallback_model.load_weights(local_path, by_name=True)
        _sidebar("Loaded weights by name into fallback model.")
        return fallback_model, image_size

    except Exception as e_fallback:
        raise RuntimeError("Failed to load model (full load and weight-by-name fallback both failed). "
                           f"Error detail: {e_fallback}")

# ---------------------------
# Session state init
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Load model (main)
# ---------------------------
try:
    model, TARGET_SIZE_USED = load_model_with_fallback(MODEL_PATH, image_size=TARGET_SIZE, num_classes=len(CLASS_INDICES), debug=False)
except Exception as e:
    st.sidebar.error(f"Failed to load model at '{MODEL_PATH}': {e}")
    st.stop()

# ---------------------------
# Sidebar: Fruit selector & Charts
# ---------------------------
st.sidebar.header("Options")
selected_fruit = st.sidebar.selectbox("Select fruit (for days-to-ripe estimates)", FRUITS)
st.sidebar.markdown("---")
st.sidebar.header("Charts")
st.sidebar.markdown("**Days to ripen (when unripe)** — edit if you want custom estimates")
days_map = {}
for f in FRUITS:
    days_map[f] = st.sidebar.number_input(f"{f} (days)", min_value=0, value=DEFAULT_DAYS_TO_RIPE[f], key=f"days_{f}")
st.sidebar.markdown("")
if st.sidebar.button("Reset stats"):
    st.session_state.history = []
    st.sidebar.success("Cleared history")

# ---------------------------
# Main layout (two columns)
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🍉 Fruit Ripeness Classifier")
    st.write("Upload or capture an image of a fruit. Choose the fruit type (left) to get days-to-ripe estimates for *unripe* predictions.")
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    capture = st.button("📸 Capture from camera")

    img_pil = None
    if capture:
        st.info("Opening camera... waiting 3 seconds before capture.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera.")
        else:
            time.sleep(3)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                st.image(img_pil, caption="Captured image", width=400)
            else:
                st.error("Failed to capture image.")
    elif uploaded_file is not None:
        try:
            img_pil = Image.open(BytesIO(uploaded_file.read()))
            st.image(img_pil, caption="Uploaded image", width=400)
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")

    if img_pil is not None:
        with st.spinner("Running model inference..."):
            preds = predict_with_preproc(img_pil, model, target_size=TARGET_SIZE_USED)

        probs = preds[0].tolist()
        top_idx = int(np.argmax(probs))
        predicted_class = IDX_TO_CLASS.get(top_idx, f"idx_{top_idx}")

        st.markdown("### Prediction")
        st.success(f"**{predicted_class.upper()}**")

        st.write("Confidence scores:")
        # probs are in index order: 0,1,2 -> use IDX_TO_CLASS
        conf_table = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))}
        st.json(conf_table)

        st.session_state.history.append({
            "fruit": selected_fruit,
            "ripeness": predicted_class,
            "probs": probs
        })

        # Simple actionable info
        if predicted_class == "unripe":
            days_needed = days_map.get(selected_fruit, DEFAULT_DAYS_TO_RIPE[selected_fruit])
            st.info(f"Estimated days to ripe for **{selected_fruit}**: **{days_needed} days**")
        elif predicted_class == "overripe":
            st.info("This looks **overripe** — consume, cook, or process it soon.")
        else:  # ripe
            st.info("This looks **ripe** — ready to eat!")

    st.markdown("### Prediction history (session)")
    if st.session_state.history:
        hist_rows = []
        for i, rec in enumerate(reversed(st.session_state.history[-100:])):
            hist_rows.append({
                "idx": len(st.session_state.history) - i,
                "fruit": rec["fruit"],
                "ripeness": rec["ripeness"],
                "confidence_top": max(rec["probs"])
            })
        st.table(hist_rows)
    else:
        st.write("No predictions yet. Upload or capture an image to start.")

with col2:
    st.markdown("## Charts")
    ripeness_counts = {c: 0 for c in RIPENESS_CLASSES_DISPLAY}
    unripe_records = []
    for rec in st.session_state.history:
        if rec["ripeness"] in ripeness_counts:
            ripeness_counts[rec["ripeness"]] += 1
        else:
            ripeness_counts[rec["ripeness"]] = 1
        if rec["ripeness"] == "unripe":
            unripe_records.append(rec["fruit"])

    st.subheader("Ripeness Distribution")
    fig1 = plot_pie(ripeness_counts, RIPENESS_CLASSES_DISPLAY)
    st.pyplot(fig1)

    st.subheader("Estimated days to ripen for unripe fruits in session")
    fig2 = plot_days_to_ripen(unripe_records, days_map)
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Quick stats")
    total_predictions = len(st.session_state.history)
    st.metric("Total predictions (session)", total_predictions)
    if total_predictions > 0:
        ripe_pct = (ripeness_counts["ripe"] / total_predictions) * 100 if "ripe" in ripeness_counts else 0.0
        unripe_pct = (ripeness_counts["unripe"] / total_predictions) * 100 if "unripe" in ripeness_counts else 0.0
        overripe_pct = (ripeness_counts["overripe"] / total_predictions) * 100 if "overripe" in ripeness_counts else 0.0
        st.write(f"Ripe: {ripe_pct:.1f}%  •  Unripe: {unripe_pct:.1f}%  •  Overripe: {overripe_pct:.1f}%")
    else:
        st.write("No predictions to compute percentages yet.")

st.markdown("<small>Tip: Edit days-per-fruit in the left sidebar if you prefer custom ripening estimates.</small>", unsafe_allow_html=True)
