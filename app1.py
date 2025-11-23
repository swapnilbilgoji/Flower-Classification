# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time
import tempfile
import os
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fruit Ripeness Classifier", layout="wide")

# ---------------------------
# Config / model path
# ---------------------------
MODEL_PATH = "Fruit_Classification_Model.h5"   # <-- change if your model filename is different

FRUITS = ["Apple", "Banana", "Mango", "Orange", "Tomato"]
RIPENESS_CLASSES = ["unripe", "ripe", "overripe"]  # model's class order - ensure matches training

# reasonable default days required to ripen from 'unripe' (average estimate)
DEFAULT_DAYS_TO_RIPE = {
    "Apple": 5,
    "Banana": 3,
    "Mango": 4,
    "Orange": 7,
    "Tomato": 4
}

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path)
    # infer target size if possible
    try:
        shape = model.input_shape  # typically (None, H, W, C)
        if shape and len(shape) >= 3:
            h = shape[1] if shape[1] is not None else 224
            w = shape[2] if shape[2] is not None else 224
        else:
            h, w = 224, 224
    except Exception:
        h, w = 224, 224
    return model, (int(h), int(w))

def preprocess_pil(img: Image.Image, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image_pil, model, target_size):
    arr = preprocess_pil(image_pil, target_size)
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
        # placeholder sample
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
    # unripe_records: list of fruit names that were predicted unripe
    if not unripe_records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No unripe items in history", ha='center', va='center')
        ax.axis('off')
        return fig
    # count per fruit
    counts = {}
    avg_days = {}
    for f in unripe_records:
        counts[f] = counts.get(f, 0) + 1
    # For each fruit, show estimated days (we'll show the days_to_ripen * count as height)
    fruits = sorted(counts.keys())
    days = [days_map.get(f, DEFAULT_DAYS_TO_RIPE.get(f, 4)) for f in fruits]
    # We will display average days (single bar per fruit)
    fig, ax = plt.subplots()
    ax.bar(fruits, days)
    ax.set_ylabel("Estimated days to ripen (avg)")
    ax.set_title("Estimated days to ripen for unripe fruits in session")
    return fig

# ---------------------------
# Session state init
# ---------------------------
if "history" not in st.session_state:
    # history: list of dicts: { 'fruit': <str>, 'ripeness': <str>, 'probs': <list> }
    st.session_state.history = []

# ---------------------------
# Load model
# ---------------------------
try:
    model, TARGET_SIZE = load_model(MODEL_PATH)
    st.sidebar.success(f"Loaded model: {os.path.basename(MODEL_PATH)} (input size {TARGET_SIZE[0]}x{TARGET_SIZE[1]})")
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
# editable days to ripen mapping
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
        # Run prediction
        with st.spinner("Running model inference..."):
            preds = predict(img_pil, model, TARGET_SIZE)
        probs = preds[0].tolist()
        top_idx = int(np.argmax(probs))
        predicted_class = RIPENESS_CLASSES[top_idx]

        # Display results
        st.markdown("### Prediction")
        st.success(f"**{predicted_class.upper()}**")
        st.write("Confidence scores:")
        conf_table = {RIPENESS_CLASSES[i]: float(probs[i]) for i in range(len(RIPENESS_CLASSES))}
        st.json(conf_table)

        # record in session history using the selected fruit from sidebar
        st.session_state.history.append({
            "fruit": selected_fruit,
            "ripeness": predicted_class,
            "probs": probs
        })

        # If unripe, display days to ripen estimate for the selected fruit
        if predicted_class == "unripe":
            days_needed = days_map.get(selected_fruit, DEFAULT_DAYS_TO_RIPE[selected_fruit])
            st.info(f"Estimated days to ripe for **{selected_fruit}** (typical estimate): **{days_needed} days**")
        elif predicted_class == "overripe":
            st.info("This looks overripe — consume or process soon.")
        else:
            st.info("This looks ripe — ready to eat!")

    # show history table (last 20)
    st.markdown("### Prediction history (session)")
    if st.session_state.history:
        hist_df = []
        for i, rec in enumerate(reversed(st.session_state.history[-100:])):
            hist_df.append({
                "idx": len(st.session_state.history) - i,
                "fruit": rec["fruit"],
                "ripeness": rec["ripeness"],
                "confidence_top": max(rec["probs"])
            })
        st.table(hist_df)
    else:
        st.write("No predictions yet. Upload or capture an image to start.")

with col2:
    st.markdown("## Charts")
    # compute counts
    ripeness_counts = {c: 0 for c in RIPENESS_CLASSES}
    unripe_records = []
    for rec in st.session_state.history:
        ripeness_counts[rec["ripeness"]] = ripeness_counts.get(rec["ripeness"], 0) + 1
        if rec["ripeness"] == "unripe":
            unripe_records.append(rec["fruit"])

    st.subheader("Ripeness Distribution")
    fig1 = plot_pie(ripeness_counts, RIPENESS_CLASSES)
    st.pyplot(fig1)

    st.subheader("Estimated days to ripen for unripe fruits in session")
    fig2 = plot_days_to_ripen(unripe_records, days_map)
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Quick stats")
    total_predictions = len(st.session_state.history)
    st.metric("Total predictions (session)", total_predictions)
    if total_predictions > 0:
        ripe_pct = (ripeness_counts["ripe"] / total_predictions) * 100
        unripe_pct = (ripeness_counts["unripe"] / total_predictions) * 100
        overripe_pct = (ripeness_counts["overripe"] / total_predictions) * 100
        st.write(f"Ripe: {ripe_pct:.1f}%  •  Unripe: {unripe_pct:.1f}%  •  Overripe: {overripe_pct:.1f}%")
    else:
        st.write("No predictions to compute percentages yet.")

st.markdown("<small>Tip: Edit days-per-fruit in the left sidebar if you prefer custom ripening estimates.</small>", unsafe_allow_html=True)
