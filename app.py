import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Photo Editor", layout="wide")

st.title("📸 Photo Editor using OpenCV & Streamlit")

# ------------------ SESSION STATE ------------------
if "image" not in st.session_state:
    st.session_state.image = None

if "edited" not in st.session_state:
    st.session_state.edited = None

# ------------------ FUNCTIONS ------------------
def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

def warm_filter(img):
    increase_lookup = np.interp(np.arange(256), [0, 64, 128, 256], [0, 80, 160, 256]).astype("uint8")
    decrease_lookup = np.interp(np.arange(256), [0, 64, 128, 256], [0, 50, 100, 256]).astype("uint8")
    b, g, r = cv2.split(img)
    r = cv2.LUT(r, increase_lookup)
    b = cv2.LUT(b, decrease_lookup)
    return cv2.merge((b, g, r))

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def portrait_blur(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    cv2.circle(mask, (w//2, h//2), min(w, h)//4, (255,255,255), -1)
    blurred = cv2.GaussianBlur(img, (25,25), 0)
    return np.where(mask==np.array([255,255,255]), img, blurred)

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.image = np.array(image)

# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.header("Controls")

width = st.sidebar.slider("Width", 100, 1000, 500)
height = st.sidebar.slider("Height", 100, 1000, 500)

brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

gray = st.sidebar.checkbox("Grayscale")
blur = st.sidebar.checkbox("Blur")
sharpen = st.sidebar.checkbox("Sharpen")
warm = st.sidebar.checkbox("Warm Filter")
portrait = st.sidebar.checkbox("Portrait Blur")

st.sidebar.header("Extra Features")
edge = st.sidebar.checkbox("Edge Detection")
cartoon = st.sidebar.checkbox("Cartoon Effect")

apply = st.sidebar.button("✅ Apply Changes")

# ------------------ PROCESS IMAGE ------------------
if apply and st.session_state.image is not None:
    img = st.session_state.image.copy()

    # Resize
    img = cv2.resize(img, (width, height))

    # Brightness & Contrast
    img = adjust_brightness_contrast(img, brightness, contrast)

    # Filters
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        img = cv2.GaussianBlur(img, (15,15), 0)

    if sharpen:
        img = sharpen_image(img)

    if warm and len(img.shape) == 3:
        img = warm_filter(img)

    if portrait and len(img.shape) == 3:
        img = portrait_blur(img)

    # Extra Features
    if edge:
        img = cv2.Canny(img, 100, 200)

    if cartoon and len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.medianBlur(gray_img, 5)
        edges = cv2.adaptiveThreshold(
            blur_img, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(img, 9, 250, 250)
        img = cv2.bitwise_and(color, color, mask=edges)

    st.session_state.edited = img

# ------------------ DISPLAY ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    if st.session_state.image is not None:
        st.image(st.session_state.image, width=400)

with col2:
    st.subheader("Edited Image")
    if st.session_state.edited is not None:
        st.image(st.session_state.edited, width=400)

# ------------------ DOWNLOAD ------------------
if st.session_state.edited is not None:
    img = st.session_state.edited

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    _, buffer = cv2.imencode(".png", img)

    st.download_button(
        "⬇️ Download Image",
        buffer.tobytes(),
        file_name="edited.png",
        mime="image/png"
    )
