import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Draw a Digit (0–9)")

model = load_model("digit_model.keras", compile=False)

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0].astype(np.uint8)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"🔢 Predicted Digit: **{predicted_digit}**")
    st.bar_chart(prediction[0])
