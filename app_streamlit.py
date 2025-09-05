
import streamlit as st

import easyocr

import cv2

import numpy as np

import io

import csv

import datetime

from PIL import Image

# ---------------- Config ----------------

SAVE_DIR = "streamlit_captures"

import os

os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize EasyOCR reader once (slow on first load)

reader = easyocr.Reader(['en'])

st.set_page_config(page_title="EasyOCR App", layout="centered")

st.title("📷 Easy OCR — Upload & Read")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp"])

def preprocess_for_ocr(img_bgr):

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    enhanced = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(

        enhanced, 255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        35, 11

    )

    return thresh

def annotate_image(orig_bgr, results):

    out = orig_bgr.copy()

    for (bbox, text, conf) in results:

        try:

            (tl, tr, br, bl) = bbox

            x1, y1 = map(int, tl)

            x2, y2 = map(int, br)

            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)

            txt = f"{text} ({conf:.2f})"

            cv2.putText(out, txt, (x1, max(y1-8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        except Exception:

            continue

    return out

if uploaded:

    # read into OpenCV format

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original image")

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Preprocess & show preview

    thresh = preprocess_for_ocr(img_bgr)

    st.subheader("Preprocessed (threshold) preview")

    st.image(thresh, clamp=True, channels="GRAY", use_column_width=True)

    # OCR (pass numpy array)

    results = reader.readtext(thresh, detail=1, paragraph=False,

                              contrast_ths=0.05, adjust_contrast=0.7)

    # Annotate

    annotated = annotate_image(img_bgr, results)

    st.subheader("Annotated")

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Show text results & save CSV

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_fname = os.path.join(SAVE_DIR, f"raw_{timestamp}.jpg")

    ann_fname = os.path.join(SAVE_DIR, f"annotated_{timestamp}.jpg")

    cv2.imwrite(raw_fname, img_bgr)

    cv2.imwrite(ann_fname, annotated)

    st.subheader("Extracted Text")

    if results:

        rows = []

        for bbox, text, confidence in results:

            st.write(f"- **{text}** (Confidence: {confidence:.2f})")

            try:

                tl = tuple(map(int, bbox[0]))

                br = tuple(map(int, bbox[2]))

                x1,y1 = tl; x2,y2 = br

            except Exception:

                x1=y1=x2=y2=""

            rows.append([timestamp, raw_fname, ann_fname, text, f"{confidence:.2f}", x1,y1,x2,y2])

        # produce CSV for download

        csv_buffer = io.StringIO()

        writer = csv.writer(csv_buffer)

        writer.writerow(["timestamp","raw_frame","annotated_frame","text","confidence","x1","y1","x2","y2"])

        writer.writerows(rows)

        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button("Download CSV", csv_bytes, file_name=f"ocr_{timestamp}.csv", mime="text/csv")

    else:

        st.write("⚠️ No text detected.")

    st.success(f"Saved images: {raw_fname}, {ann_fname}")
 