#!/usr/bin/env python3
# OCR Live Panel with Batch-code post-processing fixes
# - Left: small live feed
# - Center: small annotated preview (updated when you press 'p')
# - Right: large text panel with OCR history and counts
#
# Controls:
# - p : capture current frame, run OCR and update annotated preview + text panel
# - q : quit

import cv2
import easyocr
import datetime
import os
import sys
import csv
import traceback
import select
import numpy as np
import re
from collections import deque, Counter

# ----------------- Configuration -----------------
SAVE_DIR = "captured_frames"
TXT_LOG = "ocr_log.txt"
CSV_LOG = "ocr_log.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

# Create CSV header if missing
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["timestamp", "raw_frame", "annotated_frame", "text", "confidence", "x1", "y1", "x2", "y2"])

# Initialize EasyOCR reader (may take a moment)
print("Initializing EasyOCR reader (this may take a moment)...")
# Restrict to english. Set gpu=True if you have a supported GPU.
reader = easyocr.Reader(['en'], gpu=False)

# ----------------- Helpers -----------------
def imshow_available():
    """Try to check if cv2.imshow likely works in this environment."""
    if not hasattr(cv2, "imshow"):
        return False
    try:
        cv2.namedWindow("test_window", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test_window")
        return True
    except Exception:
        return False

def preprocess_for_ocr(img_bgr):
    """Return thresholded image optimized for OCR on reflective surfaces (CLAHE + sharpening)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Sharpen to emphasize stroke edges (helps 1 vs 9)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], dtype=np.float32)
    sharp = cv2.filter2D(enhanced, -1, kernel)
    # Optional denoising (uncomment if very noisy)
    # sharp = cv2.fastNlMeansDenoising(sharp, None, 8, 7, 21)
    # Adaptive threshold to handle uneven lighting
    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 11
    )
    return thresh

def safe_open_file(path):
    """Open file by OS default when possible (Windows: os.startfile)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        else:
            opener = "xdg-open" if sys.platform.startswith("linux") else "open"
            os.system(f'{opener} "{path}"')
    except Exception:
        pass

def normalize_text_for_counting(text: str) -> str:
    """General normalization applied before counting:
       - uppercase
       - strip surrounding whitespace
       - remove stray chars except A-Z0-9-"""
    t = text.upper().strip()
    t = re.sub(r'\s+', '', t)
    # replace common OCR confusions conservatively
    t = t.replace('—', '-')  # em-dash -> dash
    # allow only A-Z, 0-9 and dash (we'll preserve dash)
    t = re.sub(r'[^A-Z0-9\-]', '', t)
    return t

def fix_batch_code(text: str) -> str:
    """Heuristic fixes for batch codes like BMS119F which were misread as BMS11IF or BMS11lF."""
    if not text:
        return text
    t = text.upper().replace(' ', '')
    # quick canonical replacements
    t = t.replace('L', '1')   # letter L -> digit 1 (common)
    # If there is a pipe or similar, map to 1
    t = t.replace('|', '1').replace('¡', '1')

    # If it looks like a BMS batch code pattern, apply stricter rules
    # Pattern: starts with BMS and ends with F (common in your images)
    if re.match(r'^BMS.*F$', t):
        # core between BMS and trailing F
        core = t[3:-1]
        # replace common misreads heuristically
        core = re.sub(r'11I', '119', core)
        core = re.sub(r'1I', '19', core)
        core = re.sub(r'I', '9', core)
        t = 'BMS' + core + 'F'
        t = re.sub(r'[^A-Z0-9\-]', '', t)
    else:
        # gentle replacement of 'I' -> '1' only when between digits
        t_list = list(t)
        for i, ch in enumerate(t_list):
            if ch == 'I':
                left = t_list[i-1] if i-1 >= 0 else ''
                right = t_list[i+1] if i+1 < len(t_list) else ''
                if left.isdigit() and right.isdigit():
                    t_list[i] = '1'
        t = ''.join(t_list)
        t = re.sub(r'[^A-Z0-9\-]', '', t)
    return t

# ----------------- Combined GUI Settings -----------------
LIVE_W, LIVE_H = 350, 240        # left live feed (small)
ANNOT_W, ANNOT_H = 350, 240      # middle annotated view (small)
TEXT_W, TEXT_H = 400, 480        # right text area (large)
PANEL_BG = (245, 245, 245)       # light gray background for text panel

# OCR history / counts
max_history = 500
ocr_history = deque(maxlen=max_history)   # recent entries (timestamp, original_text, final_text, conf)
counts = Counter()                        # counts per detected normalized text

# ----------------- Camera open -----------------
cap = cv2.VideoCapture(0)  # change index if needed
if not cap.isOpened():
    print("❌ Cannot open camera (index 0). Exiting.")
    sys.exit(1)

USE_IMSHOW = imshow_available()
if not USE_IMSHOW:
    print("⚠️ cv2.imshow unavailable — running in fallback/headless mode.")
else:
    print("✅ cv2.imshow available — displaying combined window.")

print("📷 Controls: Press 'p' to capture and decode, 'q' to quit.")
print("If in headless mode: type 'p' + Enter to capture, 'q' + Enter to quit.")

# stored annotated preview between captures
last_annotated_preview = np.full((ANNOT_H, ANNOT_W, 3), 200, dtype=np.uint8)  # blank initially

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        # small live preview (left)
        live_preview = cv2.resize(frame, (LIVE_W, LIVE_H), interpolation=cv2.INTER_AREA)

        # annotated_preview (center) - use last captured annotated (scaled)
        annotated_preview = last_annotated_preview.copy()

        # build text panel (right)
        text_panel = np.full((TEXT_H, TEXT_W, 3), PANEL_BG, dtype=np.uint8)
        header = "OCR Results (most recent at top)"
        cv2.putText(text_panel, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,10,10), 2)

        # show counts summary block
        cv2.putText(text_panel, "Counts:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1)
        y = 85
        for txt, cnt in counts.most_common(10):
            if y > TEXT_H - 60:
                break
            summary_line = f"{txt}  -> {cnt}"
            cv2.putText(text_panel, summary_line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50,50,50), 1)
            y += 22

        # Display history (most recent first)
        cv2.putText(text_panel, "History:", (10, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1)
        y_hist = y + 30
        for ts, orig, final, conf in list(ocr_history)[::-1]:
            if y_hist > TEXT_H - 10:
                break
            time_str = ts[-8:]
            line = f"{time_str} | {final} ({conf:.2f})"
            # low confidence highlight (simple): print small red circle to left if < 0.6
            if conf < 0.60:
                cv2.circle(text_panel, (8, y_hist - 6), 4, (0,0,255), -1)
            cv2.putText(text_panel, line, (30, y_hist), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60,60,60), 1)
            y_hist += 18

        # Combine left + center + right into a single canvas
        def fit_col(img, w, h):
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        col1 = fit_col(live_preview, LIVE_W, TEXT_H)
        col2 = fit_col(annotated_preview, ANNOT_W, TEXT_H)
        col3 = text_panel  # already TEXT_H

        combined = np.hstack((col1, col2, col3))

        # show combined
        if USE_IMSHOW:
            cv2.imshow("OCR - Live | Annotated | Text", combined)
            key = cv2.waitKey(1) & 0xFF
        else:
            preview_path = os.path.join(SAVE_DIR, "preview.jpg")
            cv2.imwrite(preview_path, combined)
            print("Type 'p' + Enter to capture, 'q' + Enter to quit (or wait): ", end="", flush=True)
            i, _, _ = select.select([sys.stdin], [], [], 2)
            if i:
                s = sys.stdin.readline().strip().lower()
                key = ord(s[0]) if s else None
            else:
                key = None

        if key == ord('p'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"🔍 Capturing frame at {timestamp}...")

            raw_filename = os.path.join(SAVE_DIR, f"frame_raw_{timestamp}.jpg")
            cv2.imwrite(raw_filename, frame)

            # preprocess and OCR
            thresh = preprocess_for_ocr(frame)

            # Use allowlist so EasyOCR focuses on alphanumeric + dash (reduces I vs 9 errors)
            try:
                results = reader.readtext(
                    thresh,
                    detail=1,
                    paragraph=False,
                    # allowlist parameter instructs EasyOCR to restrict possible characters
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                    contrast_ths=0.05,
                    adjust_contrast=0.7
                )
            except TypeError:
                # fallback to without allowlist
                results = reader.readtext(thresh, detail=1, paragraph=False, contrast_ths=0.05, adjust_contrast=0.7)

            annotated = frame.copy()
            if results:
                for bbox, text, conf in results:
                    try:
                        (tl, tr, br, bl) = bbox
                        x1, y1 = map(int, tl)
                        x2, y2 = map(int, br)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                        text_pos = (x1, max(10, y1-8))
                        cv2.putText(annotated, f"{text} ({conf:.2f})", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                        # Normalization and batch-specific correction
                        normalized = normalize_text_for_counting(text)
                        corrected = fix_batch_code(normalized)

                        # If fix_batch_code produced nothing (unlikely), fall back to normalized
                        final_text = corrected if corrected else normalized

                        # Update history and counts
                        ocr_history.append((timestamp, text, final_text, conf))
                        counts[final_text] += 1
                    except Exception:
                        continue
            else:
                ocr_history.append((timestamp, "", "No text", 0.0))

            annotated_filename = os.path.join(SAVE_DIR, f"frame_annotated_{timestamp}.jpg")
            cv2.imwrite(annotated_filename, annotated)

            # update the small annotated preview shown in the GUI center
            last_annotated_preview = cv2.resize(annotated, (ANNOT_W, ANNOT_H), interpolation=cv2.INTER_AREA)

            # Log to TXT and CSV
            try:
                with open(TXT_LOG, "a", encoding="utf-8") as ftxt, open(CSV_LOG, "a", newline="", encoding="utf-8") as fcsv:
                    csv_writer = csv.writer(fcsv)
                    ftxt.write(f"\n--- {timestamp} ---\n")
                    if results:
                        for bbox, text, conf in results:
                            try:
                                (tl, tr, br, bl) = bbox
                                x1, y1 = map(int, tl)
                                x2, y2 = map(int, br)
                            except Exception:
                                x1 = y1 = x2 = y2 = ""
                            normalized = normalize_text_for_counting(text)
                            corrected = fix_batch_code(normalized)
                            final_text = corrected if corrected else normalized
                            line = f"{final_text} (Confidence: {conf:.2f})  [raw:{text}]"
                            print("- " + line)
                            ftxt.write(line + "\n")
                            csv_writer.writerow([timestamp, os.path.abspath(raw_filename),
                                                 os.path.abspath(annotated_filename), final_text, f"{conf:.2f}", x1, y1, x2, y2])
                    else:
                        print("⚠️ No text detected.")
                        ftxt.write("No text detected.\n")
                        csv_writer.writerow([timestamp, os.path.abspath(raw_filename),
                                             os.path.abspath(annotated_filename), "No text", "0.00", "", "", "", ""])
                print(f"Saved raw: {raw_filename}")
                print(f"Saved annotated: {annotated_filename}")
                print(f"Logs -> TXT: {os.path.abspath(TXT_LOG)}, CSV: {os.path.abspath(CSV_LOG)}")
            except Exception as e:
                print("❌ Error writing logs:", e)
                traceback.print_exc()

        elif key == ord('q'):
            print("👋 Exiting...")
            break

finally:
    cap.release()
    if USE_IMSHOW:
        cv2.destroyAllWindows()
