import cv2
import easyocr
import datetime
import os
import sys
import csv
import traceback
import select
# ----------------- Configuration -----------------
SAVE_DIR = "captured_frames"
TXT_LOG = "ocr_log.txt"
CSV_LOG = "ocr_log.csv"
os.makedirs(SAVE_DIR, exist_ok=True)
# Initialize EasyOCR reader (this may print device info)
reader = easyocr.Reader(['en'])
# Prepare CSV header if not exists
if not os.path.exists(CSV_LOG):
   with open(CSV_LOG, "w", newline="", encoding="utf-8") as cf:
       writer = csv.writer(cf)
       writer.writerow(["timestamp", "raw_frame", "annotated_frame", "text", "confidence", "x1", "y1", "x2", "y2"])
# ----------------- Helpers -----------------
def imshow_available():
   """Try to check if cv2.imshow likely works in this environment."""
   if not hasattr(cv2, "imshow"):
       return False
   try:
       # Try to create and destroy a window quickly
       cv2.namedWindow("test_window", cv2.WINDOW_NORMAL)
       cv2.destroyWindow("test_window")
       return True
   except Exception:
       return False
def preprocess_for_ocr(img_bgr):
   """Return thresholded image optimized for OCR on reflective metal surfaces."""
   gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
   # CLAHE (contrast limited adaptive hist equalization)
   clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
   enhanced = clahe.apply(gray)
   # Optional denoising (uncomment if noisy)
   # enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
   # Adaptive threshold to handle uneven lighting
   thresh = cv2.adaptiveThreshold(
       enhanced, 255,
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
# ----------------- Main loop -----------------
cap = cv2.VideoCapture(0)  # change index if needed
if not cap.isOpened():
   print("❌ Cannot open camera (index 0). Exiting.")
   sys.exit(1)
USE_IMSHOW = imshow_available()
if not USE_IMSHOW:
   print("⚠️ cv2.imshow unavailable — running in fallback/headless mode.")
else:
   print("✅ cv2.imshow available — showing live feed window.")
print("📷 Controls: Press 'p' to capture and decode, 'q' to quit.")
print("If in headless mode: type 'p' + Enter to capture, 'q' + Enter to quit.")
try:
   while True:
       ret, frame = cap.read()
       if not ret:
           print("⚠️ Failed to grab frame.")
           break
       if USE_IMSHOW:
           cv2.imshow("Live Feed - Press 'p' to capture", frame)
           key = cv2.waitKey(1) & 0xFF
       else:
           # In headless mode we periodically prompt the user for action (2s timeout)
           # Save a light preview to inspect if needed (not every loop to avoid spam)
           preview_path = os.path.join(SAVE_DIR, "preview.jpg")
           tmp_preview = frame.copy()
           cv2.putText(tmp_preview, "Headless mode: type 'p'+Enter to capture or wait", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
           cv2.imwrite(preview_path, tmp_preview)
           print("Type 'p' + Enter to capture, 'q' + Enter to quit (or wait to continue): ", end="", flush=True)
           i, _, _ = select.select([sys.stdin], [], [], 2)  # 2 second timeout
           if i:
               s = sys.stdin.readline().strip().lower()
               key = ord(s[0]) if s else None
           else:
               key = None
       if key == ord('p'):
           timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
           print(f"🔍 Capturing frame at {timestamp}...")
           # Save raw frame
           raw_filename = os.path.join(SAVE_DIR, f"frame_raw_{timestamp}.jpg")
           cv2.imwrite(raw_filename, frame)
           # Preprocess
           thresh = preprocess_for_ocr(frame)
           # Run OCR (tweak parameters for better detection)
           # paragraph=False keeps individual boxes; paragraph=True groups into lines (choose as needed)
           results = reader.readtext(thresh,
                                     detail=1,
                                     paragraph=False,   # change to True if you prefer grouped lines
                                     contrast_ths=0.05,
                                     adjust_contrast=0.7)
           # Annotate
           annotated = frame.copy()
           if results:
               for (bbox, text, confidence) in results:
                   try:
                       (top_left, top_right, bottom_right, bottom_left) = bbox
                       x1, y1 = map(int, top_left)
                       x2, y2 = map(int, bottom_right)
                       cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                       # Put text above the box if there's room
                       text_pos = (x1, max(y1 - 10, 10))
                       cv2.putText(annotated, f"{text} ({confidence:.2f})", text_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                   except Exception:
                       # If bbox format unexpected, skip drawing for that box
                       continue
           annotated_filename = os.path.join(SAVE_DIR, f"frame_annotated_{timestamp}.jpg")
           cv2.imwrite(annotated_filename, annotated)
           # Try to show annotated image if GUI available
           if USE_IMSHOW:
               # small display window to show result briefly
               cv2.imshow("Annotated", annotated)
           else:
               # Attempt to auto-open on OS (Windows will open by default)
               safe_open_file(annotated_filename)
           # Logging: TXT + CSV with bbox coords
           try:
               with open(TXT_LOG, "a", encoding="utf-8") as ftxt, \
                    open(CSV_LOG, "a", newline="", encoding="utf-8") as fcsv:
                   csv_writer = csv.writer(fcsv)
                   ftxt.write(f"\n--- {timestamp} ---\n")
                   if results:
                       for (bbox, text, confidence) in results:
                           try:
                               (top_left, top_right, bottom_right, bottom_left) = bbox
                               x1, y1 = map(int, top_left)
                               x2, y2 = map(int, bottom_right)
                           except Exception:
                               x1 = y1 = x2 = y2 = ""
                           line = f"{text} (Confidence: {confidence:.2f})"
                           print(f"- {line}")
                           ftxt.write(line + "\n")
                           csv_writer.writerow([timestamp, os.path.abspath(raw_filename),
                                                os.path.abspath(annotated_filename),
                                                text, f"{confidence:.2f}", x1, y1, x2, y2])
                   else:
                       print("⚠️ No text detected.")
                       ftxt.write("No text detected.\n")
                       csv_writer.writerow([timestamp, os.path.abspath(raw_filename),
                                            os.path.abspath(annotated_filename),
                                            "No text", "0.00", "", "", "", ""])
               print(f"Saved raw: {os.path.abspath(raw_filename)}")
               print(f"Saved annotated: {os.path.abspath(annotated_filename)}")
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
 