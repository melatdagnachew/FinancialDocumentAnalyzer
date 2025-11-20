
# app.py
"""
Streamlit OCR Extraction App

Features:
- Upload image(s) or ZIP of images
- Read from a local directory (for deployment)
- Preprocess images (gray, blur, threshold, optional deskew)
- Extract text using Tesseract OCR
- Extract structured fields using regex (Invoice No, Date, Total, Vendor)
- Download OCR results as JSON
"""

import streamlit as st
from pathlib import Path
import tempfile
import zipfile
import json
import re
import io
import cv2
import numpy as np
from PIL import Image
import pytesseract


# ---------------------------
# Utility: Load Image
# ---------------------------
def load_image_from_bytes(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        pil = Image.open(io.BytesIO(b)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return img


# ---------------------------
# Utility: Preprocessing
# ---------------------------
def preprocess_image(img: np.ndarray, deskew: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if deskew:
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return th


# ---------------------------
# Utility: OCR
# ---------------------------
def ocr_image(img: np.ndarray, lang: str, tess_config: str) -> str:
    pil = Image.fromarray(img)
    text = pytesseract.image_to_string(pil, lang=lang, config=tess_config)
    return text


# ---------------------------
# Utility: Regex Extraction
# ---------------------------
def extract_fields(text: str):
    patterns = {
        "invoice_number": r"(?:Invoice|Invoice No\.?|Invoice #|Inv)[:\-\s]*([A-Za-z0-9\-/]+)",
        "date": r"([0-3]?\d[\/\-][0-1]?\d[\/\-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},\s*\d{4})",
        "total": r"(?:Total|Grand Total|Amount Due)[:\-\s]*\$?([0-9\.,]+)",
        "vendor": r"(?:Vendor|From)[:\-\s]*(.+)"
    }

    cleaned = "\n".join([l.strip() for l in text.splitlines() if l.strip()])
    results = {}

    for key, pat in patterns.items():
        m = re.search(pat, cleaned, re.IGNORECASE)
        results[key] = m.group(1).strip() if m else ""

    if not results.get("vendor"):
        lines = cleaned.split("\n")
        results["vendor"] = lines[0] if lines else ""

    return results


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="OCR Engine", layout="wide")
st.title("📄 OCR Extraction System")

st.sidebar.header("Settings")
lang = st.sidebar.text_input("Language", "eng")
psm = st.sidebar.selectbox("PSM Mode", [3, 4, 6, 11, 12], index=2)
deskew = st.sidebar.checkbox("Deskew Image", False)
show_raw = st.sidebar.checkbox("Show OCR Text", True)

tess_config = f"--psm {psm}"

uploaded_files = st.file_uploader(
    "Upload image(s) or ZIP",
    type=["png", "jpg", "jpeg", "tiff", "bmp", "zip"],
    accept_multiple_files=True
)

local_dir = st.text_input("Local directory (optional for server)")
process_btn = st.button("Process")

results = []


# ---------------------------
# Process Uploaded Files
# ---------------------------
def handle_upload(f):
    temp_dir = Path(tempfile.mkdtemp())
    file_path = temp_dir / f.name

    with open(file_path, "wb") as w:
        w.write(f.getvalue())

    if file_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(temp_dir / "unzipped")

        return [
            p for p in (temp_dir / "unzipped").rglob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        ]

    return [file_path]


# ---------------------------
# Main Processing
# ---------------------------
if process_btn:
    input_paths = []

    if local_dir:
        p = Path(local_dir)
        if p.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
                input_paths.extend(p.glob(ext))

    for f in uploaded_files or []:
        input_paths.extend(handle_upload(f))

    if not input_paths:
        st.error("No images found!")
    else:
        st.info(f"Processing {len(input_paths)} file(s)...")

        for path in input_paths:
            try:
                with open(path, "rb") as img_file:
                    img = load_image_from_bytes(img_file.read())

                processed = preprocess_image(img, deskew)
                text = ocr_image(processed, lang, tess_config)
                fields = extract_fields(text)

                results.append({
                    "file": str(path),
                    "text": text,
                    "fields": fields
                })

            except Exception as e:
                results.append({"file": str(path), "error": str(e)})

        st.success("Processing Completed")

        for r in results:
            st.subheader(r["file"])
            if "error" in r:
                st.error(r["error"])
                continue

            if show_raw:
                st.text_area("OCR Text", r["text"], height=150)

            st.json(r["fields"])

        export = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button("Download Results", export, "ocr_results.json", "application/json")


# ---------------------------
# Accuracy Metrics
# ---------------------------

def levenshtein(a, b):
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]


def CER(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    if len(gt) == 0:
        return 0
    return levenshtein(pred, gt) / len(gt)


def WER(pred, gt):
    pred_words = pred.strip().split()
    gt_words = gt.strip().split()
    if len(gt_words) == 0:
        return 0
    return levenshtein(pred_words, gt_words) / len(gt_words)


def field_accuracy(pred_fields, gt_fields):
    total_fields = 0
    correct = 0

    for key in gt_fields:
        total_fields += 1
        if key in pred_fields and pred_fields[key].strip() == gt_fields[key].strip():
            correct += 1

    return correct / total_fields if total_fields > 0 else 0


# --- Evaluate Accuracy ---
if "calculate_accuracy" in locals() and calculate_accuracy and results:
    gt_fields = {
        "invoice_number": gt_invoice,
        "date": gt_date,
        "total": gt_total,
        "vendor": gt_vendor,
    }

    st.subheader("🔍 Accuracy Results")

    for r in results:
        if "error" in r:
            continue

        pred_text = r["text"]
        pred_fields = r["fields"]

        cer = CER(pred_text, gt_text)
        wer = WER(pred_text, gt_text)
        fa = field_accuracy(pred_fields, gt_fields)

        st.write(f"### File: {r['file']}")
        st.write(f"- **CER:** {cer:.4f}")
        st.write(f"- **WER:** {wer:.4f}")
        st.write(f"- **Field Accuracy:** {fa*100:.2f}%")
        st.write("---")
