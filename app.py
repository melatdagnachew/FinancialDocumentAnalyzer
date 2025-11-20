# app.py
"""
Streamlit OCR Extraction App - Fixed & Improved Version

Features:
- Upload image(s), PDF(s) or ZIP of images
- Robust preprocessing (deskew, CLAHE, denoise, sharpen, morphology, upscaling)
- OCR via pytesseract (default) with improved config; optional PaddleOCR fallback if installed
- Improved regex extraction for invoice_number, date, total, vendor
- Accuracy evaluation: CER, WER, Precision/Recall/F1, Jaccard, normalized Levenshtein, partial matches
- Save/download JSON results
- Deployable on Streamlit Cloud (requires packages.txt containing `tesseract-ocr`)
"""

import streamlit as st
from pathlib import Path
import tempfile
import zipfile
import json
import re
import io
import os
import math
import csv
import traceback

import cv2
import numpy as np
from PIL import Image

# OCR engines
import pytesseract

# Optional heavy dependencies: use if installed
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    paddle_ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
except Exception:
    PADDLE_AVAILABLE = False

# Optional pdf support (only used if installed)
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="OCR Financial Document Analyzer", layout="wide")
st.title("📄 Financial Document OCR & Field Extractor (Fixed)")

# ---------------------------
# Sidebar settings
# ---------------------------
st.sidebar.header("Settings")
ocr_engine_choice = st.sidebar.selectbox("OCR Engine", ("pytesseract", "paddleocr (if installed)"))
lang = st.sidebar.text_input("Tesseract language", "eng")
psm = st.sidebar.selectbox("Tesseract PSM", [3, 4, 6, 11, 12], index=2)
use_deskew = st.sidebar.checkbox("Apply deskew", value=True)
use_clahe = st.sidebar.checkbox("Apply CLAHE (contrast)", value=True)
use_super_res = st.sidebar.checkbox("Upscale before OCR (x2)", value=False)
tess_extra_config = st.sidebar.text_input("Extra tesseract config (optional)", "--oem 3")
st.sidebar.markdown("---")

# Accuracy controls
st.sidebar.subheader("Accuracy (optional)")
calculate_accuracy = st.sidebar.checkbox("Enable accuracy metrics", value=False)
gt_json_upload = st.sidebar.file_uploader("Upload ground-truth JSON (optional)", type=["json"])
st.sidebar.markdown("Or enter ground-truth manually for the file currently processed.")
gt_text_manual = st.sidebar.text_area("GT: Full text (manual)", value="", height=80, disabled=not calculate_accuracy)
gt_invoice_manual = st.sidebar.text_input("GT: Invoice number", value="", disabled=not calculate_accuracy)
gt_date_manual = st.sidebar.text_input("GT: Date", value="", disabled=not calculate_accuracy)
gt_total_manual = st.sidebar.text_input("GT: Total amount", value="", disabled=not calculate_accuracy)
gt_vendor_manual = st.sidebar.text_input("GT: Vendor", value="", disabled=not calculate_accuracy)

st.sidebar.markdown("---")
st.sidebar.write("Notes: For Streamlit Cloud, include `packages.txt` with `tesseract-ocr`.")


# ---------------------------
# Helper: file handlers
# ---------------------------
def extract_zip_to_temp(zip_bytes):
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_zip_"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(temp_dir)
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"):
        images.extend(sorted(temp_dir.glob(ext)))
    return images


def pdf_bytes_to_images(pdf_bytes):
    """Convert pdf bytes to PIL images using pdf2image if available."""
    if not PDF2IMAGE_AVAILABLE:
        return []
    pil_pages = convert_from_bytes(pdf_bytes)
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
    paths = []
    for i, page in enumerate(pil_pages):
        p = temp_dir / f"page_{i+1}.png"
        page.save(p, "PNG")
        paths.append(p)
    return paths


# ---------------------------
# Image loading + preprocessing
# ---------------------------
def load_image_from_bytes(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(b)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def upscale_image(img, fx=2, fy=2):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)


def denoise_gray(gray, h=10):
    return cv2.fastNlMeansDenoising(gray, h=h)


def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def deskew_image(gray):
    # compute angle of rotation and rotate
    coords = np.column_stack(np.where(gray < 255))
    if coords.shape[0] == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_image_for_ocr(img_bgr, deskew=True, clahe=True, upscale=False):
    img = img_bgr.copy()
    if upscale:
        img = upscale_image(img, fx=2, fy=2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # contrast
    if clahe:
        try:
            gray = apply_clahe(gray)
        except Exception:
            pass

    # denoise and sharpen
    gray = denoise_gray(gray, h=10)
    gray = sharpen_image(gray)

    # deskew
    if deskew:
        try:
            gray = deskew_image(gray)
        except Exception:
            pass

    # morphology closing to connect broken characters
    try:
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    except Exception:
        pass

    # final adaptive threshold (both options)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 8
    )

    # return both original-like (gray) and binarized for tesseract display
    return gray, th


# ---------------------------
# OCR functions
# ---------------------------
def ocr_with_pytesseract(image_np, lang="eng", config="--oem 3 --psm 6"):
    pil = Image.fromarray(image_np)
    try:
        text = pytesseract.image_to_string(pil, lang=lang, config=config)
    except Exception as e:
        # fallback: try without config
        text = pytesseract.image_to_string(pil, lang=lang)
    return text


def ocr_with_paddle(image_bgr):
    # expects BGR or RGB? paddle accepts numpy image in RGB; ensure conversion
    try:
        # paddle returns list of lines; join them
        results = paddle_ocr_engine.ocr(image_bgr, cls=True)
        lines = []
        for line in results:
            if isinstance(line, list) and len(line) >= 2:
                lines.append(line[1][0])
            else:
                # fallback, try to stringify
                lines.append(str(line))
        return "\n".join(lines)
    except Exception:
        return ""


def run_ocr(img_bgr):
    # Preprocess
    gray, th = preprocess_image_for_ocr(img_bgr, deskew=use_deskew, clahe=use_clahe, upscale=use_super_res)

    # Prefer Paddle if selected & available
    if ocr_engine_choice == "paddleocr (if installed)" and PADDLE_AVAILABLE:
        text = ocr_with_paddle(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if text and text.strip():
            return text, gray, th

    # else pytesseract on binarized image (th)
    tess_cfg = f"{tess_extra_config} --psm {psm}"
    text = ocr_with_pytesseract(th, lang=lang, config=tess_cfg)
    if not text.strip() and PADDLE_AVAILABLE:
        # fallback to paddle if tesseract returned nothing
        text = ocr_with_paddle(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return text, gray, th


# ---------------------------
# Field extraction (improved regex)
# ---------------------------
DATE_PATTERN = r"(\b\d{1,2}\s*[-~\/]?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[A-Za-z]*\s*[-~\/]?\s*\d{2,4}\b|\b[A-Za-z]+\.?\s+\d{1,2},\s*\d{4}\b|\b\d{4}[-\/]\d{2}[-\/]\d{2}\b|\b[0-3]?\d[-\/][0-1]?\d[-\/]\d{2,4}\b)"
INVOICE_PAT = r"(?:Invoice|Invoice No\.?|Invoice #|INVOICE ID|INVOICE)\s*[:#\-\s]*([A-Za-z0-9\/\-\_]+)"
TOTAL_PAT = r"(?:(?:Total|TOTAL|Total Due|Amount Due|Grand Total)\s*[:\-\s]*\$?\s*([0-9\.,]+))"
VENDOR_PAT = r"^(.*?)\s*(?=(?:INVOICE|Invoice|Invoice No|Invoice #|INVOICE ID))"  # vendor before invoice label


def extract_fields_from_text(text):
    """
    Returns dict with fields: invoice_number, date, total, vendor
    """
    res = {"invoice_number": "", "date": "", "total": "", "vendor": ""}

    if not text or not text.strip():
        return res

    cleaned = text.replace("\r", "\n")
    # Quick normalize spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # unify some OCR artifacts
    cleaned = cleaned.replace("O9AAB", "09AAB").replace("O9AABCS", "09AABCS")
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]

    joined = "\n".join(lines)

    # invoice number
    m_inv = re.search(INVOICE_PAT, joined, flags=re.IGNORECASE)
    if m_inv:
        res["invoice_number"] = m_inv.group(1).strip()

    # date
    m_date = re.search(DATE_PATTERN, joined, flags=re.IGNORECASE)
    if m_date:
        res["date"] = m_date.group(1).strip()

    # total - search reversed lines as well (sometimes "TOTAL" at bottom)
    m_total = re.search(TOTAL_PAT, joined, flags=re.IGNORECASE)
    if not m_total:
        # try lines bottom-up
        for l in reversed(lines[-8:]):
            m = re.search(TOTAL_PAT, l, flags=re.IGNORECASE)
            if m:
                m_total = m
                break
    if m_total:
        res["total"] = m_total.group(1).strip().rstrip(".,$")

    # vendor - try match before invoice label
    m_vendor = re.search(VENDOR_PAT, joined, flags=re.IGNORECASE | re.DOTALL)
    if m_vendor:
        vendor_candidate = m_vendor.group(1).strip()
        # clean common headers or leading enumerators like "A)" or "g)"
        vendor_candidate = re.sub(r"^[A-Za-z0-9\)\.\-]{1,4}\s*", "", vendor_candidate)
        # keep first line or two
        vendor_candidate = vendor_candidate.splitlines()[0]
        res["vendor"] = vendor_candidate.strip()

    # fallbacks: vendor = first non-empty line if still empty
    if not res["vendor"] and lines:
        res["vendor"] = lines[0]

    # normalize some globals (numbers with comma separators)
    res["total"] = res["total"].replace(",", "") if isinstance(res["total"], str) else res["total"]

    return res


# ---------------------------
# Accuracy metrics
# ---------------------------
def levenshtein(a, b):
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [list(range(m + 1))]
    dp.extend([[i + 1] + [0] * m for i in range(n)])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def CER(pred, gt):
    pred = "" if pred is None else str(pred).strip()
    gt = "" if gt is None else str(gt).strip()
    if len(gt) == 0:
        return 0.0
    return levenshtein(pred, gt) / len(gt)


def WER(pred, gt):
    pred_words = str(pred).strip().split()
    gt_words = str(gt).strip().split()
    if len(gt_words) == 0:
        return 0.0
    return levenshtein(pred_words, gt_words) / len(gt_words)


def field_precision_recall_f1(pred_fields, gt_fields):
    tp = 0
    fp = 0
    fn = 0
    # only consider keys present in ground truth as the set to evaluate
    for key in gt_fields:
        gt_val = (gt_fields.get(key) or "").strip()
        pred_val = (pred_fields.get(key) or "").strip()
        if gt_val == "":
            continue
        if pred_val == gt_val:
            tp += 1
        else:
            # predicted but wrong => FP (only if non-empty)
            if pred_val:
                fp += 1
            # missing => FN
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def jaccard_similarity(pred, gt):
    pred_tokens = set(str(pred).lower().split())
    gt_tokens = set(str(gt).lower().split())
    if not gt_tokens:
        return 0.0
    inter = pred_tokens.intersection(gt_tokens)
    union = pred_tokens.union(gt_tokens)
    return len(inter) / len(union)


def normalized_levenshtein(pred, gt):
    pred = str(pred)
    gt = str(gt)
    if len(pred) == 0 and len(gt) == 0:
        return 1.0
    dist = levenshtein(pred, gt)
    return 1.0 - (dist / max(len(pred), len(gt)))


def partial_match_score(pred, gt):
    pred_l = str(pred).lower()
    gt_l = str(gt).lower()
    if gt_l in pred_l and gt_l != "":
        return 1.0
    return jaccard_similarity(pred, gt)


# ---------------------------
# UI - uploads & processing
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload image(s), PDF(s) or ZIP (images inside)",
    type=["png", "jpg", "jpeg", "tiff", "bmp", "zip", "pdf"],
    accept_multiple_files=True
)

local_dir = st.text_input("Local directory (optional; server only)")
process_btn = st.button("Process files")

# session results container
if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = []

# helper to append result
def append_result(entry):
    st.session_state["ocr_results"].append(entry)

# processing routine
if process_btn:
    st.session_state["ocr_results"] = []
    input_paths = []

    # handle local dir if provided (server deployments)
    if local_dir:
        p = Path(local_dir)
        if p.exists() and p.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"):
                input_paths.extend(sorted(p.glob(ext)))

    # handle uploaded files
    for f in uploaded_files or []:
        fname = f.name.lower()
        try:
            raw = f.read()
            if fname.endswith(".zip"):
                imgs = extract_zip_to_temp(raw)
                input_paths.extend(imgs)
            elif fname.endswith(".pdf"):
                if PDF2IMAGE_AVAILABLE:
                    imgs = pdf_bytes_to_images(raw)
                    input_paths.extend(imgs)
                else:
                    st.warning("pdf2image not installed: skipping PDF conversion. Install pdf2image & poppler for PDF support.")
            else:
                # write bytes to temp file
                tmp = Path(tempfile.mkdtemp()) / f.name
                tmp.write_bytes(raw)
                input_paths.append(tmp)
        except Exception as e:
            st.error(f"Failed to read uploaded file {f.name}: {e}")
            continue

    if not input_paths:
        st.warning("No images found to process.")
    else:
        progress_bar = st.progress(0)
        total = len(input_paths)
        for i, path in enumerate(input_paths, start=1):
            try:
                b = path.read_bytes() if isinstance(path, Path) else open(path, "rb").read()
                img = load_image_from_bytes(b)
                ocr_text, gray_vis, bin_vis = run_ocr(img)
                fields = extract_fields_from_text(ocr_text)

                entry = {
                    "file": str(path.name),
                    "raw_text": ocr_text,
                    "fields": fields
                }
                append_result(entry)
            except Exception as e:
                append_result({"file": str(path), "error": str(e)})
            progress_bar.progress(i / total)
        st.success("Processing complete.")

# display results
if st.session_state["ocr_results"]:
    st.header("Results")
    cols = st.columns([2, 2, 2, 2])
    # Show each result
    for r in st.session_state["ocr_results"]:
        st.subheader(r.get("file", "unknown"))
        if "error" in r:
            st.error(r["error"])
            continue
        # two-column layout: text and fields
        left, right = st.columns([2, 1])
        with left:
            st.markdown("**OCR text**")
            st.text_area("OCR output", r["raw_text"], height=180)
        with right:
            st.markdown("**Extracted fields**")
            st.json(r["fields"])

    # Download JSON
    export_json = json.dumps(st.session_state["ocr_results"], indent=2, ensure_ascii=False)
    st.download_button("Download results (JSON)", data=export_json, file_name="ocr_results.json", mime="application/json")

# ---------------------------
# Accuracy evaluation (UI + compute)
# ---------------------------
def parse_gt_json_file(f):
    try:
        data = json.load(f)
        # Expecting list of {"file": "name", "fields": {...}}
        return {item["file"]: item["fields"] for item in data}
    except Exception:
        return {}

if calculate_accuracy and st.session_state["ocr_results"]:
    st.header("Accuracy Evaluation")

    # ground truth dictionary from upload (highest priority)
    gt_map = {}
    if gt_json_upload:
        gt_map = parse_gt_json_file(gt_json_upload)

    # if manual GT provided and single file processed, use those
    # compute evaluations per-file
    rows = []
    for r in st.session_state["ocr_results"]:
        if "error" in r:
            continue
        fname = r["file"]
        pred_text = r["raw_text"]
        pred_fields = r["fields"]

        # determine GT for this file
        gt_fields = {}
        gt_text = ""
        if fname in gt_map:
            gt_fields = gt_map[fname]
            gt_text = gt_map[fname].get("full_text", "")
        else:
            # use manual GT fields if provided (applies to all files)
            gt_fields = {
                "invoice_number": gt_invoice_manual,
                "date": gt_date_manual,
                "total": gt_total_manual,
                "vendor": gt_vendor_manual
            }
            gt_text = gt_text_manual

        # compute metrics
        cer = CER(pred_text, gt_text)
        wer = WER(pred_text, gt_text)
        field_acc = field_precision_recall_f1(pred_fields, gt_fields)
        precision, recall, f1 = field_acc

        # aggregate token/char similarities for display
        vendor_jaccard = jaccard_similarity(pred_fields.get("vendor", ""), gt_fields.get("vendor", ""))
        invoice_jaccard = jaccard_similarity(pred_fields.get("invoice_number", ""), gt_fields.get("invoice_number", ""))
        total_norm_lev = normalized_levenshtein(pred_fields.get("total", ""), gt_fields.get("total", ""))

        rows.append({
            "file": fname,
            "CER": round(cer, 4),
            "WER": round(wer, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "vendor_jaccard": round(vendor_jaccard, 4),
            "invoice_jaccard": round(invoice_jaccard, 4),
            "total_norm_lev": round(total_norm_lev, 4)
        })

    # display table
    if rows:
        st.subheader("Per-file metrics")
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # summary
        summary = {
            "CER_mean": sum(r["CER"] for r in rows) / len(rows),
            "WER_mean": sum(r["WER"] for r in rows) / len(rows),
            "Precision_mean": sum(r["Precision"] for r in rows) / len(rows),
            "Recall_mean": sum(r["Recall"] for r in rows) / len(rows),
            "F1_mean": sum(r["F1"] for r in rows) / len(rows),
        }
        st.metric("Mean CER", f"{summary['CER_mean']:.4f}")
        st.metric("Mean WER", f"{summary['WER_mean']:.4f}")
        st.metric("Mean F1", f"{summary['F1_mean']:.4f}")

        # download CSV
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=df.columns)
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download accuracy CSV", data=csv_buf.getvalue(), file_name="accuracy_report.csv", mime="text/csv")
    else:
        st.info("No ground truth provided or no rows to compute metrics.")

# Footer notes
st.markdown("---")
st.caption("Notes: For reliable deployment on Streamlit Cloud, include `packages.txt` with line: `tesseract-ocr`. "
           "Optional improvements: install PaddleOCR + pdf2image + poppler for PDF support and better OCR results.")

