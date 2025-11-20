# app.py
"""
OpenCV-free Streamlit OCR Financial Document Analyzer
- Uses Pillow + NumPy + pytesseract
- Better preprocessing and field extraction
- Per-field confidence estimates using pytesseract.image_to_data
- Optional PDF -> images via pdf2image (if installed)
- Accuracy metrics (CER, WER, field precision/recall/F1)
"""

import streamlit as st
from pathlib import Path
import tempfile
import zipfile
import json
import re
import io
import csv
from typing import Dict, Tuple, List, Any

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import pytesseract

# Optional PDF conversion
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except Exception:
    PDF2IMAGE_AVAILABLE = False

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="OCR Financial Document Analyzer (no OpenCV)", layout="wide")
st.title("📄 OCR Financial Document Analyzer — OpenCV-free")

# ---------------------------
# Sidebar settings
# ---------------------------
st.sidebar.header("OCR & Preprocessing Settings")
lang_input = st.sidebar.text_input("Tesseract languages (e.g. 'eng' or 'eng+tha')", "eng")
psm = st.sidebar.selectbox("Tesseract PSM (page segmentation)", [3,4,6,11,12], index=2)
use_osd = st.sidebar.checkbox("Use Tesseract OSD for rotation (deskew)", True)
upscale_factor = st.sidebar.slider("Upscale factor (1.0 = original)", min_value=1.0, max_value=2.0, value=1.5, step=0.1)
apply_median = st.sidebar.checkbox("Apply small median filter (denoise)", True)
apply_unsharp = st.sidebar.checkbox("Apply light unsharp mask (sharpen)", False)
apply_autocontrast = st.sidebar.checkbox("Apply autocontrast", True)
tess_extra = st.sidebar.text_input("Extra Tesseract config (optional)", "--oem 3")

st.sidebar.markdown("---")
st.sidebar.subheader("Processing & Output")
enable_text_area = st.sidebar.checkbox("Show OCR text area", True)
show_confidences = st.sidebar.checkbox("Show per-field confidence", True)
st.sidebar.markdown("---")

st.sidebar.subheader("Accuracy (optional)")
calc_accuracy = st.sidebar.checkbox("Enable accuracy metrics", value=False)
gt_json = st.sidebar.file_uploader("Upload ground-truth JSON (list of {file, fields})", type=["json"])
st.sidebar.markdown("Or enter ground-truth manually (applies to all processed files):")
gt_full_manual = st.sidebar.text_area("GT: full text", "", disabled=not calc_accuracy)
gt_invoice_manual = st.sidebar.text_input("GT: invoice_number", "", disabled=not calc_accuracy)
gt_date_manual = st.sidebar.text_input("GT: date", "", disabled=not calc_accuracy)
gt_total_manual = st.sidebar.text_input("GT: total", "", disabled=not calc_accuracy)
gt_vendor_manual = st.sidebar.text_input("GT: vendor", "", disabled=not calc_accuracy)

st.sidebar.markdown("---")
st.sidebar.info("For Streamlit Cloud: include packages.txt with 'tesseract-ocr' and 'poppler-utils' for PDF support.")

# ---------------------------
# Helpers: file handling
# ---------------------------
def extract_zip_to_temp(zip_bytes: bytes) -> List[Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_zip_"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(temp_dir)
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"):
        images.extend(sorted(temp_dir.glob(ext)))
    return images

def pdf_bytes_to_images(pdf_bytes: bytes) -> List[Path]:
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
# Preprocessing (PIL + NumPy)
# ---------------------------
def pil_to_np_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))

def np_to_pil_gray(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("L")

def detect_rotation_with_osd(pil_img: Image.Image) -> int:
    """Return rotation angle (0/90/180/270) using pytesseract OSD if available else 0"""
    try:
        osd = pytesseract.image_to_osd(pil_img)
        for line in osd.splitlines():
            if line.lower().startswith("rotate:"):
                angle = int(line.split(":")[1].strip())
                return angle % 360
    except Exception:
        pass
    return 0

def preprocess_image_for_ocr(pil_img: Image.Image,
                             upscale: float = 1.5,
                             median: bool = True,
                             unsharp: bool = False,
                             autocontrast_on: bool = True,
                             deskew: bool = True) -> Image.Image:
    """
    Gentle OCR-friendly preprocessing:
      - optional deskew via Tesseract OSD
      - upscaling (bicubic)
      - small median filter for denoising
      - optional light unsharp mask (use sparingly)
      - optional autocontrast (light)
    Keep image in grayscale (no aggressive binarization).
    """
    im = pil_img.convert("RGB")

    # Deskew (rotate) if requested
    if deskew:
        try:
            rot = detect_rotation_with_osd(im)
            if rot and rot != 0:
                # rotate in opposite direction to correct
                im = im.rotate(360 - rot, expand=True)
        except Exception:
            pass

    # Upscale
    if upscale and upscale != 1.0:
        new_w = max(1, int(im.width * upscale))
        new_h = max(1, int(im.height * upscale))
        im = im.resize((new_w, new_h), Image.BICUBIC)

    # Convert to gray for filters
    gray = im.convert("L")

    if median:
        try:
            gray = gray.filter(ImageFilter.MedianFilter(size=3))
        except Exception:
            pass

    if unsharp:
        try:
            gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        except Exception:
            pass

    if autocontrast_on:
        try:
            gray = ImageOps.autocontrast(gray, cutoff=1)
        except Exception:
            pass

    # return grayscale PIL image (no thresholding)
    return gray

# ---------------------------
# OCR helpers (pytesseract)
# ---------------------------
def ocr_get_text_and_data(pil_img: Image.Image, lang: str, psm_mode: int, extra_config: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (full_text, data_rows) where data_rows is the list returned by image_to_data (dicts)
    Each row has keys: level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text
    """
    cfg = f"{extra_config} --psm {psm_mode}"
    # Use image_to_data for tokens + confidences
    try:
        data = pytesseract.image_to_data(pil_img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        # build list of rows
        rows = []
        n = len(data['text'])
        for i in range(n):
            rows.append({
                'level': data['level'][i],
                'page_num': data['page_num'][i],
                'block_num': data['block_num'][i],
                'par_num': data['par_num'][i],
                'line_num': data['line_num'][i],
                'word_num': data['word_num'][i],
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else -1.0,
                'text': data['text'][i] or ''
            })
        # Reconstruct full text (preserve newlines roughly)
        full_text = pytesseract.image_to_string(pil_img, lang=lang, config=cfg)
        return full_text, rows
    except Exception:
        # fallback to simple string call
        txt = pytesseract.image_to_string(pil_img, lang=lang)
        return txt, []

def token_confidence_for_phrase(phrase: str, data_rows: List[Dict[str, Any]]) -> float:
    """
    Estimate confidence for a phrase by matching tokens from OCR data.
    Returns average confidence for matched tokens (0..1). If no match, returns 0.
    """
    if not phrase:
        return 0.0
    tokens = re.findall(r"\w+[\w\-\./]*", phrase.lower())
    if not tokens:
        return 0.0
    # Build OCR token list
    ocr_tokens = []
    for r in data_rows:
        t = r.get('text', '').strip()
        if not t:
            continue
        ocr_tokens.append((t.lower(), r.get('conf', -1.0)))
    if not ocr_tokens:
        return 0.0
    matched_confs = []
    # For each phrase token, find best matching OCR token (exact or startswith)
    for tok in tokens:
        best_conf = None
        for otok, conf in ocr_tokens:
            if otok == tok or otok.startswith(tok) or tok.startswith(otok):
                if conf >= 0:
                    if best_conf is None or conf > best_conf:
                        best_conf = conf
        if best_conf is not None:
            matched_confs.append(best_conf)
    if not matched_confs:
        return 0.0
    # convert conf (0..100) to 0..1 average
    avg = sum(matched_confs) / len(matched_confs)
    return max(0.0, min(1.0, avg / 100.0))

# ---------------------------
# Field extraction (regex)
# ---------------------------
DATE_PATTERN = r"(\b\d{1,2}\s*[-\/\s]?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[A-Za-z]*\s*[-\/\s]?\s*\d{2,4}\b|\b\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}\b|\b\d{4}[-\/]\d{2}[-\/]\d{2}\b|\b[A-Za-z]+\.?\s+\d{1,2},\s*\d{4}\b)"
INVOICE_PATTERN = r"(?:Invoice|Invoice No\.?|Invoice #|INVOICE ID|INVOICE|INV)[\s:]*([A-Za-z0-9\/\-\_\.\:]+)"
TOTAL_PATTERN = r"(?:(?:Total|TOTAL|Total Due|Amount Due|Grand Total|TOTAL:?)\s*[:\-\s]*\$?\s*([0-9\.,]+))"
VENDOR_PATTERN = r"^(.*?)\s*(?=(?:INVOICE|Invoice|Invoice No|Invoice #|INVOICE ID|INV))"

def extract_fields_from_text(text: str) -> Dict[str, str]:
    out = {"invoice_number":"", "date":"", "total":"", "vendor":""}
    if not text:
        return out
    txt = text.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = txt.strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    joined = "\n".join(lines)

    # invoice
    m = re.search(INVOICE_PATTERN, joined, flags=re.IGNORECASE)
    if m:
        out["invoice_number"] = m.group(1).strip()

    # date
    m = re.search(DATE_PATTERN, joined, flags=re.IGNORECASE)
    if m:
        out["date"] = m.group(1).strip()

    # total (try top and bottom)
    m = re.search(TOTAL_PATTERN, joined, flags=re.IGNORECASE)
    if not m:
        for l in reversed(lines[-10:]):
            mm = re.search(TOTAL_PATTERN, l, flags=re.IGNORECASE)
            if mm:
                m = mm
                break
    if m:
        out["total"] = m.group(1).strip().rstrip(".,$")

    # vendor: try text before invoice label, else first non-empty line
    mv = re.search(VENDOR_PATTERN, joined, flags=re.IGNORECASE | re.DOTALL)
    if mv:
        vendor_candidate = mv.group(1).strip()
        vendor_candidate = re.sub(r"^[A-Za-z0-9\)\.\-]{1,6}\s*", "", vendor_candidate)
        vendor_candidate = vendor_candidate.splitlines()[0].strip()
        out["vendor"] = vendor_candidate
    elif lines:
        out["vendor"] = lines[0]

    # normalize total
    if isinstance(out["total"], str):
        out["total"] = out["total"].replace(",", "")
    return out

# ---------------------------
# Accuracy metrics
# ---------------------------
def levenshtein(a: str, b: str) -> int:
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def CER(pred: str, gt: str) -> float:
    pred = "" if pred is None else str(pred)
    gt = "" if gt is None else str(gt)
    if len(gt) == 0:
        return 0.0
    return levenshtein(pred, gt) / len(gt)

def WER(pred: str, gt: str) -> float:
    pred_words = str(pred).strip().split()
    gt_words = str(gt).strip().split()
    if len(gt_words) == 0:
        return 0.0
    return levenshtein(pred_words, gt_words) / len(gt_words)

def field_precision_recall_f1(pred_fields: Dict[str,str], gt_fields: Dict[str,str]) -> Tuple[float,float,float]:
    tp = fp = fn = 0
    for key in gt_fields:
        gt = (gt_fields.get(key) or "").strip()
        if gt == "":
            continue
        pred = (pred_fields.get(key) or "").strip()
        if pred == gt:
            tp += 1
        else:
            if pred:
                fp += 1
            fn += 1
    precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
    recall = tp / (tp+fn) if (tp+fn)>0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    return precision, recall, f1

# ---------------------------
# UI: uploads & processing
# ---------------------------
uploaded = st.file_uploader("Upload images / PDFs / ZIP (images inside)", type=["png","jpg","jpeg","tiff","bmp","zip","pdf"], accept_multiple_files=True)
local_dir = st.text_input("Local directory (server only) - optional", "")
process_btn = st.button("Process")

if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = []

def append_result(x):
    st.session_state["ocr_results"].append(x)

if process_btn:
    st.session_state["ocr_results"] = []
    paths: List[Path] = []

    # local dir listing
    if local_dir:
        p = Path(local_dir)
        if p.exists() and p.is_dir():
            for ext in ("*.png","*.jpg","*.jpeg","*.tiff","*.bmp"):
                paths.extend(sorted(p.glob(ext)))

    # uploaded files
    for f in uploaded or []:
        name = f.name.lower()
        raw = f.read()
        if name.endswith(".zip"):
            imgs = extract_zip_to_temp(raw)
            paths.extend(imgs)
        elif name.endswith(".pdf"):
            if PDF2IMAGE_AVAILABLE:
                imgs = pdf_bytes_to_images(raw)
                paths.extend(imgs)
            else:
                st.warning("pdf2image not available - skipping PDF. Install pdf2image & poppler for PDF support.")
        else:
            tmp = Path(tempfile.mkdtemp()) / f.name
            tmp.write_bytes(raw)
            paths.append(tmp)

    if not paths:
        st.warning("No images found to process.")
    else:
        progress = st.progress(0)
        total = len(paths)
        for idx, p in enumerate(paths, start=1):
            try:
                raw_bytes = p.read_bytes() if isinstance(p, Path) else open(p, "rb").read()
                pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                processed = preprocess_image_for_ocr(pil, upscale=upscale_factor, median=apply_median, unsharp=apply_unsharp, autocontrast_on=apply_autocontrast, deskew=use_osd)
                # run OCR
                full_text, data_rows = ocr_get_text_and_data(processed, lang_input, psm, tess_extra)
                fields = extract_fields_from_text(full_text)
                # compute confidences per field
                confidences = {}
                if data_rows:
                    for k,v in fields.items():
                        confidences[k] = token_confidence_for_phrase(v, data_rows)
                else:
                    confidences = {k: 0.0 for k in fields}
                entry = {
                    "file": p.name if isinstance(p, Path) else str(p),
                    "text": full_text,
                    "fields": fields,
                    "field_confidence": confidences,
                    "raw_rows": data_rows
                }
                append_result(entry)
            except Exception as e:
                append_result({"file": str(p), "error": str(e)})
            progress.progress(idx/total)
        st.success("Processing complete.")

# ---------------------------
# Display results
# ---------------------------
if st.session_state["ocr_results"]:
    st.header("Results")
    for item in st.session_state["ocr_results"]:
        st.subheader(item.get("file", "unknown"))
        if "error" in item:
            st.error(item["error"])
            continue
        cols = st.columns([2,1])
        with cols[0]:
            st.markdown("**OCR Text**")
            if enable_text_area:
                st.text_area("ocr_text", item.get("text",""), height=220)
            else:
                st.write(item.get("text",""))
        with cols[1]:
            st.markdown("**Extracted Fields**")
            st.json(item.get("fields", {}))
            if show_confidences:
                st.markdown("**Field confidences (0..1)**")
                st.json(item.get("field_confidence", {}))

    # export JSON
    export_json = json.dumps(st.session_state["ocr_results"], indent=2, ensure_ascii=False)
    st.download_button("Download results (JSON)", data=export_json, file_name="ocr_results.json", mime="application/json")

# ---------------------------
# Accuracy evaluation
# ---------------------------
def parse_gt_json_file(f) -> Dict[str, Dict[str,str]]:
    try:
        data = json.load(f)
        # expecting list of {"file": "name", "fields": {...}, "full_text": "..."} or similar
        result = {}
        for it in data:
            fname = it.get("file")
            fields = it.get("fields", {})
            full = it.get("full_text", "")
            if fname:
                result[fname] = {"fields": fields, "full_text": full}
        return result
    except Exception:
        return {}

if calc_accuracy and st.session_state["ocr_results"]:
    st.header("Accuracy Evaluation")
    gt_map = {}
    if gt_json:
        gt_map = parse_gt_json_file(gt_json)
    rows = []
    for item in st.session_state["ocr_results"]:
        if "error" in item:
            continue
        fname = item.get("file")
        pred_text = item.get("text", "")
        pred_fields = item.get("fields", {})
        if fname in gt_map:
            gt_fields = gt_map[fname].get("fields", {})
            gt_full = gt_map[fname].get("full_text", "")
        else:
            gt_fields = {"invoice_number": gt_invoice_manual, "date": gt_date_manual, "total": gt_total_manual, "vendor": gt_vendor_manual}
            gt_full = gt_full_manual
        cer = CER(pred_text, gt_full)
        wer = WER(pred_text, gt_full)
        precision, recall, f1 = field_precision_recall_f1(pred_fields, gt_fields)
        rows.append({
            "file": fname,
            "CER": round(cer,4),
            "WER": round(wer,4),
            "Precision": round(precision,4),
            "Recall": round(recall,4),
            "F1": round(f1,4)
        })
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download accuracy CSV", df.to_csv(index=False), file_name="accuracy.csv", mime="text/csv")
    else:
        st.info("No ground truth provided or no results to evaluate.")

st.markdown("---")
st.caption("Notes: This app uses pytesseract + Pillow. For PDFs install pdf2image and add poppler-utils to packages.txt on Streamlit Cloud. Place runtime.txt='python-3.10' in repo root.")
