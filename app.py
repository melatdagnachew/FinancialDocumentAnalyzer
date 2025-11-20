# app.py
"""
Streamlit OCR Extraction App - OpenCV-free & improved

Key points:
- No OpenCV. Uses Pillow + NumPy + pytesseract.
- Preprocessing: upscale, denoise (median), unsharp mask, autocontrast, Otsu binarization.
- Uses pytesseract OSD to detect rotation for deskewing (when available).
- Improved regex extraction and accuracy metrics (CER, WER, Precision/Recall/F1, Jaccard, norm-lev).
- Optional PDF conversion via pdf2image (if installed). No PaddleOCR dependency.
- For Streamlit Cloud deployment, ensure:
    - runtime.txt -> "python-3.10"
    - packages.txt -> contains "tesseract-ocr" and, if using PDFs, "poppler-utils"
    - requirements.txt -> include: streamlit, pytesseract, pillow, numpy, pandas, pdf2image (optional), pypdfium2 (optional)
"""

import streamlit as st
from pathlib import Path
import tempfile
import zipfile
import json
import re
import io
import os
import csv
from typing import Tuple, Dict

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
# Streamlit layout & sidebar
# ---------------------------
st.set_page_config(page_title="OCR Financial Document Analyzer (no OpenCV)", layout="wide")
st.title("📄 Financial Document OCR & Extractor (OpenCV-free)")

st.sidebar.header("Settings")
lang = st.sidebar.text_input("Tesseract language", "eng")
psm = st.sidebar.selectbox("Tesseract PSM (page segmentation)", [3, 4, 6, 11, 12], index=2)
enable_deskew = st.sidebar.checkbox("Deskew using Tesseract OSD", value=True)
upscale = st.sidebar.checkbox("Upscale (x2) before OCR", value=False)
autocontrast = st.sidebar.checkbox("Autocontrast", True)
sharpen = st.sidebar.checkbox("Sharpen (Unsharp Mask)", True)
median_filter = st.sidebar.checkbox("Apply median filter (denoise)", True)
tess_extra = st.sidebar.text_input("Extra tesseract config", "--oem 3")

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
st.sidebar.info("For Streamlit Cloud: include packages.txt with 'tesseract-ocr' (and 'poppler-utils' for pdf support).")
# ---------------------------
# Helpers: file handling
# ---------------------------
def extract_zip_to_temp(zip_bytes: bytes):
    temp_dir = Path(tempfile.mkdtemp(prefix="ocr_zip_"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(temp_dir)
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"):
        images.extend(sorted(temp_dir.glob(ext)))
    return images

def pdf_bytes_to_images(pdf_bytes: bytes):
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
def pil_to_numpy_gray(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("L"))

def numpy_to_pil_gray(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("L")

def otsu_threshold(arr: np.ndarray) -> int:
    # Otsu threshold for grayscale numpy array
    pixel_counts, bin_edges = np.histogram(arr.flatten(), bins=256, range=(0,256))
    total = arr.size
    sum_total = (np.arange(256) * pixel_counts).sum()
    sum_back = 0
    weight_back = 0
    max_var = 0
    threshold = 0
    for i in range(256):
        weight_back += pixel_counts[i]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += i * pixel_counts[i]
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        between_var = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        if between_var > max_var:
            max_var = between_var
            threshold = i
    return threshold

def preprocess_pil(img: Image.Image, do_upscale=False, do_clahe=False, do_median=True, do_sharpen=True, do_autocontrast=True, do_deskew=True) -> Image.Image:
    # img: PIL.Image
    im = img.convert("RGB")
    if do_upscale:
        im = im.resize((im.width * 2, im.height * 2), Image.BICUBIC)

    # Deskew via Tesseract OSD if requested
    if do_deskew:
        try:
            osd = pytesseract.image_to_osd(im)
            rot = 0
            for line in osd.splitlines():
                if line.strip().lower().startswith("rotate:"):
                    rot = int(line.split(":")[1].strip())
                    break
            if rot != 0:
                im = im.rotate(360 - rot, expand=True)
        except Exception:
            # image_to_osd may fail on some tesseract builds; ignore gracefully
            pass

    gray = im.convert("L")

    if do_median:
        try:
            gray = gray.filter(ImageFilter.MedianFilter(size=3))
        except Exception:
            pass

    if do_sharpen:
        try:
            # UnsharpMask provides better control than simple filter
            gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        except Exception:
            pass

    if do_autocontrast:
        try:
            gray = ImageOps.autocontrast(gray, cutoff=1)
        except Exception:
            pass

    # convert to numpy to apply Otsu thresholding
    arr = np.array(gray)
    t = otsu_threshold(arr)
    bin_arr = (arr > t).astype(np.uint8) * 255
    bin_img = Image.fromarray(bin_arr).convert("L")
    return bin_img

# ---------------------------
# OCR wrappers
# ---------------------------
def ocr_image_pytesseract(pil_image: Image.Image, lang="eng", psm_mode=6, extra_config="--oem 3"):
    # ensure PIL image in correct mode
    if pil_image.mode != "L" and pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    cfg = f"{extra_config} --psm {psm_mode}"
    try:
        txt = pytesseract.image_to_string(pil_image, lang=lang, config=cfg)
    except Exception:
        # fallback: call without config
        txt = pytesseract.image_to_string(pil_image, lang=lang)
    return txt

# ---------------------------
# Field extraction regex (improved)
# ---------------------------
DATE_PATTERN = r"(\b\d{1,2}\s*[-\/\s]?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[A-Za-z]*\s*[-\/\s]?\s*\d{2,4}\b|\b\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}\b|\b\d{4}[-\/]\d{2}[-\/]\d{2}\b|\b[A-Za-z]+\.?\s+\d{1,2},\s*\d{4}\b)"
INVOICE_PATTERN = r"(?:Invoice|Invoice No\.?|Invoice #|INVOICE ID|INVOICE|INV)[\s:]*([A-Za-z0-9\/\-\_\.]+)"
TOTAL_PATTERN = r"(?:(?:Total|TOTAL|Total Due|Amount Due|Grand Total|TOTAL:?)\s*[:\-\s]*\$?\s*([0-9\.,]+))"
VENDOR_PATTERN = r"^(.*?)\s*(?=(?:INVOICE|Invoice|Invoice No|Invoice #|INVOICE ID|INV))"

def extract_fields(text: str) -> Dict[str,str]:
    res = {"invoice_number":"", "date":"", "total":"", "vendor":""}
    if not text:
        return res
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # attempt to normalize common OCR mistakes
    cleaned = cleaned.replace("O9AAB", "09AAB").replace("GSTINAIIN", "GSTIN/UIN")
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    joined = "\n".join(lines)

    # invoice
    m = re.search(INVOICE_PATTERN, joined, flags=re.IGNORECASE)
    if m:
        res["invoice_number"] = m.group(1).strip()

    # date
    m = re.search(DATE_PATTERN, joined, flags=re.IGNORECASE)
    if m:
        res["date"] = m.group(1).strip()

    # total (try bottom lines too)
    m = re.search(TOTAL_PATTERN, joined, flags=re.IGNORECASE)
    if not m:
        for l in reversed(lines[-8:]):
            mm = re.search(TOTAL_PATTERN, l, flags=re.IGNORECASE)
            if mm:
                m = mm
                break
    if m:
        res["total"] = m.group(1).strip().rstrip(".,$")

    # vendor: try match before invoice label, else first line
    mv = re.search(VENDOR_PATTERN, joined, flags=re.IGNORECASE | re.DOTALL)
    if mv:
        vendor_candidate = mv.group(1).strip()
        vendor_candidate = re.sub(r"^[A-Za-z0-9\)\.\-]{1,4}\s*", "", vendor_candidate)
        vendor_candidate = vendor_candidate.splitlines()[0]
        res["vendor"] = vendor_candidate.strip()
    elif lines:
        res["vendor"] = lines[0]

    # normalize numbers
    if isinstance(res["total"], str):
        res["total"] = res["total"].replace(",", "")
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

def jaccard_similarity(pred, gt):
    pred_set = set(str(pred).lower().split())
    gt_set = set(str(gt).lower().split())
    if not gt_set:
        return 0.0
    inter = pred_set.intersection(gt_set)
    union = pred_set.union(gt_set)
    return len(inter)/len(union)

def normalized_levenshtein(pred, gt):
    pred = str(pred)
    gt = str(gt)
    if pred=="" and gt=="":
        return 1.0
    d = levenshtein(pred, gt)
    return 1.0 - (d / max(len(pred), len(gt)))

def partial_match_score(pred, gt):
    pl = str(pred).lower()
    gl = str(gt).lower()
    if gl != "" and gl in pl:
        return 1.0
    return jaccard_similarity(pred, gt)

# ---------------------------
# UI: uploads and processing
# ---------------------------
uploaded_files = st.file_uploader("Upload images / PDFs / ZIP (images inside)", type=["png","jpg","jpeg","tiff","bmp","zip","pdf"], accept_multiple_files=True)
local_dir = st.text_input("Local directory (server only)", "")
process = st.button("Process")

if "results" not in st.session_state:
    st.session_state["results"] = []

def append_result(r):
    st.session_state["results"].append(r)

if process:
    st.session_state["results"] = []
    paths = []

    # local dir option
    if local_dir:
        p = Path(local_dir)
        if p.exists() and p.is_dir():
            for ext in ("*.png","*.jpg","*.jpeg","*.tiff","*.bmp"):
                paths.extend(sorted(p.glob(ext)))

    # uploaded file handling
    for f in uploaded_files or []:
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
                st.warning("pdf2image not available - skipping PDF. Install pdf2image/poppler for PDF support.")
        else:
            tmp = Path(tempfile.mkdtemp()) / f.name
            tmp.write_bytes(raw)
            paths.append(tmp)

    if not paths:
        st.warning("No images found.")
    else:
        prog = st.progress(0)
        total = len(paths)
        for i, p in enumerate(paths, 1):
            try:
                raw = p.read_bytes() if isinstance(p, Path) else open(p, "rb").read()
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                processed = preprocess_pil(pil, do_upscale=upscale, do_clahe=False, do_median=median_filter, do_sharpen=sharpen, do_autocontrast=autocontrast, do_deskew=enable_deskew)
                text = ocr_image_pytesseract(processed, lang=lang, psm_mode=psm, extra_config=tess_extra)
                fields = extract_fields(text)
                append_result({"file": p.name if isinstance(p, Path) else str(p), "text": text, "fields": fields})
            except Exception as e:
                append_result({"file": str(p), "error": str(e)})
            prog.progress(i/total)
        st.success("Processing finished.")

# show results
if st.session_state["results"]:
    st.header("Results")
    for r in st.session_state["results"]:
        st.subheader(r.get("file","unknown"))
        if "error" in r:
            st.error(r["error"])
            continue
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("**OCR Text**")
            st.text_area("ocr", r["text"], height=180)
        with c2:
            st.markdown("**Extracted fields**")
            st.json(r["fields"])

    st.download_button("Download results (JSON)", json.dumps(st.session_state["results"], indent=2, ensure_ascii=False), file_name="ocr_results.json", mime="application/json")

# ---------------------------
# Accuracy evaluation
# ---------------------------
def parse_gt_json(f):
    try:
        data = json.load(f)
        return {item["file"]: item["fields"] for item in data}
    except Exception:
        return {}

if calc_accuracy and st.session_state["results"]:
    st.header("Accuracy Evaluation")
    gt_map = {}
    if gt_json:
        gt_map = parse_gt_json(gt_json)

    rows = []
    for r in st.session_state["results"]:
        if "error" in r:
            continue
        fname = r["file"]
        pred_text = r["text"]
        pred_fields = r["fields"]
        if fname in gt_map:
            gt_fields = gt_map[fname]
            gt_full = gt_map[fname].get("full_text","")
        else:
            gt_fields = {"invoice_number": gt_invoice_manual, "date": gt_date_manual, "total": gt_total_manual, "vendor": gt_vendor_manual}
            gt_full = gt_full_manual

        cer = CER(pred_text, gt_full)
        wer = WER(pred_text, gt_full)
        precision, recall, f1 = field_precision_recall_f1(pred_fields, gt_fields)
        vendor_j = jaccard_similarity(pred_fields.get("vendor",""), gt_fields.get("vendor",""))
        inv_j = jaccard_similarity(pred_fields.get("invoice_number",""), gt_fields.get("invoice_number",""))
        total_norm = normalized_levenshtein(pred_fields.get("total",""), gt_fields.get("total",""))

        rows.append({"file":fname, "CER":round(cer,4), "WER":round(wer,4), "Precision":round(precision,4), "Recall":round(recall,4), "F1":round(f1,4), "vendor_jaccard":round(vendor_j,4), "invoice_jaccard":round(inv_j,4), "total_norm_lev":round(total_norm,4)})

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download accuracy CSV", df.to_csv(index=False), file_name="accuracy.csv", mime="text/csv")
    else:
        st.info("No ground truth provided. Provide GT JSON or manual GT in the sidebar.")

st.markdown("---")
st.caption("Notes: This App uses pytesseract + PIL preprocessing (no OpenCV). For Streamlit Cloud deployment add packages.txt: 'tesseract-ocr' and 'poppler-utils' (if PDF). Use runtime.txt='python-3.10' and requirements.txt including: streamlit, pytesseract, pillow, numpy, pandas, pdf2image (optional).")
