Financial Document Analyzer (OCR-Based Invoice Extraction)

This project is an AI-powered financial document analyzer that automatically reads invoices, extracts key financial fields, and generates structured JSON output. It is designed to demonstrate how OCR + rule-based NLP can automate document-heavy workflows in finance, lending, and accounting.

The system accepts images, PDFs, or ZIP files containing multiple documents. It preprocesses each file, performs OCR, extracts structured fields such as invoice number, date, vendor, and total amount, and optionally computes accuracy metrics when ground truth is available.

A deployed Streamlit demo is available, and the source code is included in this repository.

🌐 Live Demo

👉 Streamlit Cloud App: https://financialdocumentanalyzer-y7b4edpqvgmawqcnlff4he.streamlit.app/

🔍 Project Objective

The goal of this project is to show how AI can automate a document-intensive process—such as invoice or financial statement reading—by converting unstructured documents into structured, machine-readable data.

This contributes to AI-driven operational efficiency by enabling:

automated invoice entry

faster loan application processing

reduced human error

cleaner financial datasets for downstream analytics

🚀 Features

pytesseract (OCR engine)

Pillow (PIL) for image processing

numpy for array operations

pdf2image / pypdfium2 for PDF conversion (optional)


2. Advanced Preprocessing Pipeline

Improves OCR accuracy using only PIL + NumPy:

Grayscale conversion

Deskewing using pytesseract.image_to_osd

Upscaling (x2)

Median filtering (noise reduction)

Unsharp masking (clarity)

Autocontrast correction

Otsu binarization implemented manually in NumPy

These steps significantly improved text clarity and extraction accuracy.

3. Financial Field Extraction

Extracted fields include:

invoice_number

date

total amount

vendor name

Extraction is powered by improved regex patterns designed specifically for invoice formats.

4. Confidence Scoring (0–1)

Each extracted field receives a confidence score based on:

Jaccard similarity

Normalized Levenshtein

Partial match scoring

5. Accuracy Metrics (When Ground Truth Is Provided)

Metrics calculated:

CER (Character Error Rate)

WER (Word Error Rate)

Precision, Recall, F1 (field-level)

Vendor & Invoice Jaccard similarity

Normalized Levenshtein accuracy

A CSV export of accuracy is generated.

6. Structured JSON Output

For images in images file 

images/Template9_Instance145.jpg
images/Template9_Instance149.jpg
images/Template9_Instance153.jpg
images/Template9_Instance156.jpg
images/Template9_Instance157.jpg
images/Template9_Instance181.jpg
images/Template9_Instance183.jpg
images/Template9_Instance184.jpg
images/Template9_Instance185.jpg
images/Template9_Instance187.jpg

OUTPUT is 

Template9_Instance145.jpg
OCR Text
g) Navarro, Ford and Bryan INVOICE # 3499-396
Date: 26-Jun-2017

GSTINIUIN: 0O9AABCS142961ZS

Email:rclark@example.org
Buyer :Tracey Mccoy
19600 Greene Meadows Suite 417
Bennettton, MN 11563 US
Tel:+(254)782-4473
Email:erikjohnson@example.net
Site:http:/iwells.info/
Qty | Description | Unit Price | Amount
3.00 _ Thousand join general, i 90.54 i 271.62 |
6.00 _ Police team, | 81.97 | 491.82 |
4,00 Memory old, $1.81 i 207,24
Total in words: nine hundred and sevent-
y-four point seven four

TOTAL : 974.74 USD

Note:
This order is shipped through blue dart courier


Extracted Fields

{
"invoice_number":"3499-396"
"date":"26-Jun-2017"
"total":"974.74"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.96
"date":0.88
"total":0.96
"vendor":0.945
}
Template9_Instance149.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan INVOICE # INV/99-05/097
Invoice Date: 28-Feb-2007
GSTINIUIN: 09AABCS142961ZS
Email:rclark@example.org
Buyer :Brittany Christian
43007 Mary Inlet Suite 831
West Michael, MO 02672 US
Tel:+(340)979-7397
Email:colonsharon@example.com
Site:http:/www.young-kemp.com/
Qty |Description [Unit Price | Amount
2.00 _ Change much, i 5.43 | 10.86
3.00 _Rule million sign, | 15.87 | 47.61
4.00 _ Job. | 87.60 } 350.40 {
2.00 _My read. 70.49 i 140.98 |
5,00 Race bank scene, 55.52 i 277.60
Total in words: eight hundred and fift-
y-one point eight three
TOTAL : 851.83 EUR
Note: All payments to be made in cash.
Contact us for queries on these quotations.

Extracted Fields

{
"invoice_number":"INV/99-05/097"
"date":"99-05/097"
"total":"851.83"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.95
"date":0
"total":0.96
"vendor":0.96
}
Template9_Instance153.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan INVOICE # 7855-356
Invoice Date: 06-Dec-2006
GSTINIUIN: 0O9AABCS142961ZS
Email:rclark@example.org
Bill to:Bradley Holmes
159 Taylor Corners Apt. 106
Adamsfort, MP 63383 US
Tel:+(488)730-6529
Email:welchaaron@example.org
Site:http://www jacobs.biz/
Qty | Description [Unit Price | Amount —_|
5.00 _ Senior car animal, | $4.53 27265
6,00 _Writer out. | 64.54 | 387.24 |
1,00 _Or this Mrs, | 68.78 | 68.78 {
3.00 _ Agency machine federal. i 12.38 | 37.14 |
2.00 Sing recent. i 32.83 i 65.66
Total in words: eight hundred and for-
ty-six point four nine
TOTAL : 846.49 $
Note:
This order is shipped through blue dart courier

Extracted Fields

{
"invoice_number":"7855-356"
"date":"06-Dec-2006"
"total":"846.49"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.96
"date":0.96
"total":0.91
"vendor":0.945
}
Template9_Instance156.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan INVOICE # INV/79-24/803
Date: 30-Apr-2014
GSTINIUIN: 0O9AABCS142961ZS
Email:rclark@example.org
Buyer :Patricia Alexander
43363 Michael Rapids
South Kimberty, MA 53078 US
Tel:+(659)680-2491
Email:tkennedy@example.org
Site:https://carter-kerr.com/
Qty | Description [Unit Price | Amount —_|
5.00 Center, | 12.88 | 64.40 |
5,00 _ Phone sport lead, | 28,15 } 140.75 1
1,00 _Room great end, | 15.12 | 15.12 {
1,00 _ Guy per. | 44.09 i 44.09 |
4.00 Language move look. I 25.76 i 103.04
Total in words: three hundred and six-
ty-one point nine one
TOTAL : 361.91 USD
Note: All payments to be made in cash.
Contact us for queries on these quotations.

Extracted Fields

{
"invoice_number":"INV/79-24/803"
"date":"79-24/803"
"total":"361.91"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.96
"date":0
"total":0.96
"vendor":0.945
}
Template9_Instance157.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan INVOICE # 2Y8MS5d-200
Invoice Date: 17-Dec-2020
GSTINIUIN: 0O9AABCS142961ZS
Email:rclark@example.org
Bill to:Raven Garcia
186 Watson Junction Suite 063
Tylerfurt, AR 44759 US
Tel:+(376)874-1014
Email:tanyacrawford@example.com
Site:http://bates-scott.com/
Qty |Description | Unit Price | Amount _|
2.00 _ Morning major before. | 25.53 | 51,06 |
3,00 Believe measure, 25.24 | 75.72
Total in words: one hundred and twent-
y-eight point four one
TOTAL : 128.41 $
Note:
This order is shipped through blue dart courier

Extracted Fields

{
"invoice_number":"2Y8MS5d-200"
"date":"17-Dec-2020"
"total":"128.41"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.48
"date":0.95
"total":0.44
"vendor":0.9425
}
Template9_Instance183.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan INVOICE # 2Y5M4d-247
Date: 09-Oct-1993

GSTINIUIN: 0O9AABCS142961ZS

Email:rclark@example.org
Bill to:Jeremy Zhang
19608 Peterson Mountain
Alaniand, NH 25201 US
Tel:+(335)470-9744
Email:williamsjulie@example.net
Site:https://www.scott-newman.com/
Qty | Description | Unit Price | Amount —_|
3,00 _ Our price. | 38.36 } 115.08 |
4,00 _ Believe indeed may. i 58.10 | 232.40 |
2.00 Sound bad fund attack. i 75.88 i 151.76
Total in words: five hundred and ni-
ne point four seven

TOTAL : 509.47 $

Note:Total payment due in 14 days.

Extracted Fields

{
"invoice_number":"2Y5M4d-247"
"date":"09-Oct-1993"
"total":"509.47"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.7
"date":0.88
"total":0.81
"vendor":0.955
}
Template9_Instance184.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan Invoice number INV/79-80/930
Date: 25-Jun-2001

GSTINIUIN: 0O9AABCS142961ZS

Email:rclark@example.org
Bill to:Ryan Walker
9325 Angela Rest
East Christopherberg, WY 10638 US
Tel:+(531)858-3269
Email:kristincortez@example.com
Site:http://young.com/
Qty | Description | Unit Price | Amount
1,00 _ Daughter wait whose, i 77.05 } 77.05 |
5.00 _ Return scene run, | 42.87 | 214.35 {
6.00 Forward between war, 37.90 i 227.40
Total in words: five hundred and th-
irty point eight one

TOTAL : 530.81 USD

Note: All payments to be made in cash.
Contact us for queries on these quotations.

Extracted Fields

{
"invoice_number":"number"
"date":"79-80/930"
"total":"530.81"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.93
"date":0
"total":0.96
"vendor":0.96
}
Template9_Instance185.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan INVOICE # 5028-040
Date: 26-Jul-2009
GSTINIUIN: 0O9AABCS142961ZS
Email:rclark@example.org
Bill to:Daniel Johnson
11564 Harris Shores
South Territown, IL 11705 US
Tel:+(015)199-0037
Email:morrowjoel@example.net
Site:https://haynes.com/
Qty | Description [| UnitPrice | Amount |
4.00 _ Win art gun specific, | 47,77 | 191,08 |
3.00 _ Film visit each, | 61.17 i 183.51 |
4,00 _Lead around, { 55.86 i 223.44 {
5.00 _ Thousand need nature. | 85.54 | 427.70
1,00 Challenge firm, i 41.66 i 41.66
Total in words: one thousand and ninet-
y-nine point three five
TOTAL : 1099.35 USD
Note:Total payment due in 14 days.

Extracted Fields

{
"invoice_number":"5028-040"
"date":"26-Jul-2009"
"total":"1099.35"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.96
"date":0.7
"total":0.96
"vendor":0.9425
}
Template9_Instance187.jpg
OCR Text

ocr_text

g) Navarro, Ford and Bryan INVOICE # 5636-040
Invoice Date: 30-Dec-1999
GSTINIUIN: 0O9AABCS142961ZS
Email:rclark@example.org
Buyer :Andrea Ali
37315 Mark Square Suite 719
New Mary, AL 09017 US
Tel:+(284)657-9224
Email:bridgetmorton@example.org
Site:http:/www.hunter.com/
Qty | Description [unit price | Amount —_|
2.00 _Call everybody agreement, i 6.94 it 13.88 |
4.00 _ Discuss, | 21,23 | 84.92 {
6.00 _ Card race executive. { 89.70 | $38.20
5.00 Federal have. i 32.45 i 162.25
Total in words: eight hundred and twen-
ty-four point eight six
TOTAL : 824.86 USD
Note:
This order is shipped through blue dart courier

Extracted Fields

{
"invoice_number":"5636-040"
"date":"30-Dec-1999"
"total":"824.86"
"vendor":"Navarro, Ford and Bryan"
}
Field confidences (0..1)

{
"invoice_number":0.96
"date":0.95
"total":0.96
"vendor":0.945
}



🧪 Methodology
Step 1 — Input Handling

Single images

Multi-page PDFs (converted to PNG per page)

ZIP files containing multiple images

Server-side directory loading option

Step 2 — Image Preprocessing

Enhances text clarity before OCR.

Step 3 — OCR Extraction

Performed using pytesseract.image_to_string.

Step 4 — Field Extraction

Regex patterns customized for financial invoices.

Step 5 — Accuracy Computation

Ground truth can be provided as:

A JSON file mapping each filename to the correct fields, or

Manual sidebar text entry.

Step 6 — Output Generation

Streamlit displays results and allows JSON/CSV download.

🧰 Libraries Used
Category	Libraries
OCR	pytesseract
Preprocessing	Pillow (PIL), numpy
PDF Support	pdf2image, pypdfium2
Web App	Streamlit
Accuracy Metrics	numpy, Python regex
Deployment	Streamlit Cloud
📦 Deployment Files

To run on Streamlit Cloud, the following are included:

requirements.txt

Contains all Python dependencies (OpenCV-free).

packages.txt

Ensures system-level dependencies:

tesseract-ocr
poppler-utils

runtime.txt
python-3.10


This guarantees compatibility with pytesseract and numpy.

⚙️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py


Make sure Tesseract OCR is installed on your system.

🔒 Limitations

While the system performs well, challenges remain:

Low-resolution or blurry documents reduce OCR accuracy.

Non-English invoices require additional language models.

Handwritten invoices are not supported.

Very irregular invoice formats may extract incorrect fields.

A human should review low-confidence fields (<0.75) before approval.

🏢 Business Impact

By automating the extraction of financial data:

Back-office teams can reduce manual data entry time.

Hundreds of hours per month can be saved in invoice-heavy organizations.

Human errors in manual transcription are significantly reduced.

This demonstrates how AI bridges unstructured documents → structured databases, enabling faster operations and more accurate analytics.

📂 Project Structure
📁 FinancialDocumentAnalyzer
 ├── app.py
 ├── app.ipynb
 ├── requirements.txt
 ├── packages.txt
 ├── runtime.txt
 ├── README.md

