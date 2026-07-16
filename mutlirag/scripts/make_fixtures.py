"""Generate test fixtures: a sample Excel and a MIXED PDF (text + an image
whose words only OCR can recover). Run from the project root."""
import os

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

os.makedirs("samples", exist_ok=True)

# --- 1. Sample Excel ---------------------------------------------------------
df = pd.DataFrame(
    [
        {"Employee": "Alice Chen", "Dept": "Engineering", "City": "Austin", "Salary": 145000},
        {"Employee": "Bob Ortiz", "Dept": "Sales", "City": "Denver", "Salary": 98000},
        {"Employee": "Priya Nair", "Dept": "Engineering", "City": "Seattle", "Salary": 158000},
        {"Employee": "Sam Lee", "Dept": "Marketing", "City": "Austin", "Salary": 87000},
    ]
)
df.to_excel("samples/employees.xlsx", index=False, sheet_name="Staff")
print("wrote samples/employees.xlsx")

# --- 2. An image containing text (simulates a chart/screenshot in a doc) -----
img = Image.new("RGB", (900, 260), "white")
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 54)
except Exception:
    font = ImageFont.load_default(size=54)
draw.text((40, 40), "ACCESS CODE: ZEBRA-42", fill="black", font=font)
draw.text((40, 140), "Valid until 2026-12-31", fill="black", font=font)
img.save("samples/code.png")
print("wrote samples/code.png")

# --- 3. Mixed PDF: typed paragraph + the image above -------------------------
doc = fitz.open()
page = doc.new_page()
page.insert_text(
    (72, 90),
    "Quarterly Report Q3 2026\n"
    "Revenue grew by 20 percent compared to Q2.\n"
    "The building access code is printed in the image below.",
    fontsize=13,
)
page.insert_image(fitz.Rect(72, 180, 522, 310), filename="samples/code.png")
doc.save("samples/mixed_report.pdf")
doc.close()
print("wrote samples/mixed_report.pdf")
print("DONE")
