import os
import io
from flask import Flask, render_template, request, send_file, redirect, url_for
from ai_detector import check_plagiarism, check_ai_generated
from stylometry import stylometry_verdict
from dataset_loader import load_reference_texts
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import nltk

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
nltk.download('punkt')
nltk.download('punkt_tab')



# ✅ Load reference texts ONCE at startup
REFERENCE_TEXTS = load_reference_texts()
last_result = None  # Stores last detection results



def read_file_content_upload(file_storage):
    """Read .txt or .docx file from uploaded FileStorage safely."""
    filename = file_storage.filename.lower()
    content = ""
    try:
        if filename.endswith(".txt"):
            content = file_storage.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".docx"):
            tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], file_storage.filename)
            file_storage.seek(0)
            file_storage.save(tmp_path)
            doc = Document(tmp_path)
            content = "\n".join([p.text for p in doc.paragraphs])
            try:
                os.remove(tmp_path)  # cleanup
            except Exception:
                pass
    except Exception as e:
        print(f"⚠ Error reading uploaded file: {e}")
    return content


@app.route("/", methods=["GET", "POST"])
def index():
    global last_result
    result = None

    if request.method == "POST":
        # Get textarea or uploaded file
        text_area = request.form.get("text", "").strip()
        uploaded = request.files.get("file")

        input_text = ""
        if uploaded and uploaded.filename:
            input_text = read_file_content_upload(uploaded)
        else:
            input_text = text_area

        if not input_text:
            return render_template("index.html", error="⚠ Please paste text or upload a file.")

        # ✅ Perform checks
        plagiarism_result = check_plagiarism(input_text, REFERENCE_TEXTS, threshold=0.3) or {}
        ai_result = check_ai_generated(input_text) or "No verdict"
        style_result = stylometry_verdict(input_text) or "No verdict"

        # ✅ Safe structure for template
        matches = plagiarism_result.get("matches", [])
        safe_matches = []
        for m in matches:
            safe_matches.append({
                "similarity": m.get("similarity", 0),
                "reference": str(m.get("reference", ""))[:500]  # keep short for UI
            })

        last_result = {
            "input_text": str(input_text),
            "plagiarism": {"matches": safe_matches},
            "ai_verdict": str(ai_result),
            "style_verdict": str(style_result)
        }
        result = last_result

    return render_template("index.html", result=result)


@app.route("/download_report")
def download_report():
    global last_result
    if not last_result:
        return redirect(url_for("index"))

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Title
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, height - 50, "AI & Plagiarism Detection Report")

    y = height - 80
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Plagiarism Matches:")
    y -= 20

    pdf.setFont("Helvetica", 10)
    matches = last_result["plagiarism"].get("matches", [])
    if matches:
        for m in matches:
            line = f"- {m['similarity']}% similar → {m['reference'][:100]}..."
            pdf.drawString(60, y, line)
            y -= 14
            if y < 60:
                pdf.showPage()
                y = height - 60
                pdf.setFont("Helvetica", 10)
    else:
        pdf.drawString(60, y, "No matches above threshold.")
        y -= 20

    # AI detection
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "AI Detection:")
    y -= 16
    pdf.setFont("Helvetica", 10)
    pdf.drawString(60, y, last_result.get("ai_verdict", "N/A"))
    y -= 20

    # Stylometry
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Stylometry:")
    y -= 16
    pdf.setFont("Helvetica", 10)
    pdf.drawString(60, y, last_result.get("style_verdict", "N/A"))
    y -= 30

    # Truncated text
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Checked Text (truncated):")
    y -= 16
    pdf.setFont("Helvetica", 9)
    text_to_write = last_result["input_text"][:4000]
    for i in range(0, len(text_to_write), 95):
        pdf.drawString(60, y, text_to_write[i:i+95])
        y -= 12
        if y < 60:
            pdf.showPage()
            y = height - 60
            pdf.setFont("Helvetica", 9)

    pdf.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="plagiarism_report.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)