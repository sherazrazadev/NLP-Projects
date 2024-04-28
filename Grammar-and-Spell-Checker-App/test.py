from flask import Flask, request, jsonify, render_template
import language_tool_python
from textblob import TextBlob
import docx
import PyPDF2

tool = language_tool_python.LanguageTool('en-US')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def highlight_errors(text, matches):
    highlighted_text = text
    details = []
    for match in matches:
        length = match.errorLength
        offset = match.offset
        error_context = text[max(0, offset - 10):offset + length + 10]  # Context around the error
        suggested_correction = match.replacements
        error_text = text[offset:offset + length]
        highlighted_text = (highlighted_text[:offset] + "<span class='error'>" + 
                            highlighted_text[offset:offset + length] + "</span>" + 
                            highlighted_text[offset + length:])
        details.append({
            'error_text': error_text,
            'context': error_context,
            'suggestions': suggested_correction
        })
    return highlighted_text, details

@app.route('/check_grammar', methods=['POST'])
def check_grammar():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    filename = file.filename

    if not filename:
        return jsonify({"error": "No file selected"})

    if filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        return jsonify({"error": "Unsupported file format"})

    corrected_text = TextBlob(text).correct()
    matches = tool.check(str(corrected_text))
    highlighted_text, error_details = highlight_errors(text, matches)

    return jsonify({
        'revised_text': highlighted_text,
        'errors': error_details,
        'total_errors': len(matches)
    })

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

if __name__ == '__main__':
    app.run(debug=True)
