from flask import Flask, request, jsonify
import ddddocr
from PIL import Image
import io
import base64

app = Flask(__name__)
ocr = ddddocr.DdddOcr(show_ad=False)


@app.route("/", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["POST"])
def ocr_endpoint():
    raw = request.form.get("base64Image", "")
    if not raw:
        return jsonify({
            "ParsedResults": [],
            "OCRExitCode": 0,
            "IsErroredOnProcessing": True,
            "ErrorMessage": ["Missing base64Image parameter"],
        }), 400

    if "," in raw:
        raw = raw.split(",")[1]

    raw = raw.replace(" ", "+")

    try:
        img_bytes = base64.b64decode(raw)
    except Exception:
        return jsonify({
            "ParsedResults": [],
            "OCRExitCode": 0,
            "IsErroredOnProcessing": True,
            "ErrorMessage": ["Invalid base64 data"],
        }), 400

    try:
        text = ocr.classification(img_bytes)
    except Exception:
        return jsonify({
            "ParsedResults": [],
            "OCRExitCode": 0,
            "IsErroredOnProcessing": True,
            "ErrorMessage": ["OCR failed"],
        }), 500

    clean = text.strip().replace(" ", "").upper()

    return jsonify({
        "ParsedResults": [{"ParsedText": clean}],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
