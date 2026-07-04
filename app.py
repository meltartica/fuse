from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, ImageFilter, ImageOps
import io
import base64

app = Flask(__name__)


def preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    img = img.filter(ImageFilter.MedianFilter(3))
    return img


@app.route("/", methods=["GET", "OPTIONS"])
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["POST"])
def ocr():
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

    # Fix: x-www-form-urlencoded decodes '+' as space; base64 never has spaces
    raw = raw.replace(" ", "+")

    try:
        img = Image.open(io.BytesIO(base64.b64decode(raw)))
    except Exception:
        return jsonify({
            "ParsedResults": [],
            "OCRExitCode": 0,
            "IsErroredOnProcessing": True,
            "ErrorMessage": ["Invalid image data"],
        }), 400

    img = preprocess(img)

    text = pytesseract.image_to_string(
        img,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    )

    clean = text.strip().replace(" ", "").upper()

    return jsonify({
        "ParsedResults": [{"ParsedText": clean}],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
