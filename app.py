from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, ImageFilter, ImageOps
import io
import base64

app = Flask(__name__)

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
TESS_CONFIG = f"--psm 7 -c tessedit_char_whitelist={WHITELIST}"


def preprocess(img: Image.Image, threshold: int = 140, contrast_cutoff: int = 2) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=contrast_cutoff)
    img = img.point(lambda x: 0 if x < threshold else 255, "1")
    img = img.filter(ImageFilter.MedianFilter(3))
    return img


def preprocess_sharp(img: Image.Image, threshold: int = 140, contrast_cutoff: int = 2) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=contrast_cutoff)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda x: 0 if x < threshold else 255, "1")
    img = img.filter(ImageFilter.MedianFilter(3))
    return img


def upscale(img: Image.Image, factor: int = 3) -> Image.Image:
    w, h = img.size
    return img.resize((w * factor, h * factor), Image.LANCZOS)


def ocr_with_confidence(img: Image.Image) -> tuple[str, float]:
    """Run Tesseract and return (text, confidence)."""
    data = pytesseract.image_to_data(img, config=TESS_CONFIG, output_type=pytesseract.Output.DICT)
    texts = []
    confs = []
    for t, c in zip(data["text"], data["conf"]):
        if t.strip() and int(c) > 0:
            texts.append(t)
            confs.append(int(c))
    if not texts:
        return "", 0.0
    return "".join(texts), sum(confs) / len(confs)


def fix_ocr_confusions(text: str) -> str:
    """Fix common Tesseract misreads in captchas (letters ↔ digits)."""
    has_digit = any(c.isdigit() for c in text)
    has_letter = any(c.isalpha() for c in text)

    if has_digit and has_letter:
        replacements = {
            "O": "0", "o": "0", "D": "0", "Q": "0",
            "I": "1", "l": "1", "i": "1",
            "Z": "2", "z": "2",
            "S": "5", "s": "5",
            "B": "8", "b": "8",
            "T": "7", "t": "7",
            "G": "6", "g": "6",
            "q": "9",
        }
        result = []
        for c in text:
            result.append(replacements.get(c, c))
        return "".join(result)

    if has_letter and not has_digit:
        replacements = {
            "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
            "7": "T", "6": "G", "9": "q", "3": "B",
        }
        result = []
        for c in text:
            result.append(replacements.get(c, c))
        return "".join(result)

    return text


def try_ocr(img: Image.Image) -> tuple[str, float]:
    """Try multiple preprocessing pipelines, return best result."""
    pipelines = [
        {"threshold": 140, "contrast_cutoff": 2, "sharp": False},
        {"threshold": 120, "contrast_cutoff": 3, "sharp": False},
        {"threshold": 160, "contrast_cutoff": 1, "sharp": False},
        {"threshold": 100, "contrast_cutoff": 5, "sharp": False},
        {"threshold": 140, "contrast_cutoff": 2, "sharp": True},
        {"threshold": 120, "contrast_cutoff": 3, "sharp": True},
        {"threshold": 160, "contrast_cutoff": 1, "sharp": True},
    ]

    best_text, best_conf = "", 0.0

    for params in pipelines:
        sharp = params.pop("sharp")
        fn = preprocess_sharp if sharp else preprocess
        processed = fn(img, **params)
        for scaled in [processed, upscale(processed)]:
            text, conf = ocr_with_confidence(scaled)
            text = text.replace(" ", "").upper()
            text = fix_ocr_confusions(text)
            if 2 <= len(text) <= 10 and conf > best_conf:
                best_text, best_conf = text, conf

    return best_text, best_conf


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

    clean, confidence = try_ocr(img)

    return jsonify({
        "ParsedResults": [{"ParsedText": clean}],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "Confidence": round(confidence, 1),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
