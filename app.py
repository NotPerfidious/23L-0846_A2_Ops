import os
import base64
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
# from src.inference import predict_image_bytes
import psycopg2
from psycopg2.extras import DictCursor
import requests

AWS_MODEL_URL = "http://13.53.135.120:5000/predict"

app = Flask(__name__, template_folder="./templates")

# Config from environment variables
app.config["MAX_UPLOAD_SIZE"] = int(os.getenv("MAX_UPLOAD_SIZE", 5 * 1024 * 1024))
DATABASE_ADDRESS = os.getenv("DATABASE_ADDRESS", "visionguard_history.db")
MODEL_PATH = os.getenv("MODEL_PATH", "models:/VisionGuard_Classifier@champion")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize Database with the new "model_name" column for Task 5
def init_db():
    conn = psycopg2.connect(os.getenv("DATABASE_ADDRESS"))
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS history
                 (id SERIAL PRIMARY KEY,
                  filename TEXT,
                  label TEXT,
                  confidence REAL,
                  model_name TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


@app.route("/", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def process_upload():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("error.html", message="No file was selected."), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        return (
            render_template(
                "error.html",
                message="Invalid file type. Only PNG, JPG, and JPEG are allowed.",
            ),
            400,
        )

    file_bytes = file.read()

    if len(file_bytes) > app.config["MAX_UPLOAD_SIZE"]:
        return (
            render_template("error.html", message=f"File exceeds the size limit."),
            413,
        )

    if len(file_bytes) == 0:
        return render_template("error.html", message="The uploaded file is empty."), 400

    try:
        # result = predict_image_bytes(file_bytes)

        files = {"file": (filename, file_bytes, file.mimetype)}
        response = requests.post(AWS_MODEL_URL, files=files, timeout=30)

        if response.status_code != 200:
            return (
                render_template("error.html", message="Cloud model inference failed."),
                502,
            )

        result = response.json()

        # Save to Database including the model_name
        conn = psycopg2.connect(os.getenv("DATABASE_ADDRESS"))
        c = conn.cursor()
        c.execute(
            "INSERT INTO history (filename, label, confidence, model_name) VALUES (%s, %s, %s, %s)",
            (filename, result["label"], result["confidence"], MODEL_PATH),
        )
        conn.commit()
        conn.close()

    except requests.exceptions.RequestException:
        return (
            render_template(
                "error.html", message="Could not connect to the cloud model server."
            ),
            503,
        )
    except ValueError as e:
        return render_template("error.html", message=str(e)), 422
    except Exception as e:
        return (
            render_template(
                "error.html", message="An unexpected server error occurred."
            ),
            500,
        )

    encoded_img = base64.b64encode(file_bytes).decode("utf-8")
    img_data_uri = f"data:{file.mimetype};base64,{encoded_img}"

    return render_template(
        "result.html", result=result, filename=filename, image_data=img_data_uri
    )


@app.route("/history", methods=["GET"])
def history_page():
    # Fetch from Database
    conn = psycopg2.connect(os.getenv("DATABASE_ADDRESS"))

    c = conn.cursor(cursor_factory=DictCursor)
    c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 50")
    records = c.fetchall()
    conn.close()
    return render_template("history.html", history=records)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "VisionGuard API"}), 200


if __name__ == "__main__":
    init_db()
    app.run(debug=False, host="0.0.0.0", port=5000)
