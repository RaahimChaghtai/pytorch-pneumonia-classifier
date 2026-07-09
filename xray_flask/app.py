import base64
import os
import sys
from pathlib import Path

from flask import Flask, render_template, request

# RunPod version calls a remote API. Locally we import handler.py from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from handler import handler

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    original_image = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            return render_template("index.html", error="Please choose an image file.")

        image_data = base64.b64encode(file.read()).decode("utf-8")
        job = {"input": {"image": image_data}}
        result = handler(job)

        if "error" in result:
            return render_template("index.html", error=result["error"])

        prediction = result["prediction"]
        original_image = f"data:image/jpeg;base64,{image_data}"

    return render_template(
        "index.html",
        original_image=original_image,
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
