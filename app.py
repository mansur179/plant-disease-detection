app_code = '''
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("best.pt")  # ✅ local path for deployment

def get_disease_description(predicted_class):
    descriptions = {
        "healthy": "The leaf appears healthy. No signs of disease detected.",
        "Bacterial_spot": "Bacterial infection detected. Causes water-soaked dark spots on leaves. Treat with copper-based bactericides.",
        "Early_blight": "Early blight fungal disease detected. Brown spots with concentric rings visible. Apply fungicide and remove infected leaves.",
        "Late_blight": "Late blight detected - a serious fungal disease. Can destroy entire crops rapidly. Apply fungicide immediately.",
        "Leaf_Mold": "Leaf mold detected. Yellow patches on upper leaf surface with mold underneath. Improve ventilation and apply fungicide.",
        "Septoria_leaf_spot": "Septoria leaf spot detected. Small circular spots with dark borders. Remove infected leaves and apply fungicide.",
        "Spider_mites": "Spider mite infestation detected. Causes yellowing and stippling. Apply miticide or neem oil.",
        "Target_Spot": "Target spot fungal disease detected. Circular lesions with rings like a target. Apply appropriate fungicide.",
        "YellowLeaf": "Yellow Leaf Curl Virus detected. Causes yellowing and curling. Manage whitefly vectors immediately.",
        "mosaic_virus": "Mosaic virus detected. Causes mosaic-like discoloration. Remove infected plants to prevent spread.",
    }
    for key, desc in descriptions.items():
        if key.lower() in predicted_class.lower():
            return desc
    return "Disease detected. Consult an agricultural expert for treatment recommendations."

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    results = model(filepath)
    predicted_class = results[0].names[results[0].probs.top1]
    confidence = round(results[0].probs.top1conf.item() * 100, 2)
    description = get_disease_description(predicted_class)
    return render_template("result.html",
        prediction=predicted_class,
        confidence=confidence,
        description=description,
        image_path=file.filename
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
'''

with open('/content/drive/MyDrive/webapp/app.py', 'w') as f:
    f.write(app_code)

print("✅ app.py updated!")