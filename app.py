import os
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# FIX: Create the upload folder automatically if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the YOLOv11 model
model = YOLO('best.pt')
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform prediction
        results = model(filepath)
        
        # Get the top class name and confidence score
        top1_idx = results[0].probs.top1
        prediction = results[0].names[int(top1_idx)] 
        confidence = round(results[0].probs.top1conf.item() * 100, 2)

        # FIX: Call the function to get the correct description
        description_text = get_disease_description(prediction)

        # FIX: Ensure variable names match what result.html expects
        return render_template("result.html",
                               prediction=prediction,
                               confidence=confidence,
                               description=description_text,
                               image_path=filename) # Added missing closing parenthesis
if __name__ == '__main__':
    # FIX: Use dynamic port for deployment environments
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)