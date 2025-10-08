import torch
import joblib
import json
import io
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import pandas as pd
from network import MultiInputNN, image_transform

app = FastAPI(title="Skin Cancer Detection API")

preprocessor = joblib.load('models/tabular_preprocessor.pkl')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
num_meta_features = preprocessor.transform(pd.DataFrame([{
    'age': 45,
    'sex': 'male',
    'localization': 'scalp'
}])).shape[1]
model = MultiInputNN(num_meta_features=num_meta_features, num_classes=3)

model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.to(device)
model.eval()

class MetaData(BaseModel):
    age: int
    sex: str
    localization: str


@app.get('/')
async def home():
    return {"message": "Welcome to Skin Cancer Classifier API"}

@app.post('/predict')
async def predict(image: UploadFile = File(..., description="The skin lesion image file."),
    data: str = Form(..., description="A JSON string with patient metadata: {'age': int, 'sex': str, 'localization': str}.")
):

    # -------- Preprocess Inputs -------- #
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = image_transform(img).unsqueeze(0).to(device)

    meta_dict = json.loads(data)
    # Validate data using the Pydantic model
    validated_meta = MetaData(**meta_dict)
    # Convert to DataFrame for the preprocessor
    meta_df = pd.DataFrame([validated_meta.model_dump()])
    meta_processed = preprocessor.transform(meta_df)
    meta_tensor = torch.tensor(meta_processed.toarray(), dtype=torch.float32).to(device)


    # -------- Make Prediction -------- #
    with torch.no_grad():
        outputs = model(image_tensor, meta_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        class_names = ['Benign', 'Malignant', 'Other']
        predicted_class = class_names[predicted_class]
        
    result = {'class': predicted_class,
                    'probabilities': probabilities.tolist()[0]}
    
    return result

