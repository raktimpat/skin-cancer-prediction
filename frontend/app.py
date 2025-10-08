import streamlit as st
import requests
from PIL import Image
import io
import json
import os

# --- Configuration ---
API_URL = os.getenv("API_URL")  # IMPORTANT: Replace with your Cloud Run URL

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🔬")

text = st.button("Check API Status", on_click=lambda: requests.get(API_URL).status_code == 200)
if text:
    st.success(f"Your API is live")
# --- App UI ---
st.title("🔬 Skin Cancer Detection")
st.markdown("""
This app uses a Deep Learning model to classify a skin lesion as either **Benign** or **Malignant (Melanoma)**.

**Disclaimer:** This is an educational tool and **not a substitute for professional medical advice**.
""")
col1, col2 = st.columns(2)
with st.form("prediction_form", height="content"):
    

    with col1:
        st.markdown("### Upload Lesion Image")
        uploaded_file = st.file_uploader("Choose an image of a skin lesion...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', width=250)

            with col2:
                st.markdown("### Enter Metadata")
                st.markdown("""
                Please provide the following information:""")  
                st.write(" ") 
                st.write(" ") 
                st.write(" ") 
                st.write(" ") 
                st.write(" ") 
                st.write(" ") 
                st.write(" ") 
                st.write(" ") 
                st.write(" ") 
                age = st.select_slider(
                    'Select your age:',
                    options=range(0, 100), value=40.0)

                sex = st.radio("Select your gender:", options=["male", "female"])
                localization = st.selectbox("Select lesion area:", options=[
                    'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 
                    'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity'
                ])

                print("Submitting for prediction...")
                x = {"age": age, "sex": sex, "localization": localization}
                print(f"Payload: {x}")
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        submitted = st.form_submit_button("Classify Lesion", use_container_width=True)



if submitted and uploaded_file is not None:
    
    # Prepare the image for the API
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    metadata = {"age": age, "sex": sex, "localization": localization}
    files = {
        'image': ('image.jpg', img_bytes, 'image/jpeg'),
        'data': (None, json.dumps(metadata), 'application/json')
    }
    # Send request to the API
    try:
        response = requests.post(f"{API_URL}predict", files=files)

        if response.status_code == 200:
            result = response.json()
            prediction = result['class']
            if prediction == 'Benign':
                confidence = result['probabilities'][0]
                st.success(f"**Prediction: {prediction}**")
                st.info(f"Confidence: {confidence*100:.2f}%")
                st.markdown("✅ While the model predicts a benign lesion, always consult a healthcare professional for any skin concerns.")

            elif prediction == 'Malignant':
                confidence = result['probabilities'][1]
                st.error(f"**Prediction: {prediction}**")
                st.warning(f"Confidence: {confidence*100:.2f}%")
                st.markdown("🚨 **Please consult a dermatologist immediately.** This model has identified features consistent with melanoma.")
            else:
                confidence = result['probabilities'][2]
                st.success(f"**Prediction: {prediction}**")
                st.warning(f"Confidence: {confidence*100:.2f}%")
                st.markdown("🚨 **Please consult a dermatologist immediately.** This model has identified features consistent with pre cancer symptoms.")
        else:
            st.error(f"Error: Could not get a prediction. Status code: {response.status_code}")
            st.error(response.text)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the prediction service: {e}")