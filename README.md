# Skin Cancer Detection System 🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

A full-stack, end-to-end machine learning project that classifies dermoscopic images of skin lesions into three medically relevant categories: **Benign**, **Melanoma**, and **Other Cancers**. This system is built using a multi-input PyTorch model and deployed on a serverless, cost-effective cloud architecture.

---

### 🚀 [Live Demo](https://myeloma-prediction-rkt.streamlit.app/)


---

### 🏛️ System Architecture

The project follows a modern, scalable MLOps architecture designed for efficiency and low cost.



1.  **Frontend:** A user-friendly web interface built with **Streamlit** and deployed on Streamlit Community Cloud.
2.  **Backend:** A high-performance REST API built with **FastAPI** to serve model predictions.
3.  **Containerization:** The backend is containerized using **Docker** to ensure a consistent and reproducible environment.
4.  **Deployment:** The Docker container is pushed to **Google Artifact Registry** and deployed as a serverless service on **Google Cloud Run**.

---

### ✨ Key Features

* **Multi-Input Model:** Leverages both image data and tabular patient metadata (age, sex, localization) for more accurate predictions.
* **Deep Learning with PyTorch:** Utilizes a pretrained MobileNetV2 for transfer learning, optimized for performance on an low end hardware.
* **Serverless API:** A robust FastAPI backend capable of handling file uploads and JSON data, deployed on a scalable, pay-per-use infrastructure.

---

### 🛠️ Technology Stack

| Category           | Technologies                                                                                                                                                                                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ML Framework** | `PyTorch`, `Torchvision`                                                                                                                                                                                                                                                                                |
| **Data Science** | `Pandas`, `Scikit-learn`, `Pillow`                                                                                                                                                                                                                                                                        |
| **Backend API** | `FastAPI`, `Uvicorn`                                                                                                                                                                                                                                                                                    |
| **Frontend App** | `Streamlit`                                                                                                                                                                                                                                                                                             |
| **Cloud & MLOps** | `Docker`, `Google Cloud Run`, `Google Artifact Registry`                                                                                                                                                                                                                                                |
| **Environment** | `Python 3.9`                                                                                                                                                                                                                                                                                   |

---

### ⚙️ Local Setup and Installation

To run this project on your local machine, follow these steps:

**1. Clone the Repository**

**2. Create a Conda Environment**
This project uses a environment to manage dependencies.

```bash
python3.9 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
Install all required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Download Model Artifacts**
* Download the trained model weights (`best_skin_cancer_model.pth`).
* Download the saved preprocessor (`tabular_preprocessor.pkl`).
* Place both files in the root directory of the backend application.

**5. Run the Backend API**
Navigate to the backend directory and run the Uvicorn server.
```bash
python -m uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

**6. Run the Frontend Application**
Open a new terminal window and run the Streamlit app.
```bash
# Run the app
streamlit run app.py
```
The Streamlit app will be available at `http://localhost:8501`.

---


### ⚠️ Ethical Disclaimer

This application is an **educational proof-of-concept** and must **NEVER** be used for actual medical diagnosis. The model's accuracy is not perfect, and self-diagnosis can be dangerous. Always consult a qualified dermatologist for any health concerns.

