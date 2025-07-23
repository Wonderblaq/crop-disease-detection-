🥔 Potato Disease Detection API
A simple FastAPI endpoint for detecting potato leaf diseases using a TensorFlow transfer learning model.

🚩 Features
✅ Detects potato leaf diseases from images.
✅ Returns predicted disease class and confidence score.
✅ Built for hackathon 2025 crop disease detection workflow.

🚀 Usage
1️⃣ Start the server:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

2️⃣ Test the API:

GET / → Health check.

POST /predict → Upload a leaf image (file field) to get prediction.

🛠 Requirements
Python 3.11+

TensorFlow

FastAPI

Uvicorn

python-multipart

Install dependencies:
pip install tensorflow fastapi uvicorn python-multipart

📦 Model
Uses a MobileNetV2 transfer learning model trained on potato leaf images with:

Early blight

Late blight

Healthy

🤝 Author
Sarfo Wonder Nana
Hackathon 2025
University of Energy and Natural Resources (UENR), Ghana

how to run fast api to generate endpoint :
Run------>
"Uvicorn main:app --reload " in main.py(fast api)
terminal to generate endpoint and test with 
postman








