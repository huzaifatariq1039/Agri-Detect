🌿 Agri-Detect: Plant Disease Detection System

Agri-Detect is an AI-powered web application designed to help farmers and agricultural researchers identify plant diseases instantly from leaf images. By leveraging Deep Learning and Transfer Learning, the system provides high-accuracy diagnosis to help mitigate crop loss.

🚀 Project Overview

In many regions, including Pakistan, farmers lack immediate access to agricultural experts. This project bridges that gap by providing a localized, accessible tool for disease triage.

Model Accuracy: 93.58% (Fine-tuned MobileNetV2)

Frameworks: TensorFlow/Keras, Flask

Dataset: PlantVillage (38 classes of healthy and diseased leaves)

📊 Performance

The model was initially trained using Transfer Learning, achieving 92% accuracy. Through meticulous fine-tuning (unfreezing top layers and training with a very low learning rate of $10^{-5}$), the accuracy was pushed to 93.58%.

Training: 70%

Validation: 15%

Test: 15% (Strictly unseen data)

🛠️ Tech Stack

Backend: Flask (Python)

Deep Learning: TensorFlow 2.x, Keras

Frontend: HTML5, CSS3 (Bootstrap), JavaScript (Fetch API)

Environment: Google Colab (Training), VS Code (Deployment)

📂 Project Structure

agri-detect/
├── app.py                      # Flask Backend logic
├── plant_disease_model.h5      # Fine-tuned Model weights
├── static/                     # CSS, JS, and UI assets
├── templates/
│   └── index.html              # Frontend User Interface
├── notebooks/
│   └── Training_FineTuning.ipynb # Colab notebook for reproducibility
└── README.md


⚙️ Installation & Setup

1. Prerequisites

Python 3.9+ (64-bit)

Virtual Environment (recommended)

2. Clone the Repository

git clone [https://github.com/your-username/agri-detect.git](https://github.com/your-username/agri-detect.git)
cd agri-detect


3. Create a Virtual Environment

python -m venv venv
# Activate on Windows
.\venv\Scripts\activate
# Activate on Mac/Linux
source venv/bin/activate


4. Install Dependencies

pip install flask tensorflow-cpu numpy pillow werkzeug


5. Run the Application

python app.py


Open your browser and navigate to http://127.0.0.1:5000.

🧠 Model Fine-Tuning Strategy

The project utilized a "Two-Step" training approach:

Initial Training: Frozen base layers, training only the classification head.

Fine-Tuning: Unfreezing from block_13_expand onwards and re-training with ReduceLROnPlateau to achieve optimal convergence.

🌟 Future Roadmap

[ ] Integration with a Mobile App (Flutter/React Native).

[ ] Multilingual support (Urdu/Regional languages).

[ ] Offline diagnosis using TFLite.

[ ] IoT integration for automated greenhouse monitoring.

📄 License

Distributed under the MIT License. See LICENSE for more information.

Developed by Huzaifa - AI Engineer
