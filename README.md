# 🌿 PlantGuard AI — Plant Disease Detection

> Developed by **Nivetha D**

![PlantGuard AI](assets/screenshot.png)

## 📌 Project Overview
PlantGuard AI detects plant diseases using **MobileNetV2**, a lightweight and efficient deep learning model. Upload a leaf image and instantly get the disease name, severity, treatment, and prevention tips — all through a clean, modern web interface built with Streamlit.

## 🚀 Features
✅ **MobileNetV2-based CNN** for efficient and accurate classification
✅ **Pretrained on ImageNet** — no dataset required to run the prototype
✅ **38 PlantVillage disease classes** supported
✅ **Severity indicators** — None / Moderate / High / Critical
✅ **Treatment & Prevention** recommendations per disease
✅ **Top 5 predictions** with confidence scores
✅ **Modern dark UI** built with Streamlit

## 📁 Project Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application |
| `plant_disease_detection1.ipynb` | Data loading, EDA, model building & training |
| `plant_disease_detection2.ipynb` | Fine-tuning, evaluation, metrics & prediction |
| `modified_class_indices.json` | Human-readable class label mappings |
| `requirements.txt` | Python dependencies |

## 🛠️ Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🧠 Model Architecture
- **Base**: MobileNetV2 (ImageNet weights, frozen)
- **Head**: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(38, Softmax)
- **Fine-tuning**: Top 20 layers unfrozen, trained at `lr=1e-5`

## 📊 Training Pipeline (`plant_disease_detection1.ipynb`)
1. Load & extract dataset from zip
2. Set up `ImageDataGenerator` with augmentation
3. Exploratory Data Analysis — class distribution
4. Build MobileNetV2 model with custom head
5. Train for 5 epochs (frozen base)

## 🔬 Evaluation & Prediction (`plant_disease_detection2.ipynb`)
1. Evaluate on test set — accuracy & loss plots
2. Fine-tune top 20 layers for 5 more epochs
3. Calculate Precision, Recall & F1 Score
4. Single image prediction demo
5. Save model to `.h5`

## 📸 Demo
Upload any plant leaf image to get:
- 🔍 Predicted disease name
- 🟡 Severity badge
- 💊 Treatment advice
- 🛡️ Prevention tips
- 📋 Top 5 alternative predictions

## 📄 License
For educational use only.
