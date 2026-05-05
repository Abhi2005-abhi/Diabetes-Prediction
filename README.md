# 🩺 Diabetes Prediction App

Welcome to my **Diabetes Prediction** project!  
This simple but powerful web app helps predict whether a person has diabetes based on common medical check‑up numbers.  
You can use it to test a single patient, upload a batch of data, or just learn about the model behind it.

👉 **Live Demo:** *[Your Streamlit URL after deployment]*

---

## 🎯 What does this app do?

- **Single Prediction** – Enter medical values (glucose, BMI, age, etc.) and click a button.  
  The app instantly tells you if the person is likely to have diabetes and shows a probability score.

- **Batch Prediction** – Upload a CSV file with many patient records.  
  The app will add a column saying “Diabetic” or “Not Diabetic” and let you download the results.

- **About & Model Info** – Learn what each medical measurement means and how the model performs.

---

## 📊 Where does the data come from?

The app uses the **PIMA Indian Diabetes Dataset** (provided by the National Institute of Diabetes and Digestive and Kidney Diseases).  
It contains real medical records of 768 female patients.

---

## 🧠 How does it work?

Under the hood, the app uses a **Logistic Regression** model – a popular machine learning algorithm for yes/no predictions.  
The model was trained on 80% of the dataset and tested on the remaining 20%.  
It achieves about **78–80% accuracy**, which is quite good for this type of medical prediction.

---

## 🛠️ What’s inside the project?

- `app.py` – The main Streamlit application.
- `diabetes.csv` – The dataset.
- `requirements.txt` – List of Python libraries needed.
- `models/` – Folder where the trained model is saved.
- `README.md` – This file you’re reading.

---

## 🚀 How to run the app on your own computer

1. **Clone this repository**  
   ```bash
   git clone https://github.com/Abhi2005-abhi/Diabetes-Prediction.git
   cd Diabetes-Prediction
