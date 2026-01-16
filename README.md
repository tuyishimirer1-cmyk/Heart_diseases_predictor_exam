# Heart Disease Risk Prediction System

This project is a **clinical decision support system** to predict the **risk level of heart disease** in patients. The system uses **machine learning models** trained on patient demographic, clinical, and diagnostic data, and provides **color-coded risk predictions** with probability distribution for all classes.  

The system includes:
- **Data preprocessing and machine learning model training**
- **Model evaluation and hyperparameter tuning**
- **Flask API deployment**
- **Responsive HTML frontend for medical staff**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Deployment](#deployment)
7. [Frontend](#frontend)
8. [Testing](#testing)
9. [Video Demonstration](#video-demonstration)
10. [License](#license)

---

## Project Overview

The Heart Disease Risk Prediction System is designed to help medical staff **assess a patient's risk of heart disease** quickly and accurately. 

Key features:
- Supports **5 risk classes**: No Disease, Very Mild, Mild, Severe, Immediate Danger
- Uses **13 input features** such as age, blood pressure, cholesterol, ECG results, chest pain type, vessels affected, thalassemia status, etc.
- **Color-coded outputs** for easier clinical interpretation
- Accessible through a **modern, responsive frontend**

---

## Dataset

- Size: 5,000 patient records
- Features: 13 input features
- Target: 5-class diagnosis label
  - 0 = No Disease
  - 1 = Very Mild
  - 2 = Mild
  - 3 = Severe
  - 4 = Immediate Danger

The dataset includes both **numerical and categorical features**.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tuyishimirer1-cmyk/Heart_diseases_predictor_exam.git

