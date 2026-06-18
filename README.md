# 🐍 Slitherin — Indian Snake Species Classification App

Slitherin is an AI-powered mobile application that identifies Indian snake species from photographs using deep learning and computer vision. It's built to support wildlife conservation, public awareness, and faster, safer responses during snakebite incidents.

📲 **Live on Google Play:** [Play Store](https://play.google.com/store/apps/details?id=com.gs.slitherin)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Performance](#performance)
- [Deployment](#deployment)
- [Android Application](#android-application)
- [Multilingual Support](#multilingual-support)
- [Tech Stack](#tech-stack)
- [Project Status](#project-status)
- [Real-World Applications](#real-world-applications)
- [Research](#research)
- [Dataset & Acknowledgements](#dataset--acknowledgements)
- [License](#license)

---

## Overview

Slitherin classifies Indian snake species directly from smartphone images, combining a fine-tuned ResNet50 model with a cloud-hosted inference pipeline and a production Android app. The system is live on the Google Play Store and supports four languages, making accurate snake identification accessible to a wide range of users across India.

## Motivation

India hosts a large diversity of snake species, many of which are difficult for the general public to tell apart. Misidentification commonly leads to the unnecessary killing of non-venomous snakes, delayed medical treatment during snakebite incidents, and weaker outcomes for wildlife conservation efforts. Slitherin addresses this by putting a fast, image-based identification tool directly in people's hands.

## Dataset

A custom snake image dataset was collected and curated from multiple sources:

- **~13 snake species classes**, covering both venomous and non-venomous species found in India
- **~250–350 images per class**
- Diverse backgrounds, lighting conditions, poses, and orientations to support real-world robustness

## Data Preprocessing

Several preprocessing and augmentation techniques were used to improve generalization and handle class imbalance:

- Image resizing
- Normalization
- Horizontal flipping
- Rotation
- Zoom augmentation
- CutMix augmentation
- SMOTE-based class balancing

These techniques were chosen to reduce overfitting, improve robustness across image conditions, correct class imbalance, and increase overall dataset diversity.

## Model

The classifier was built using **transfer learning** on a **ResNet50** backbone.

**Training strategy:**
- Transfer learning + fine-tuning
- Data augmentation
- Regularization techniques

**Frameworks:** TensorFlow, Keras, NumPy, Scikit-learn

## Performance

The trained model achieved high classification accuracy on the held-out test set, with generalization gains driven by:

- Transfer learning
- CutMix augmentation
- SMOTE balancing
- Hyperparameter optimization

## Deployment

The trained model is deployed on **Hugging Face** for real-time, cloud-based inference:

```
Image Upload → Hugging Face API → Model Inference → Species Prediction → Mobile App Display
```

This setup keeps the mobile app lightweight, allows model updates without app releases, and scales inference in the cloud rather than on-device.

## Android Application

A production-ready Android app was built in **Kotlin**.

**Features:**
- Capture image via camera
- Select image from gallery
- Real-time snake species prediction
- Species information display
- Fast cloud-based inference
- User-friendly interface

**Frontend:** Kotlin, Android Studio, XML
**Backend:** Hugging Face Inference API

## Multilingual Support

To improve accessibility across India, the app supports:

- English
- Tamil
- Telugu
- Hindi

## Tech Stack

**Machine Learning:** TensorFlow, Keras, NumPy, Scikit-learn
**Deployment:** Hugging Face
**Mobile Development:** Kotlin, Android Studio
**Cloud Inference:** REST API integration

## Project Status

**✅ Completed**
- Dataset collection
- Data preprocessing
- Model training
- Model deployment
- Android application development
- Google Play Store release
- Multilingual integration

**🚧 In Progress**
- Research paper preparation
- Model enhancement
- Additional species expansion

## Real-World Applications

- Wildlife conservation
- Snake awareness programs
- Educational tools
- Emergency snake identification
- Biodiversity monitoring
- Citizen science initiatives

## Research

The project is being extended into a research publication, with ongoing work on performance evaluation, comparative analysis of deep learning architectures, methodology documentation, and manuscript preparation.

## Dataset & Acknowledgements

This project uses snake image data from:

**Giant Snake Data (60+ species) – Kaggle's Biggest**
Source: Kaggle
Link: https://www.kaggle.com/datasets/shouvikdey21/giant-snake-data60-species-kaggles-biggest

The dataset is distributed under the MIT License. See [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) for the full license text and copyright notice.

## License

* Apache 2.0 

---

*Slitherin demonstrates the practical deployment of a deep learning model from research to a production mobile application used by the public.*
