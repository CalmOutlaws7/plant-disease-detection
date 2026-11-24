# MediLeaf AI: Plant Disease Prediction Dashboard

An advanced, deep learning-powered web application designed as an **Alternative Assessment Tool (AAT)** for automated plant health classification and disease remediation guidance.

---

## Project Team
* **Monish M** (1BM24AI090)
* **Mohammed Fahim** (1BM24AI088)
* **Prajwal C V** (1BM24AI113)

**Guide:** Prof. Shridevi A Desai  
**Department of Artificial Intelligence & Machine Learning**  
**BMS College of Engineering**

---

## About the Project
Plant diseases pose a severe threat to agricultural productivity. This project leverages computer vision and deep learning to instantly classify plant leaf health across multiple species, providing accurate diagnoses alongside actionable chemical and organic treatment strategies.

### Key Features
* **Multi-Class Deep Learning Classification:** Trained on 23 unique disease and health classes across 5 core plant varieties.
* **Real-Time Inference Dashboard:** Built with Streamlit for instant image uploads, confidence breakdown meters (Top-3 predictions), and dynamic visual health badges.
* **Agricultural Knowledge Base:** Integrated treatment guidance covering chemical controls, organic remedies, and preventive measures.
* **Downloadable Diagnosis Reports:** Instant generation of professional text reports for record-keeping.

---

## Technical Stack & Specifications
* **Architecture:** MobileNetV2 (Transfer Learning with PyTorch)
* **Dataset Size:** 15,000+ images across 5 plant species (*Roxburgh Fig, Jasmine, Mango, Mint, Drumstick*)
* **Model Accuracy:** ~95.77% training accuracy
* **Hardware Acceleration:** Trained using PyTorch with CUDA on an NVIDIA RTX 5060 GPU
* **Frameworks & Libraries:** PyTorch, Torchvision, Streamlit, Pandas, Pillow

---

## Project Directory Structure
```text
plant_disease_project/
│
├── dataset/             # Local training and validation images (excluded from git)
├── samples/             # Sample test images for live demonstrations
├── app.py               # Main Streamlit web application
├── train.py             # PyTorch training script with CUDA support
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation