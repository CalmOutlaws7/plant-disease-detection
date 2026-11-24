# Plant Disease Detection System

A deep learning web application built with PyTorch and Streamlit to classify plant leaf health and diseases.

## Features
- Upload plant leaf images (JPG, PNG)
- Real-time model inference using MobileNetV2
- Clean and simple web dashboard

## Project Structure
- `app.py`: Streamlit web application interface.
- `train.py`: PyTorch training script using transfer learning.
- `model.pth`: Saved model weights.

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install torch torchvision streamlit pillow