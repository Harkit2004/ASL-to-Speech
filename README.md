# ASL to Speech Converter

A real-time American Sign Language (ASL) recognition system that converts sign language gestures into natural speech using a custom-trained machine learning model and ElevenLabs text-to-speech technology.

## Demo

https://github.com/user-attachments/assets/ba106aef-8550-4272-ae34-b05aa58a20f4

## Features

- **Real-time ASL Recognition**: Processes video input to identify ASL gestures
- **Custom Trained Model**: Self-trained machine learning model optimized for ASL detection
- **High-Quality Speech Synthesis**: Integration with ElevenLabs for natural voice output
- **User-Friendly Interface**: Simple setup and operation

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Harkit2004/ASL-to-Speech.git
cd ASL-to-Speech
```

### 2. ElevenLabs API Configuration
1. Sign up for an ElevenLabs account at https://elevenlabs.io
2. Generate your API key from the dashboard
3. Add your API key to the configuration:
   ```python
   ELEVENLABS_API_KEY = "your_api_key_here"
   ```

### 3. Install Dependencies
Run the first cell in the Colab notebook to install required packages:
```python
pip install -r requirements.txt
```

## Project Structure

```
ASL-to-Speech/
├── ASLToSpeech.ipynb
├── app.py
├── gesture_recognizer.task
├── .gitignore
├── requirements.txt
└── README.md
```

## Model Architecture

The ASL recognition model uses:
- **Convolutional Neural Networks (CNN)** for feature extraction
- **Recurrent Neural Networks (RNN)** for temporal sequence processing
- **Transfer Learning** with pre-trained vision models
- **Data Augmentation** for improved generalization

## Dataset

The model is trained on: [https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)