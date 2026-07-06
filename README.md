# 🎭 Text Emotion Classification using Deep Learning (NLP)

A comprehensive Natural Language Processing (NLP) and sequence modeling project designed to accurately classify human emotions from textual data using deep neural networks.

## 🧠 Model Architecture
The model uses an advanced deep learning sequential structure to process text data:
- **Embedding Layer:** Converts raw text strings into dense vector representations.
- **Bidirectional LSTM:** Captures deep sequential context from both left-to-right and right-to-left word orders.
- **Attention Mechanism:** Emphasizes highly informative words to improve sentiment analysis accuracy.
- **Dense Layer:** Uses a final softmax activation output to classify 6 target emotions.

## 📊 Dataset Distribution
- **Training Samples:** 16,000 text inputs
- **Validation Samples:** 2,000 text inputs
- **Testing Samples:** 2,000 text inputs

## 🏷️ Target Emotion Labels
The model can predict the following 6 distinct human emotions:
- `joy` 😊
- `sadness` 😢
- `anger` 😡
- `fear` 😨
- `love` ❤️
- `surprise` 😲

## 🧰 Tech Stack & Tools
- **Language:** Python
- **Libraries:** NLTK, NumPy, Pandas
- **Frameworks:** TensorFlow & Keras (Deep Learning)
- **Environment:** Google Colab (With T4 GPU hardware acceleration)
