# Spam Classifier

This project is a machine learning-based spam classifier that detects whether a given message is spam or not. It uses a Support Vector Machine (SVM) model—a powerful supervised learning algorithm—trained on a dataset of SMS messages, and provides a user-friendly web interface built with Streamlit.
## What is SVM?
Support Vector Machine (SVM) is a supervised machine learning algorithm commonly used for classification tasks. SVM works by finding the optimal boundary (hyperplane) that best separates data points of different classes. In this project, SVM is used to distinguish between spam and non-spam (ham) messages based on their features extracted from the text.


## Features
- Classifies messages as spam or ham (not spam)
- Interactive web app using Streamlit
- Pre-trained SVM model and TF-IDF vectorizer
- Easy-to-use interface for message input and prediction

## How It Works
1. The user enters a message in the web app.
2. The message is transformed using a TF-IDF vectorizer.
3. The SVM model predicts whether the message is spam or ham.
4. The result is displayed instantly.

## Project Structure
```
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── models/
│   ├── spam_svm_model.pkl  # Trained SVM model
│   └── tfidf_vectorizer.pkl# Trained TF-IDF vectorizer
├── data/
│   └── spam_ham_dataset.csv# Dataset used for training
├── notebook/
│   └── ml_mini_porject.ipynb # Jupyter notebook for model development
└── README.md               # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo.git
   cd spam_Classifier
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv env
   .\env\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Model Training
- The model was trained using the dataset in `data/spam_ham_dataset.csv`.
- Training and evaluation code is available in the Jupyter notebook under `notebook/`.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) for the dataset
- Streamlit for the web app framework

---
Feel free to contribute or raise issues!
