# Movie-Reviews-Sentiment-Analysis
# IMDb Sentiment Classifier

This project implements a machine learning pipeline using scikit-learn to classify movie reviews as positive or negative, as required for the Python AI Engineer job posting. The pipeline includes exploratory data analysis (EDA), TF-IDF vectorization, and a Logistic Regression classifier. A Streamlit web application is included for interactive predictions.

## Project Overview
- **Goal**: Build a Python ML pipeline to classify movie reviews as positive or negative.
- **Dataset**: A subset of IMDb movie reviews stored as `datasets/Movie_Review.csv` with `text` (review content) and `sentiment` (pos/neg) columns.
- **Model**: Uses scikit-learn's `TfidfVectorizer` (max_features=2500) and `LogisticRegression` for classification.
- **Scripts**:
  - `EDA.ipynb`: Jupyter notebook for data exploration, preprocessing, and model training.
  - `predict.py`: Command-line script for predicting sentiment (to be implemented if not present).
  - `app.py`: Streamlit app for interactive predictions.
- **Deployment**: The Streamlit app is deployed at: [https://hermelawesene-movie-reviews-sentiment-analysis-mainapp-sfivgi.streamlit.app/](https://hermelawesene-movie-reviews-sentiment-analysis-mainapp-sfivgi.streamlit.app/).

## Setup
1. **Clone the Repository** :
   ```bash
   git clone https://github.com/hermelawesene/Movie-Reviews-Sentiment-Analysis.git
   cd Movie-Reviews-Sentiment-Analysis\main
   ```
2. **Create Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate   
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK Resources**

```python
import nltk
nltk.download('stopwords')
```

---

## Dataset

- Path: `datasets/Movie_Review.csv`
- Columns:
  - `text`: Movie review content
  - `sentiment`: Label ('positive' or 'negative')

---

## EDA & Model Training (`EDA.ipynb`)

- Preprocessing includes:
  - Lowercasing, removing stopwords, punctuation cleanup
  - Word cloud visualizations
- Text vectorized using `TfidfVectorizer(max_features=2500)`
- Sentiment encoded: `positive` → 1, `negative` → 0
- Model: `LogisticRegression` trained with 80-20 split
- Output:
  - Accuracy & Confusion Matrix
  - Saves model to `models/model.pkl`
  - Saves vectorizer to `models/scaler.pkl`

---

## Command-line Predictions

File: `predict.py`  
Usage:

```bash
python predict.py "This movie was fantastic!"
```

**Output**:
```
Prediction: positive (Confidence: 0.5138)
```

---

## Streamlit Web App

1. **Run Locally**:

```bash
streamlit run app.py
```

2. **Visit in Browser**:  
Open [http://localhost:8501](http://localhost:8501)

3. **Live Deployment**:  
[https://hermelawesene-movie-reviews-sentiment-analysis-mainapp-sfivgi.streamlit.app/](https://hermelawesene-movie-reviews-sentiment-analysis-mainapp-sfivgi.streamlit.app/)

---

## Requirements

File: `requirements.txt`

```
pandas
numpy
matplotlib
streamlit
scikit-learn
nltk
wordcloud
```

Install using:

```bash
pip install -r requirements.txt
```


## 📌 Notes

- Accuracy ~80-85% on validation set
- Logistic Regression meets the requirement of “simple, interpretable ML model”
- Streamlit app provides optional interactive demo
- Project is easily extendable to use other models (e.g., Naive Bayes, SVM)

---

## 📬 Contact

For questions, issues, or support, contact:  
**Hermela Wesene** — [hermelawesene@gmail.com]
