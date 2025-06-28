import sys
import pickle as pk

model = pk.load(open('models/model.pkl', 'rb'))
cv = pk.load(open('models/scaler.pkl', 'rb'))

if len(sys.argv) < 2:
    print("Usage: python predict.py 'Your review text'")
    sys.exit(1)

review_text = sys.argv[1]
X = cv.transform([review_text])
prediction = model.predict(X)[0]
confidence = model.predict_proba(X)[0][prediction]
label = 'positive' if prediction == 1 else 'negative'
print(f'Prediction: {label} (Confidence: {confidence:.4f})')