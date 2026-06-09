# Fraud Detection — Decision Tree and Random 
Fraud detection classifier using a custom Decision Tree and Random Forest built from scratch in Python, trained on the IEEE-CIS Fraud Detection dataset from Kaggle.
## How to Run
1. Create a virtual environment: python -m venv venv
2. Activate it: venv\Scripts\activate (Windows) or source venv/bin/activate (Mac/Linux)
3. Install dependencies: pip install -r requirements.txt
4. Download the dataset from Kaggle and place train.csv and test.csv inside a data/ folder
5. Run: python classifier.py

## Results

| Model | Accuracy | Balanced Accuracy |
|---|---|---|
| Decision Tree | 96.82% | 56.77% |
| Random Forest | **97.02%** | **59.03%** |
