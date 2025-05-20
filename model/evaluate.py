from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    preds = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, preds, target_names=['Ne', 'Taip']))

