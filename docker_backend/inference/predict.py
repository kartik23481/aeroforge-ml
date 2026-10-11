# backend/inference/predict.py

from .model_loader import MODEL, COLUMN_TRANSFORMER
from .preprocess import preprocess_input

def predict_price(user_input: dict) -> float:
    """
    Returns predicted flight price
    """
    df = preprocess_input(user_input)
    transformed = COLUMN_TRANSFORMER.transform(df)
    prediction = MODEL.predict(transformed)[0]
    return float(prediction)
  