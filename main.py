from src.preprocess import load_data
from src.model import train_model

df = load_data()
acc = train_model(df)
print(f"Accuracy: {acc:.2f}")