import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

def train():
    print("--- Starting Training Process ---")
    # 1. อ่านข้อมูลจากโฟลเดอร์ data 
    df = pd.read_csv('data/iris_new.csv')
    print(f"📊 Dataset loaded: {len(df)} rows.")
    X = df.drop('target', axis=1)
    y = df['target']

    # 2. สร้างตัวเทรน Decision Tree [cite: 132]
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # 3. บันทึกโมเดลใหม่เป็นไฟล์ในโฟลเดอร์ model
    import os
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/challenger_model.pkl')
    print("Challenger Model trained and saved successfully!")

if __name__ == "__main__":
    train()