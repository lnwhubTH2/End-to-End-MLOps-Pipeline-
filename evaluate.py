import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import sys
import os
import shutil

def evaluate():
    print("--- 🎯 Starting Model Evaluation ---")
    # 1. โหลดข้อมูลทดสอบ (ข้อสอบชุดใหม่)
    df = pd.read_csv('data/iris_new.csv')
    X = df.drop('target', axis=1)
    y_true = df['target']

    # 2. ดึงโมเดลเดิมในระบบ (Champion) มาลองทำข้อสอบ
    champion_acc = 0.0
    if os.path.exists('model/champion_model.pkl'):
        champion = joblib.load('model/champion_model.pkl')
        y_pred_champ = champion.predict(X)
        champion_acc = accuracy_score(y_true, y_pred_champ)
        print(f"🥇 โมเดลระบบเดิม (Champion) ความแม่นยำ: {champion_acc*100:.2f}%")
    else:
        print("⚠️ ไม่พบโมเดลเดิมในระบบ เริ่มต้นความแม่นยำที่ 0%")

    # 3. ดึงโมเดลตัวใหม่ (Challenger) มาลองทำข้อสอบแข่ง
    challenger = joblib.load('model/challenger_model.pkl')
    y_pred_challenger = challenger.predict(X)
    challenger_acc = accuracy_score(y_true, y_pred_challenger)
    print(f"🥊 โมเดลที่พึ่งเทรน (Challenger) ความแม่นยำ: {challenger_acc*100:.2f}%")

    # 4. ตัดสินผล: ถ้าเด็กใหม่เก่งกว่าเดิม
    if challenger_acc > champion_acc:
        print("\n🏆 RESULT: CHALLENGER WINS! (โมเดลใหม่ชนะ! อนุมัติการนำขึ้นระบบ...)")
        # โปรโมทโมเดลใหม่ขึ้นเป็นแชมป์แทนที่ตัวเก่า
        shutil.copy('model/challenger_model.pkl', 'model/champion_model.pkl')
        sys.exit(0) 
    else:
        print("\n❌ RESULT: CHAMPION STAYS. (โมเดลใหม่โง่กว่าเดิม... หยุดระบบ Deployment ทันที)")
        sys.exit(1) 

if __name__ == "__main__":
    evaluate()