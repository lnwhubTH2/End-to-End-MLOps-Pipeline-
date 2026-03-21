# End-to-End MLOps Pipeline 🚀

ยินดีต้อนรับสู่โปรเจกต์ MLOps แบบครบวงจร! โปรเจกต์นี้สาธิตการทำงานร่วมกันระหว่าง **Continuous Training (CT)** และ **Model Deployment (API / Docker)** แบบอัตโนมัติ 

เมื่อมีข้อมูลใหม่เข้ามา ระบบจะเทรนโมเดล วัดผลความแม่นยำ ทดสอบ API (Unit Test) และสร้าง Docker Image ให้พร้อมใช้งานได้ทันที

## 📂 Project Structure

```text
LAB_CT/
├── .github/workflows/
│   ├── ct_pipeline.yml  # หุ่นยนต์ตัวที่ 1: งานฝั่งเทรนโมเดล (Continuous Training)
│   └── ci_cd_pipeline.yml # หุ่นยนต์ตัวที่ 2: งานตรวจคุณภาพโค้ดและแพ็กลง Docker (CI/CD)
├── data/
│   └── iris_new.csv     # ข้อมูลดิบสำหรับนำไปสอนโมเดล Iris
├── app/
│   └── main.py          # โค้ด FastAPI สำหรับเปิดให้บริการโมเดลผ่าน API
├── tests/
│   └── test_app.py      # ไฟล์ข้อสอบ Unit Test สำหรับทดสอบการทำงานของ API
├── train.py             # สคริปต์ฝึกสอน Decision Tree (สร้าง challenger_model.pkl)
├── evaluate.py          # สคริปต์ตรวจข้อสอบโมเดล (Champion vs. Challenger)
├── Dockerfile           # ใบสั่งประกอบร่าง Docker Image สำหรับ API
└── requirements.txt     # รายชื่อ Library ที่ใช้ทั้งหมด (pandas, scikit-learn, fastapi, ฯลฯ)
```

## 🛠️ การใช้งานระบบบนเครื่อง Local

1. **เตรียมสภาพแวดล้อม (Virtual Environment):**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **จำลองการทำงาน Continuous Training สั่งโมเดลเรียนรู้และวัดผล:**  
   ```bash
   python train.py
   python evaluate.py
   ```
   *(ผลลัพธ์: จะได้ไฟล์ `challenger_model.pkl` หรือโมเดลตัวใหม่ที่พร้อมลุยงาน)*

3. **ทดสอบระบบ API (Unit Test) ให้อุ่นใจ:**  
   ```bash
   PYTHONPATH=. pytest tests/
   ```

4. **เปิดให้บริการ API จำลอง (FastAPI & Uvicorn):**  
   ```bash
   uvicorn app.main:app --reload
   ```
   *เปิดเบราว์เซอร์ไปที่ `http://127.0.0.1:8000/docs` เพื่อทดสอบยิงข้อมูลแบบ JSON ได้เลย*

5. **จำลองการห่อโปรเจกต์ลงกล่องขึ้น Server (Dockerize):**  
   ```bash
   docker build -t iris-ml-api:latest .
   docker run -p 8000:8000 iris-ml-api:latest
   ```

---
**💡 ทิปส์การนำขึ้น GitHub:** 
ทุกครั้งที่คุณแก้โค้ด พัฒนาโมเดล หรือได้ข้อมูล `iris_new.csv` ชุดใหม่ คุณสามารถใช้คำสั่ง:
`git add .` -> `git commit -m "update data"` -> `git push`
เเล้วไปจิบกาแฟรอให้ **GitHub Actions** ทำงานข้อ 2-5 อัตโนมัติบน Cloud ได้เลยครับ! ☕
