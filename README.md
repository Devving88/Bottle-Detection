# Water Bottle Detection (Streamlit)

ระบบตรวจจับขวดน้ำด้วย YOLO

## โครงสร้างโฟลเดอร์

```
streamlit_app/
├── app.py
├── requirements.txt
├── model/
│   └── Yolo8n.pt        <- โมเดลที่เทรนไว้แล้ว
├── upload/               <- ไฟล์วิดีโอที่อัปโหลดจะถูกเก็บที่นี่อัตโนมัติ
└── .streamlit/
    └── config.toml       <- ธีมสีดำ-น้ำเงิน (ไม่ใช้ CSS)
```

## วิธีติดตั้งและรัน

```bash
pip install -r requirements.txt
streamlit run app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่ `http://localhost:8501`

## ฟีเจอร์

- **แท็บอัปโหลดวิดีโอ**: อัปโหลดไฟล์วิดีโอ (mp4/avi/mov/mkv) ระบบจะบันทึกไว้ในโฟลเดอร์ `upload/` แล้วรันตรวจจับ+ติดตาม object พร้อมแสดงผลแบบเฟรมต่อเฟรม พร้อม progress bar
- **แท็บเว็บแคมเรียลไทม์**: เปิดกล้องเว็บแคมผ่าน `streamlit-webrtc` (⚠️ **ใช้ได้เฉพาะบน Localhost เท่านั้น** เนื่องจากข้อจำกัดการเข้าถึงกล้องและ HTTPS บน Cloud เซิร์ฟเวอร์)
- ปรับ **confidence threshold** ได้อิสระในแต่ละแท็บ
- ธีมสีดำ-น้ำเงินตั้งค่าผ่าน `.streamlit/config.toml` ล้วนๆ ไม่มีการฝัง CSS ใดๆ ในหน้าเว็บ

## หมายเหตุ

- ฟีเจอร์เว็บแคมเรียลไทม์ (`streamlit-webrtc`) จำเป็นต้องรันบนเครื่อง Localhost (`http://localhost:8501`) เพื่อให้เบราว์เซอร์อนุญาตให้เข้าถึงกล้องเว็บแคมได้
