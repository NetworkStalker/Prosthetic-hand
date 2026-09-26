"""
firebase_config.example.py
เทมเพลตไฟล์ตั้งค่า Firebase — ให้คัดลอกไฟล์นี้เป็น "firebase_config.py" (ชื่อไฟล์ต้องตรงเป๊ะ)
แล้วใส่ค่าจริงของโปรเจกต์ตัวเองแทนค่า placeholder ด้านล่าง

วิธีทำ:
    cp firebase_config.example.py firebase_config.py
    (แล้วแก้ค่าใน firebase_config.py ให้เป็นค่าจริงจาก Firebase Console)

ค่าจริงหาได้จาก Firebase Console → เลือกโปรเจกต์ → ไอคอนฟันเฟือง (Project settings)
→ แท็บ General → เลื่อนลงไปที่ "Your apps" → เลือกแอปแบบ Web (</>) → SDK setup and
configuration → เลือก "Config"

⚠️ ห้าม commit ไฟล์ "firebase_config.py" (ไฟล์ที่มีค่าจริง) ขึ้น Git — ไฟล์นี้ถูกใส่ไว้ใน
.gitignore แล้ว มีแค่ไฟล์ตัวอย่างนี้ (.example.py) เท่านั้นที่ควรอยู่ใน Git repo
"""

FIREBASE_CONFIG = {
    "apiKey":            "YOUR_API_KEY",
    "authDomain":        "YOUR_PROJECT_ID.firebaseapp.com",
    "databaseURL":       "https://YOUR_PROJECT_ID-default-rtdb.YOUR_REGION.firebasedatabase.app/",
    "projectId":         "YOUR_PROJECT_ID",
    "storageBucket":     "YOUR_PROJECT_ID.firebasestorage.app",
    "messagingSenderId": "YOUR_SENDER_ID",
    "appId":             "YOUR_APP_ID",
}
