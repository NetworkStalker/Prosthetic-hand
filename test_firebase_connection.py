"""
test_firebase_connection.py
สคริปต์ทดสอบว่า FIREBASE_CONFIG ใน firebase_sync.py ใส่ค่าถูกต้องและเชื่อมต่อได้จริง
รันบน Windows ได้เลย (ไม่ต้องมี Hardware) — เอาไว้เช็คก่อนเอาไปรันจริงบน Raspberry Pi

วิธีใช้:
    1. แก้ FIREBASE_CONFIG ใน firebase_sync.py ให้เป็นค่าจริงจาก Firebase Console ก่อน
    2. pip install pyrebase4
    3. python test_firebase_connection.py
"""

from firebase_sync import FIREBASE_CONFIG, PYREBASE_AVAILABLE

print("=" * 60)
print("ทดสอบการเชื่อมต่อ Firebase")
print("=" * 60)

if not PYREBASE_AVAILABLE:
    print("❌ ไม่ได้ติดตั้ง pyrebase4 — รัน: pip install pyrebase4")
    raise SystemExit(1)

if "YOUR_" in FIREBASE_CONFIG.get("apiKey", "YOUR_"):
    print("❌ ยังไม่ได้แก้ FIREBASE_CONFIG ในไฟล์ firebase_sync.py")
    print("   ให้ใส่ค่าจริงจาก Firebase Console ก่อน (Project Settings > General > Web app)")
    raise SystemExit(1)

import pyrebase

try:
    app = pyrebase.initialize_app(FIREBASE_CONFIG)
    db = app.database()
    print("✅ initialize_app สำเร็จ — กำลังทดสอบเขียน/อ่านข้อมูลจริง...")

    test_payload = {"message": "Hello จาก test_firebase_connection.py", "ok": True}
    db.child("connection_test").set(test_payload)
    print("✅ เขียนข้อมูลทดสอบขึ้น Firebase สำเร็จ (node: connection_test)")

    result = db.child("connection_test").get()
    print("✅ อ่านข้อมูลกลับมาได้:", result.val())

    print("\nไปดูใน Firebase Console → Realtime Database ควรเห็น node 'connection_test'")
    print("ถ้าเห็นแปลว่า config ถูกต้องแล้ว พร้อมเอาไปรันบน Raspberry Pi ได้เลย")

except Exception as e:
    print(f"❌ เชื่อมต่อไม่สำเร็จ: {e}")
    print("\nสาเหตุที่พบบ่อย:")
    print("  1. databaseURL ผิด (ต้องเป็น Realtime Database ไม่ใช่ Firestore)")
    print("  2. Realtime Database ยังไม่ได้ Create Database ใน Console")
    print("  3. Rules ปิด read/write ไว้ (เช็คแท็บ Rules ต้องเป็น true ตอน dev)")
    print("  4. ถ้า error เกี่ยวกับ requests_toolbelt/MultipartEncoder ให้รัน:")
    print("     pip install requests_toolbelt==0.9.1   (บั๊กรู้จักกันของ pyrebase4)")
