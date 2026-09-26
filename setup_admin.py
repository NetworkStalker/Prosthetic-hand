"""
setup_admin.py
สคริปต์ตั้งค่าเริ่มต้น — สร้างบัญชี Admin สำหรับระบบ Login
(สร้างบัญชีใน Firebase Authentication + เพิ่มอีเมลเข้า node "admins" ใน Realtime Database
ในขั้นตอนเดียว)

รันครั้งแรกตอนตั้งระบบ Login (หรือรันซ้ำอีกเพื่อเพิ่ม Admin คนใหม่ทีหลังก็ได้ — เช็คให้
อัตโนมัติว่าบัญชีมีอยู่แล้วหรือยัง ไม่สร้างซ้ำ)

ก่อนรัน ต้องเปิดใช้งาน Email/Password sign-in ใน Firebase Console ก่อน:
    Authentication → Sign-in method → Email/Password → Enable

รัน:
    python3 setup_admin.py
"""

from firebase_sync import FIREBASE_CONFIG, PYREBASE_AVAILABLE
from auth_manager import ADMIN_LIST_PATH, _normalize_admin_list


def main():
    if not PYREBASE_AVAILABLE:
        print("❌ ไม่ได้ติดตั้ง pyrebase4 — รัน: pip install pyrebase4")
        return
    if "YOUR_" in FIREBASE_CONFIG.get("apiKey", "YOUR_"):
        print("❌ ยังไม่ได้แก้ FIREBASE_CONFIG ในไฟล์ firebase_sync.py ให้เป็นค่าจริงก่อน")
        return

    import pyrebase
    app = pyrebase.initialize_app(FIREBASE_CONFIG)
    auth = app.auth()
    db = app.database()

    print("=== ตั้งค่า Admin สำหรับระบบ Login ===")
    email = input("อีเมล Admin: ").strip()
    password = input("รหัสผ่าน (อย่างน้อย 6 ตัวอักษร): ").strip()

    if len(password) < 6:
        print("❌ รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร (ข้อกำหนดของ Firebase Authentication)")
        return

    # เช็คก่อนว่ามีบัญชีนี้อยู่แล้วหรือยัง (ล็อกอินด้วยรหัสผ่านที่กรอกมา) ถ้ายังไม่มีค่อยสร้างใหม่
    try:
        auth.sign_in_with_email_and_password(email, password)
        print(f"ℹ️  บัญชี {email} มีอยู่แล้ว (รหัสผ่านตรงกับที่ใส่) — ข้ามขั้นตอนสร้างบัญชีใหม่")
    except Exception:
        try:
            auth.create_user_with_email_and_password(email, password)
            print(f"✅ สร้างบัญชี Firebase Authentication สำหรับ {email} สำเร็จ")
        except Exception as e:
            print(f"❌ สร้างบัญชีไม่สำเร็จ: {e}")
            print("   (ถ้า error บอกว่า EMAIL_EXISTS แปลว่ามีบัญชีนี้แล้วแต่รหัสผ่านที่กรอกไม่ตรง)")
            return

    # เพิ่มเข้ารายชื่อ admins (เก็บเป็น array ของอีเมล กันเพิ่มซ้ำ)
    try:
        current_list = _normalize_admin_list(db.child(ADMIN_LIST_PATH).get().val())
        if email not in current_list:
            current_list.append(email)
            db.child(ADMIN_LIST_PATH).set(current_list)
            print(f"✅ เพิ่ม {email} เข้ารายชื่อ Admin แล้ว")
        else:
            print(f"ℹ️  {email} อยู่ในรายชื่อ Admin อยู่แล้ว")
    except Exception as e:
        print(f"❌ อัพเดตรายชื่อ Admin ไม่สำเร็จ: {e}")
        return

    print("\n🎉 ตั้งค่าเสร็จสมบูรณ์ — ใช้อีเมลนี้ล็อกอินเข้าโปรแกรมได้เลย")


if __name__ == "__main__":
    main()
