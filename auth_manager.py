"""
auth_manager.py
ระบบ Login ผ่าน Firebase Authentication (อีเมล/รหัสผ่าน) + ตรวจสิทธิ์ Admin
จากรายชื่อที่เก็บไว้ใน Realtime Database (node "admins")

ทำไมต้องมี 2 ชั้น (Firebase Auth + Admin whitelist):
  - Firebase Authentication ยืนยันว่า "อีเมล/รหัสผ่านถูกต้องจริง" — Firebase เช็ครหัสผ่าน
    ให้เองฝั่งเซิร์ฟเวอร์ เราไม่ต้องเก็บ/เทียบรหัสผ่านเองเลย ปลอดภัยกว่าการทำเองมาก
  - แต่แค่มีบัญชี Firebase ไม่ได้แปลว่ามีสิทธิ์ใช้แอปนี้ — ต้องเช็คอีกชั้นว่าอีเมลนั้น
    อยู่ใน node "admins" ที่ตั้งไว้ใน Realtime Database ด้วย ถึงจะเข้าใช้งานโปรแกรมได้จริง

ก่อนใช้งานจริง ต้องตั้งค่าใน Firebase Console ก่อน:
  1. Authentication → Sign-in method → เปิดใช้งาน "Email/Password"
  2. รัน setup_admin.py เพื่อสร้างบัญชี Admin คนแรก (สร้างทั้งบัญชี Auth และเพิ่มชื่อ
     เข้า node "admins" ให้อัตโนมัติในขั้นตอนเดียว) — หรือให้ผู้ใช้สมัครเองผ่านหน้า Login
     (ดู signup() ด้านล่าง) ก็ได้เหมือนกัน ทำสิ่งเดียวกันแค่ผ่าน GUI แทน CLI

⚠️ signup()/self-signup: ตั้งใจให้ "สมัครเสร็จ = ใช้งานได้ทันที" (เพิ่มเข้า node "admins"
ให้อัตโนมัติ ไม่ต้องรอ Admin คนอื่นอนุมัติ) เหมาะกับเครื่องมือเทียมส่วนตัว/ทดลองที่ไม่ได้
จำกัดว่าใครสมัครได้บ้าง — ถ้าเอาไปใช้ในบริบทที่ต้องคุมสิทธิ์เข้าถึงมอเตอร์/servo เข้มงวดกว่านี้
(เช่น มีหลายคนใช้เครื่องเดียวกันจริง) ควรเปลี่ยน signup() ให้ "สมัครแล้วรอ Admin อนุมัติ" แทน
(สร้างบัญชี Auth แต่ไม่เพิ่มเข้า node "admins" ทันที ให้ Admin มาเพิ่มเองทีหลัง)

โครงสร้างข้อมูลใน Realtime Database:
    /admins: ["someone@example.com", "another@example.com"]
    (เก็บเป็น array ของอีเมลตรงๆ แก้ไข/เพิ่มคนได้ตรงใน Firebase Console เลย)
"""

import re

from firebase_sync import FIREBASE_CONFIG, PYREBASE_AVAILABLE

ADMIN_LIST_PATH = "admins"

# error code จาก Firebase Auth (ภาษาอังกฤษ) -> ข้อความไทยที่อ่านง่าย
_ERROR_MESSAGES = {
    "EMAIL_NOT_FOUND": "ไม่พบบัญชีอีเมลนี้ในระบบ",
    "INVALID_PASSWORD": "รหัสผ่านไม่ถูกต้อง",
    "INVALID_LOGIN_CREDENTIALS": "อีเมลหรือรหัสผ่านไม่ถูกต้อง",
    "USER_DISABLED": "บัญชีนี้ถูกระงับการใช้งาน",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "ลองผิดหลายครั้งเกินไป กรุณารอสักครู่แล้วลองใหม่",
    "MISSING_PASSWORD": "กรุณาใส่รหัสผ่าน",
    "INVALID_EMAIL": "รูปแบบอีเมลไม่ถูกต้อง",
    # error code เพิ่มเติมที่เจอเฉพาะตอนสมัครสมาชิก (signup) / ขอรีเซ็ตรหัสผ่าน
    "EMAIL_EXISTS": "มีบัญชีอีเมลนี้อยู่แล้ว — ลองเข้าสู่ระบบแทน หรือกดลืมรหัสผ่านถ้าจำรหัสไม่ได้",
    "WEAK_PASSWORD": "รหัสผ่านสั้นเกินไป ต้องมีอย่างน้อย 6 ตัวอักษร",
    "OPERATION_NOT_ALLOWED": "ระบบยังไม่ได้เปิดใช้งานการสมัครด้วยอีเมล/รหัสผ่าน "
                              "(ผู้ดูแลระบบต้องเปิดที่ Firebase Console ก่อน)",
}


def _parse_firebase_error(exc: Exception) -> str:
    """ดึง error code จาก exception ของ pyrebase แล้วแปลเป็นข้อความไทยที่อ่านง่าย"""
    text = str(exc)
    for code, thai_msg in _ERROR_MESSAGES.items():
        if code in text:
            return thai_msg
    m = re.search(r'"message"\s*:\s*"([A-Z_]+)"', text)
    if m:
        return f"เข้าสู่ระบบไม่สำเร็จ: {m.group(1)}"
    return f"เข้าสู่ระบบไม่สำเร็จ: {text[:120]}"


def _normalize_admin_list(raw) -> list:
    """Firebase คืน list ธรรมดาถ้าเก็บเป็น array แต่บางทีก็เป็น dict {key: email}
    ถ้า index ไม่ต่อเนื่อง — รองรับทั้งสองแบบ และกัน None ที่หลุดมาจาก sparse array"""
    if raw is None:
        return []
    values = list(raw.values()) if isinstance(raw, dict) else list(raw)
    return [v for v in values if v]


class AuthManager:
    def __init__(self):
        self.available = PYREBASE_AVAILABLE
        self.auth = None
        self.db = None
        if self.available:
            try:
                import pyrebase
                app = pyrebase.initialize_app(FIREBASE_CONFIG)
                self.auth = app.auth()
                self.db = app.database()
            except Exception as e:
                self.available = False
                print(f"[AuthManager] เชื่อมต่อ Firebase ไม่สำเร็จ: {e}")

    def login(self, email: str, password: str):
        """พยายามล็อกอิน คืนค่า (success: bool, message: str, user_info: dict|None)"""
        email = (email or "").strip()
        password = password or ""

        if not self.available:
            return False, "ไม่ได้ติดตั้ง pyrebase4 หรือเชื่อมต่อ Firebase ไม่ได้ — รัน: pip install pyrebase4", None
        if not email or not password:
            return False, "กรุณาใส่อีเมลและรหัสผ่านให้ครบ", None

        try:
            user = self.auth.sign_in_with_email_and_password(email, password)
        except Exception as e:
            return False, _parse_firebase_error(e), None

        # ล็อกอินผ่านแล้ว (รหัสผ่านถูกต้องจริง ยืนยันจาก Firebase) — เช็คต่อว่าอยู่ใน
        # whitelist admin ไหม ถึงจะอนุญาตให้เข้าใช้งานโปรแกรมนี้ได้
        try:
            raw_admins = self.db.child(ADMIN_LIST_PATH).get().val()
        except Exception as e:
            return False, f"เข้าสู่ระบบสำเร็จ แต่ตรวจสอบสิทธิ์ Admin ไม่ได้: {e}", None

        admin_emails = _normalize_admin_list(raw_admins)

        if email not in admin_emails:
            return False, f"บัญชี {email} เข้าสู่ระบบได้ แต่ไม่มีสิทธิ์ Admin ใช้งานโปรแกรมนี้\n" \
                          f"(ให้ Admin ที่มีสิทธิ์อยู่แล้วเพิ่มอีเมลนี้ในไฟล์ Firebase Console " \
                          f"หรือรัน setup_admin.py)", None

        return True, "เข้าสู่ระบบสำเร็จ", {
            "email": email,
            "localId": user.get("localId"),
            "idToken": user.get("idToken"),
        }

    def signup(self, email: str, password: str, confirm_password: str = None):
        """สมัครสมาชิกใหม่ผ่าน Firebase Authentication แล้วเพิ่มเข้ารายชื่อ Admin
        ("admins") ให้อัตโนมัติทันที — self-signup: สมัครเสร็จ = ใช้งานได้ทันที ไม่ต้องรอ
        Admin คนอื่นมาอนุมัติ (เป็น flow เดียวกับที่ setup_admin.py ทำผ่าน CLI แค่ทำผ่าน
        GUI แทน — ดู module docstring ด้านบนถ้าอยากเปลี่ยนเป็นแบบ "รอ Admin อนุมัติ")

        confirm_password: ถ้าส่งมา (ไม่ใช่ None) จะเช็คว่าตรงกับ password ไหมก่อน — ปล่อยเป็น
        None ได้ถ้าฝั่งเรียกไม่มีช่องยืนยันรหัสผ่าน (เช็คแค่ email/password พอ)

        คืนค่า (success: bool, message: str, user_info: dict|None) รูปแบบเดียวกับ login()
        เพื่อให้ฝั่ง GUI ใช้โค้ดจัดการผลลัพธ์ร่วมกันได้"""
        email = (email or "").strip()
        password = password or ""

        if not self.available:
            return False, "ไม่ได้ติดตั้ง pyrebase4 หรือเชื่อมต่อ Firebase ไม่ได้ — รัน: pip install pyrebase4", None
        if not email or not password:
            return False, "กรุณาใส่อีเมลและรหัสผ่านให้ครบ", None
        if confirm_password is not None and password != confirm_password:
            return False, "รหัสผ่านและยืนยันรหัสผ่านไม่ตรงกัน", None
        if len(password) < 6:
            return False, "รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร (ข้อกำหนดของ Firebase Authentication)", None

        try:
            user = self.auth.create_user_with_email_and_password(email, password)
        except Exception as e:
            return False, _parse_firebase_error(e), None

        # สมัครบัญชี Auth สำเร็จแล้ว → เพิ่มเข้ารายชื่อ admins ให้อัตโนมัติทันที (self-signup)
        # กันเพิ่มซ้ำด้วย _normalize_admin_list เหมือน setup_admin.py
        try:
            current_list = _normalize_admin_list(self.db.child(ADMIN_LIST_PATH).get().val())
            if email not in current_list:
                current_list.append(email)
                self.db.child(ADMIN_LIST_PATH).set(current_list)
        except Exception as e:
            # บัญชี Auth สร้างสำเร็จแล้วจริง (login ได้แล้ว) แค่เพิ่มสิทธิ์ admin ไม่สำเร็จ —
            # แจ้งตรงๆ แทนที่จะทำเหมือนสมัครไม่สำเร็จทั้งหมด กันสับสน/สมัครซ้ำแล้วเจอ EMAIL_EXISTS
            return False, (f"สมัครสมาชิกสำเร็จ แต่เพิ่มสิทธิ์ Admin ไม่สำเร็จ: {e}\n"
                           f"ลองเข้าสู่ระบบใหม่อีกครั้ง หรือติดต่อผู้ดูแลระบบให้เพิ่ม {email} "
                           f"เข้า node \"admins\" ใน Firebase Console เอง"), None

        return True, "สมัครสมาชิกสำเร็จ — เข้าใช้งานได้ทันที", {
            "email": email,
            "localId": user.get("localId"),
            "idToken": user.get("idToken"),
        }

    def send_password_reset(self, email: str):
        """ส่งอีเมลลิงก์รีเซ็ตรหัสผ่านผ่าน Firebase Authentication — ผู้ใช้กดลิงก์ในอีเมล
        แล้วตั้งรหัสผ่านใหม่เอง ฝั่งแอปนี้ไม่เคยเห็น/เก็บรหัสผ่านใหม่เลย (Firebase จัดการ
        ทั้งหมดฝั่งเซิร์ฟเวอร์) คืนค่า (success: bool, message: str)"""
        email = (email or "").strip()

        if not self.available:
            return False, "ไม่ได้ติดตั้ง pyrebase4 หรือเชื่อมต่อ Firebase ไม่ได้ — รัน: pip install pyrebase4"
        if not email:
            return False, "กรุณาใส่อีเมล"

        try:
            self.auth.send_password_reset_email(email)
        except Exception as e:
            return False, _parse_firebase_error(e)

        return True, f"ส่งอีเมลรีเซ็ตรหัสผ่านไปที่ {email} แล้ว — เช็คกล่องจดหมาย (รวมถึง Spam/Junk mail)"