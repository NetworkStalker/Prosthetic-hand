"""
login_window.py
หน้าต่าง Login ก่อนเข้าโปรแกรมหลัก — ยืนยันตัวตนผ่าน Firebase Authentication
แล้วเช็คว่าอีเมลนั้นอยู่ในรายชื่อ Admin ใน Realtime Database ไหม (ดู auth_manager.py)

หน้าต่างเดียวนี้มี 2 โหมดสลับกันได้ (ดู _set_mode):
  - "login"  (ค่าเริ่มต้น) — เข้าสู่ระบบด้วยอีเมล/รหัสผ่านที่มีอยู่แล้ว
  - "signup" — สมัครสมาชิกใหม่ (self-signup: สมัครเสร็จ = ใช้งานได้ทันที ดู
    auth_manager.AuthManager.signup() สำหรับรายละเอียด/ข้อควรระวัง)
สลับโหมดด้วยการโชว์/ซ่อน widget เดิม ไม่ได้สร้างหน้าต่างใหม่ — เก็บ state (เช่นอีเมลที่
พิมพ์ไปแล้ว) ไว้ให้ระหว่างสลับโหมดด้วย

ปุ่ม "ลืมรหัสผ่าน?" เปิด ForgotPasswordDialog แยกต่างหาก (Toplevel แบบ modal) เพราะเป็น
flow สั้นๆ ที่ไม่เกี่ยวกับ state ของฟอร์ม login/signup หลักเลย (แค่กรอกอีเมล → Firebase
ส่งลิงก์รีเซ็ตรหัสผ่านไปให้ทางอีเมล)
"""

import tkinter as tk
import tkinter.font as tkfont

from auth_manager import AuthManager

COLORS = {
    "bg": "#0f1729", "surface": "#1a2540", "card": "#1e2a4a",
    "accent": "#6366f1", "text": "#e2e8f0", "text_dim": "#94a3b8",
    "danger": "#ef4444", "success": "#22c55e",
}
FONT = ("TH Sarabun New", 13)
FONT_B = ("TH Sarabun New", 13, "bold")


class ForgotPasswordDialog(tk.Toplevel):
    """หน้าต่างเล็กๆ แยกต่างหาก (modal) สำหรับขออีเมลรีเซ็ตรหัสผ่าน — เปิดจากลิงก์
    "ลืมรหัสผ่าน?" ในหน้า Login หลัก ไม่ปนกับฟอร์ม login/signup หลักเพื่อไม่ให้ state
    ของฟอร์มหลักยุ่งเหยิง ปิดหน้าต่างนี้แล้วกลับไปหน้า Login หลักได้ตามปกติเสมอ"""

    def __init__(self, parent, auth_manager: AuthManager, prefill_email: str = ""):
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.title("ลืมรหัสผ่าน")
        self.geometry("340x240")  # ค่าเริ่มต้นชั่วคราว — ตั้งขนาดจริงท้าย __init__ ด้านล่าง
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])
        self.transient(parent)
        self.grab_set()  # modal — บล็อกไม่ให้กดหน้าต่าง Login หลักจนกว่าจะปิดอันนี้

        tk.Label(self, text="🔑", font=("TH Sarabun New", 28), bg=COLORS["bg"],
                 fg=COLORS["accent"]).pack(pady=(18, 4))
        tk.Label(self, text="รีเซ็ตรหัสผ่าน", font=FONT_B, bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(pady=(0, 4))
        tk.Label(self, text="กรอกอีเมลที่ใช้สมัคร ระบบจะส่งลิงก์ตั้งรหัสผ่านใหม่ไปให้ทางอีเมล",
                 font=("TH Sarabun New", 10), bg=COLORS["bg"], fg=COLORS["text_dim"],
                 wraplength=300, justify="center").pack(pady=(0, 10))

        form = tk.Frame(self, bg=COLORS["bg"])
        form.pack(fill="x", padx=24)
        self.email_var = tk.StringVar(value=prefill_email)
        self.email_entry = tk.Entry(form, textvariable=self.email_var, font=FONT,
                                     bg=COLORS["card"], fg=COLORS["text"],
                                     insertbackground=COLORS["text"], relief="flat")
        self.email_entry.pack(fill="x", ipady=6)
        self.email_entry.bind("<Return>", lambda e: self._send())

        self.status_lbl = tk.Label(self, text="", font=("TH Sarabun New", 10),
                                    bg=COLORS["bg"], fg=COLORS["danger"], wraplength=300,
                                    justify="left")
        self.status_lbl.pack(padx=24, pady=(6, 4), anchor="w")

        self.btn_send = tk.Button(self, text="ส่งอีเมลรีเซ็ตรหัสผ่าน", font=FONT_B,
                                   bg=COLORS["accent"], fg="white", relief="flat",
                                   activebackground=COLORS["accent"], cursor="hand2",
                                   command=self._send)
        self.btn_send.pack(fill="x", padx=24, ipady=6, pady=(4, 4))

        if not self.auth_manager.available:
            self.status_lbl.config(
                text="⚠️ ไม่ได้ติดตั้ง pyrebase4 หรือเชื่อมต่อ Firebase ไม่ได้",
                fg=COLORS["danger"])
            self.btn_send.config(state="disabled")

        # ⚠️ เจอบั๊กเดียวกับ "380x240" คงที่ตอนทดสอบจริง — เนื้อหาต้องการสูงถึง ~278px
        # (มากกว่านั้นอีกถ้าโชว์ป้ายเตือน pyrebase4 ด้านบน) ทำให้ปุ่มส่งอีเมลถูกตัดขอบล่าง —
        # วัดจาก winfo_reqheight() จริงแทนเดาเลขคงที่ เหมือนที่แก้ไปแล้วใน LoginWindow._set_mode
        self.update()
        self.geometry(f"340x{self.winfo_reqheight() + 8}")

        self.email_entry.focus_set()
        if prefill_email:
            self.email_entry.select_range(0, "end")

    def _send(self):
        email = self.email_var.get()
        self.btn_send.config(state="disabled", text="กำลังส่ง...")
        self.status_lbl.config(text="", fg=COLORS["danger"])
        self.update_idletasks()

        success, message = self.auth_manager.send_password_reset(email)

        if success:
            self.status_lbl.config(text=f"✅ {message}", fg=COLORS["success"])
            self.btn_send.config(text="ส่งอีเมลรีเซ็ตรหัสผ่าน")
            # ปิดให้เองหลัง 2.5 วิ พอให้อ่านข้อความสำเร็จทัน ไม่ต้องรอผู้ใช้กดปิดเอง —
            # กลับไปหน้า Login หลักได้เลย (ไม่ได้ล็อกอินให้อัตโนมัติ ผู้ใช้ต้องไปตั้งรหัสผ่าน
            # ใหม่ในอีเมลก่อน แล้วค่อยพิมพ์รหัสใหม่เข้าสู่ระบบเองอีกที)
            self.after(2500, self.destroy)
        else:
            self.status_lbl.config(text=f"❌ {message}", fg=COLORS["danger"])
            self.btn_send.config(state="normal", text="ส่งอีเมลรีเซ็ตรหัสผ่าน")


class LoginWindow(tk.Tk):
    """ถ้าล็อกอิน/สมัครสมาชิกสำเร็จ self.logged_in_user จะถูกตั้งเป็น dict แล้วหน้าต่าง
    ปิดตัวเอง ผู้เรียกใช้ (เช่น prosthetic_gui.py) เช็ค self.logged_in_user หลัง mainloop() จบ:
        - ถ้า None แปลว่าผู้ใช้ปิดหน้าต่างเองโดยไม่ได้ล็อกอินสำเร็จ ให้จบโปรแกรมไปเลย
        - ถ้ามีค่า แปลว่าล็อกอิน/สมัครสมาชิกผ่านและมีสิทธิ์ Admin แล้ว ค่อยเปิดหน้าต่างหลักต่อ"""

    def __init__(self):
        super().__init__()
        self.title("เข้าสู่ระบบ — มือเทียม EMG Control System")
        self.geometry("380x420")
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])

        self.logged_in_user = None
        self.auth_manager = AuthManager()
        self.mode = "login"  # "login" | "signup" — ดู _set_mode()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        # ⚠️ เดิมตั้งความสูงหน้าต่างเป็นเลขคงที่ (380x420) เดายอดตอนออกแบบ แต่วัดจริงแล้ว
        # เนื้อหาโหมด login ต้องการสูงถึง ~428px (หรือ ~466px ตอนโชว์ป้ายเตือน pyrebase4
        # ไม่พร้อมใช้งาน — ข้อความ 2 บรรทัด) ทำให้แถวลิงก์ "สมัครสมาชิก/ลืมรหัสผ่าน" ล้นขอบ
        # ล่างหน้าต่างไปเงียบๆ (บั๊กแบบเดียวกับ Radar Chart ที่เพิ่งแก้ไปก่อนหน้านี้ — ตั้ง
        # ขนาดคงที่ไว้ไม่พอกับเนื้อหาจริง) แก้โดยให้ _set_mode() วัดความสูงที่ต้องใช้จริงจาก
        # widget เอง (winfo_reqheight) แล้วค่อยตั้ง geometry ทีหลังเสมอ แทนการเดาตัวเลขนิ่งๆ
        self._set_mode("login")
        self.email_entry.focus_set()

    def _build_ui(self):
        self.icon_lbl = tk.Label(self, text="🔒", font=("TH Sarabun New", 40), bg=COLORS["bg"],
                                  fg=COLORS["accent"])
        self.icon_lbl.pack(pady=(24, 4))
        self.title_lbl = tk.Label(self, text="เข้าสู่ระบบ Admin", font=("TH Sarabun New", 18, "bold"),
                                   bg=COLORS["bg"], fg=COLORS["text"])
        self.title_lbl.pack(pady=(0, 16))

        form = tk.Frame(self, bg=COLORS["bg"])
        form.pack(fill="x", padx=30)

        tk.Label(form, text="อีเมล", font=FONT, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")
        self.email_var = tk.StringVar()
        self.email_entry = tk.Entry(form, textvariable=self.email_var, font=FONT,
                                     bg=COLORS["card"], fg=COLORS["text"],
                                     insertbackground=COLORS["text"], relief="flat")
        self.email_entry.pack(fill="x", ipady=6, pady=(2, 12))
        self.email_entry.bind("<Return>", lambda e: self.password_entry.focus_set())

        tk.Label(form, text="รหัสผ่าน", font=FONT, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")
        self.password_var = tk.StringVar()
        self.password_entry = tk.Entry(form, textvariable=self.password_var, show="•",
                                        font=FONT, bg=COLORS["card"], fg=COLORS["text"],
                                        insertbackground=COLORS["text"], relief="flat")
        self.password_entry.pack(fill="x", ipady=6, pady=(2, 4))
        self.password_entry.bind("<Return>", lambda e: self._try_login())

        # ── ช่องยืนยันรหัสผ่าน — โชว์เฉพาะโหมดสมัครสมาชิก (pack/pack_forget ใน _set_mode
        # ไม่ได้สร้าง/ทำลาย widget ใหม่ทุกครั้งที่สลับโหมด ง่ายกว่าและกัน state ตัวแปรเพี้ยน) ──
        self.confirm_frame = tk.Frame(form, bg=COLORS["bg"])
        tk.Label(self.confirm_frame, text="ยืนยันรหัสผ่าน", font=FONT, bg=COLORS["bg"],
                 fg=COLORS["text_dim"]).pack(anchor="w")
        self.confirm_var = tk.StringVar()
        self.confirm_entry = tk.Entry(self.confirm_frame, textvariable=self.confirm_var, show="•",
                                       font=FONT, bg=COLORS["card"], fg=COLORS["text"],
                                       insertbackground=COLORS["text"], relief="flat")
        self.confirm_entry.pack(fill="x", ipady=6, pady=(2, 4))
        self.confirm_entry.bind("<Return>", lambda e: self._try_signup())
        # (ไม่ pack ตอนสร้าง — โหมดเริ่มต้นคือ login ที่ไม่มีช่องนี้ _set_mode("signup") เป็น
        # คนสั่ง pack ให้ทีหลังตอนสลับโหมด)

        self.status_lbl = tk.Label(self, text="", font=("TH Sarabun New", 11),
                                    bg=COLORS["bg"], fg=COLORS["danger"], wraplength=320,
                                    justify="left")
        self.status_lbl.pack(padx=30, pady=(4, 8), anchor="w")

        self.btn_action = tk.Button(self, text="เข้าสู่ระบบ", font=FONT_B,
                                     bg=COLORS["accent"], fg="white", relief="flat",
                                     activebackground=COLORS["accent"], cursor="hand2",
                                     command=self._try_login)
        self.btn_action.pack(fill="x", padx=30, ipady=8, pady=(4, 4))

        # ฟอนต์ขีดเส้นใต้แยกต่างหาก — tuple font ปกติของ tkinter ใส่ "underline" panel
        # ปนกับ weight ไม่ได้ตรงๆ ต้องสร้างเป็น tkfont.Font object แทน
        link_font = tkfont.Font(family="TH Sarabun New", size=10, underline=True)

        # ── แถวลิงก์โหมด login: "สมัครสมาชิก" (ซ้าย) + "ลืมรหัสผ่าน?" (ขวา) ──────────
        self.login_links = tk.Frame(self, bg=COLORS["bg"])
        self.signup_link_lbl = tk.Label(self.login_links, text="ยังไม่มีบัญชี? สมัครสมาชิก",
                                         font=link_font, bg=COLORS["bg"], fg=COLORS["accent"],
                                         cursor="hand2")
        self.signup_link_lbl.pack(side="left")
        self.forgot_link_lbl = tk.Label(self.login_links, text="ลืมรหัสผ่าน?",
                                         font=link_font, bg=COLORS["bg"], fg=COLORS["text_dim"],
                                         cursor="hand2")
        self.forgot_link_lbl.pack(side="right")
        self.signup_link_lbl.bind("<Button-1>", lambda e: self._set_mode("signup"))
        self.forgot_link_lbl.bind("<Button-1>", lambda e: self._open_forgot_password())

        # ── แถวลิงก์โหมด signup: "เข้าสู่ระบบ" (กลับไปโหมด login) ─────────────────────
        self.signup_links = tk.Frame(self, bg=COLORS["bg"])
        self.login_link_lbl = tk.Label(self.signup_links, text="← มีบัญชีอยู่แล้ว? เข้าสู่ระบบ",
                                        font=link_font, bg=COLORS["bg"], fg=COLORS["accent"],
                                        cursor="hand2")
        self.login_link_lbl.pack(side="left")
        self.login_link_lbl.bind("<Button-1>", lambda e: self._set_mode("login"))

        self.login_links.pack(fill="x", padx=30, pady=(0, 12))  # โหมดเริ่มต้น = login
        # หมายเหตุ: ไม่ตั้งป้ายเตือน "pyrebase4 ไม่พร้อมใช้งาน" ตรงนี้ — ปล่อยให้ _set_mode()
        # (ซึ่ง __init__ เรียกทันทีหลัง _build_ui() คืนค่า) เป็นคนตั้งแทน ที่เดียวกับที่คุม
        # status_lbl อยู่แล้ว กันบั๊ก _set_mode() เคลียร์ข้อความนี้ทิ้งไปเงียบๆ ตอนสลับโหมด

    def _set_mode(self, mode: str):
        """สลับฟอร์มระหว่างโหมด 'login' และ 'signup' (โชว์/ซ่อน widget แทนการสร้างใหม่ —
        ง่ายกว่าและกัน state widget เพี้ยน) เคลียร์ status message เก่าทุกครั้งที่สลับ กัน
        ข้อความ error จากโหมดเดิมค้างอยู่ให้สับสน — อีเมลที่พิมพ์ไปแล้วไม่หาย (ใช้ StringVar
        ตัวเดียวกันทั้ง 2 โหมด)

        ความสูงหน้าต่างคำนวณจาก winfo_reqheight() จริงของเนื้อหาหลังสลับโหมดเสมอ (ไม่ใช้เลข
        คงที่ที่เดาไว้ตอนออกแบบ) เพราะความสูงที่ต้องใช้จริงเปลี่ยนได้หลายทาง — โหมด signup
        มีช่อง "ยืนยันรหัสผ่าน" เพิ่มมาอีกแถว, ป้ายเตือน "pyrebase4 ไม่พร้อมใช้งาน" (ตอน
        auth_manager.available=False) เป็นข้อความ 2 บรรทัดที่กินพื้นที่เพิ่ม ฯลฯ — เดาเลข
        คงที่ไว้ล่วงหน้าเสี่ยงเนื้อหาล้นขอบหน้าต่างเงียบๆ แบบเดียวกับบั๊ก Radar Chart ก่อนหน้านี้"""
        self.mode = mode
        # ป้ายเตือน "pyrebase4 ไม่พร้อมใช้งาน" ต้องอยู่ค้างทุกโหมด (เป็นปัญหาระดับ environment
        # ไม่ใช่ผลลัพธ์ของการกดปุ่ม login/signup ครั้งล่าสุด) — เคลียร์ status_lbl เป็นค่าว่าง
        # เฉพาะตอนเชื่อมต่อ Firebase ได้ปกติเท่านั้น ไม่งั้นป้ายเตือนจะหายไปทุกครั้งที่สลับโหมด
        if self.auth_manager.available:
            self.status_lbl.config(text="")
        else:
            self.status_lbl.config(
                text="⚠️ ไม่ได้ติดตั้ง pyrebase4 หรือเชื่อมต่อ Firebase ไม่ได้\n"
                     "รัน: pip install pyrebase4",
                fg=COLORS["danger"])

        if mode == "signup":
            self.title_lbl.config(text="สมัครสมาชิกใหม่")
            self.confirm_frame.pack(fill="x", after=self.password_entry)
            self.confirm_var.set("")
            self.btn_action.config(text="สมัครสมาชิก", command=self._try_signup)
            self.password_entry.unbind("<Return>")
            self.password_entry.bind("<Return>", lambda e: self.confirm_entry.focus_set())
            self.login_links.pack_forget()
            self.signup_links.pack(fill="x", padx=30, pady=(0, 12))
        else:
            self.title_lbl.config(text="เข้าสู่ระบบ Admin")
            self.confirm_frame.pack_forget()
            self.btn_action.config(text="เข้าสู่ระบบ", command=self._try_login)
            self.password_entry.unbind("<Return>")
            self.password_entry.bind("<Return>", lambda e: self._try_login())
            self.signup_links.pack_forget()
            self.login_links.pack(fill="x", padx=30, pady=(0, 12))

        self.btn_action.config(state="normal" if self.auth_manager.available else "disabled")

        # วัดความสูงที่ต้องใช้จริงหลัง pack/pack_forget มีผลแล้ว แล้วเผื่อ margin อีกเล็กน้อย
        # (8px) กันเส้นขอบ/เงาฟอนต์บางเครื่องถูกตัดพอดีเป๊ะ — ⚠️ ต้องใช้ update() (ไม่ใช่แค่
        # update_idletasks()) โดยเฉพาะตอนเรียกครั้งแรกจาก __init__ ก่อนหน้าต่างเคยถูก map ขึ้น
        # จอเลยสักครั้ง: label ที่ตั้ง wraplength ไว้ (เช่น status_lbl/ป้ายเตือน pyrebase4)
        # ต้องให้ font ตัวจริงถูก resolve ผ่านการ map/draw จริงก่อน ถึงจะคำนวณจำนวนบรรทัดที่
        # ข้อความจะตัดขึ้นบรรทัดใหม่ได้ถูกต้อง — เรียกแค่ update_idletasks() ตอนหน้าต่างยังไม่
        # เคย map เลยสักครั้ง จะได้ reqheight ที่ยังไม่นับความสูงจริงของ label พวกนี้ ทำให้
        # หน้าต่างเปิดมาเล็กเกินไปตั้งแต่ครั้งแรก (พบระหว่างทดสอบ — ไม่ใช่แค่ทฤษฎี)
        self.update()
        self.geometry(f"380x{self.winfo_reqheight() + 8}")
        self.email_entry.focus_set()

    def _open_forgot_password(self):
        ForgotPasswordDialog(self, self.auth_manager, prefill_email=self.email_var.get())

    def _try_login(self):
        email = self.email_var.get()
        password = self.password_var.get()
        self.btn_action.config(state="disabled", text="กำลังตรวจสอบ...")
        self.status_lbl.config(text="", fg=COLORS["danger"])
        self.update_idletasks()

        success, message, user_info = self.auth_manager.login(email, password)

        if success:
            self.logged_in_user = user_info
            self.destroy()
        else:
            self.status_lbl.config(text=f"❌ {message}", fg=COLORS["danger"])
            self.btn_action.config(state="normal", text="เข้าสู่ระบบ")
            self.password_entry.focus_set()
            self.password_entry.select_range(0, "end")

    def _try_signup(self):
        email = self.email_var.get()
        password = self.password_var.get()
        confirm = self.confirm_var.get()
        self.btn_action.config(state="disabled", text="กำลังสมัคร...")
        self.status_lbl.config(text="", fg=COLORS["danger"])
        self.update_idletasks()

        success, message, user_info = self.auth_manager.signup(email, password, confirm)

        if success:
            self.logged_in_user = user_info
            self.destroy()
        else:
            self.status_lbl.config(text=f"❌ {message}", fg=COLORS["danger"])
            self.btn_action.config(state="normal", text="สมัครสมาชิก")

    def _on_cancel(self):
        self.logged_in_user = None
        self.destroy()