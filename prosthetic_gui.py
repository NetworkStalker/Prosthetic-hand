"""
มือเทียมควบคุมด้วยสัญญาณกล้ามเนื้อ
Myoelectric Prosthetic Hand - Calibration Tool & Dashboard
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import json
import math
import time
import collections
import statistics
from datetime import datetime, timedelta

# ─── กราฟฝัง GUI (แท็บ Machine Learning): loss/performance ระหว่างเทรน + เปรียบเทียบ
# predict บน test set (ดู _render_ml_loss_graph / _render_ml_performance_graph /
# _render_ml_predict_graph) ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("TkAgg")  # ต้องตั้งก่อน import pyplot/backend อื่นเสมอ (เหมือน main.py)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np  # ใช้สร้าง meshgrid สำหรับกราฟ Decision Boundary (ดู _render_ml_boundary_graph)

# ─── Real-time EMG streaming + Offline-first Firebase sync ──────────────────
# แทนที่ EMGSimulator + FirebaseManager แบบเดิม ด้วยโมดูลกลางที่ใช้ร่วมกับ main.py
# emg_stream.EMGStreamThread จะอ่านค่าจาก Hardware จริง (ADS1115) อัตโนมัติถ้ามี
# ไม่งั้น fallback เป็นสัญญาณจำลองเพื่อให้ยังพัฒนา/ทดสอบ GUI บนเครื่อง dev ได้
from emg_stream import EMGStreamThread
from emg_features import (FEATURE_NAMES, features_to_vector, PERSON_CATEGORIES,
                           PERSON_CATEGORY_LABELS, DEFAULT_PERSON_CATEGORY)

# ป้ายภาษาไทย <-> key ภายใน (ดู emg_features.PERSON_CATEGORY_LABELS) — Combobox ในหน้าจอ
# โชว์เป็นภาษาไทย แต่ทุกจุดที่ส่งต่อไปยัง firebase_sync/ml_online ต้องใช้ key ภาษาอังกฤษ
# เสมอ (เป็น "สัญญา" เดียวกับ emg_features.PERSON_CATEGORIES) กันปัญหาพิมพ์ภาษาไทยเพี้ยน/
# ตัวสะกดไม่ตรงกันระหว่างจุดที่บันทึกกับจุดที่อ่านย้อนกลับมาเทรน
PERSON_CATEGORY_VALUES = [PERSON_CATEGORY_LABELS[c] for c in PERSON_CATEGORIES]
_PERSON_CATEGORY_LABEL_TO_KEY = {v: k for k, v in PERSON_CATEGORY_LABELS.items()}
from firebase_sync import FirebaseSyncManager
from servo_control import EMGServoController
from ml_online import OnlineTrainer, BASE_MODELS, ENSEMBLE_METHODS, CLASSES
from event_recorder import GestureEventRecorder

# ─── สีและฟอนต์ ───────────────────────────────────────────────────────────────
# ปรับปาเลตใหม่ให้ทันสมัยขึ้น: โทนกรมท่า/เนวี่เข้ม (แทนม่วงเทาเดิม) ตัดด้วยไวโอเล็ต-ฟ้าไซแอน
# สดใสสำหรับจุดเน้น ให้ตัดกับพื้นหลังชัดเจนกว่าเดิม อิงแนวเดียวกับหน้าอ้างอิงที่ให้มา
# (การ์ดกรอบบาง ปุ่มสีสันชัด เขียวมรกต/ฟ้าไซแอนสำหรับสถานะ "พร้อม/ออนไลน์")
COLORS = {
    "bg":           "#0b1120",   # พื้นหลังหลัก กรมท่าเข้มเกือบดำ
    "surface":      "#141d33",   # พื้นผิวการ์ด/แถบควบคุม
    "card":         "#1b2540",   # การ์ดย่อย/กล่องสถิติ ให้ลึกกว่าพื้นผิวเล็กน้อย
    "accent":       "#8b6ff0",   # ม่วงไวโอเล็ต — ปุ่มหลัก/บันทึก
    "accent2":      "#22d3ee",   # ฟ้าไซแอนสด — ลิงก์/กรอบเน้น/ปุ่มรอง
    "danger":       "#fb5468",   # แดงกุหลาบ — ลบ/อันตราย
    "warning":      "#fbbf24",   # เหลืองอำพัน — คำเตือน/threshold label
    "success":      "#34d399",   # เขียวมรกต — สถานะพร้อม/ออนไลน์
    "text":         "#e8ecf7",
    "text_dim":     "#8b96b8",
    "border":       "#2a3556",
    "emg_line":     "#8b6ff0",
    "threshold":    "#fb5468",
    "grip":         "#22d3ee",
    "release":      "#fbbf24",
    "gold":         "#f5c542",
}
FONT_TH   = ("TH Sarabun New", 13)
FONT_TH_B = ("TH Sarabun New", 13, "bold")
FONT_H1   = ("TH Sarabun New", 18, "bold")
FONT_H2   = ("TH Sarabun New", 15, "bold")
FONT_MONO = ("Courier New", 11)

# ─── Custom scrollbar (วาดเองด้วย Canvas) ────────────────────────────────────
# เดิมใช้ tk.Scrollbar(width=28) แต่พอเทียบกับ screenshot จริงบนเครื่องที่ใช้งาน
# (ทั้ง dev บน macOS/Windows และ Raspberry Pi) ค่า width ที่ตั้งไม่ค่อยถูกเคารพเท่ากัน —
# บาง theme ของ Tk (โดยเฉพาะ theme ที่มากับ OS อย่าง Aqua/บาง GTK theme บน Linux)
# บังคับความกว้างขั้นต่ำของตัว native scrollbar เอง ทำให้ยังได้แถบบางๆ ~14px อยู่ดี
# ต่อให้ตั้ง width=28 ในโค้ดแล้วก็ตาม แก้ปัญหานี้ให้เด็ดขาดด้วยการ "วาดสโครลบาร์เอง"
# บน Canvas แทนการพึ่ง native widget — รับประกันความกว้าง/สีเหมือนกันทุกเครื่อง
class CustomScrollbar(tk.Canvas):
    """สโครลบาร์ที่วาดเองล้วนๆ (ไม่ใช้ tk.Scrollbar) ใช้ .set(first, last) แบบเดียวกับ
    scrollbar ปกติ จึงเสียบแทน tk.Scrollbar ได้ทุกจุด (yscrollcommand=this.set,
    command=widget.yview) รองรับทั้งลากนิ้ว (touch) และคลิกที่ราง (jump-to)"""

    def __init__(self, parent, command=None, width=32,
                 bg=None, thumb_color=None, trough_color=None):
        trough_color = trough_color or COLORS["card"]
        super().__init__(parent, width=width, bg=trough_color, highlightthickness=0)
        self.command = command
        self.thumb_color = thumb_color or COLORS["accent"]
        self._first, self._last = 0.0, 1.0
        self._drag_offset = 0
        self.bind("<Configure>", lambda e: self._redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        # เผื่อกรณี <Configure> แรกสุดยิงมาก่อนที่ layout จะนิ่ง (winfo_height() ยังเป็น 1)
        # แล้วไม่มี event อื่นมา trigger ซ้ำอีก บังคับวาดใหม่หลัง event loop ว่างอีกที
        self.after_idle(self._redraw)

    def set(self, first, last):
        self._first, self._last = float(first), float(last)
        self._redraw()

    def _redraw(self):
        self.delete("thumb")
        w, h = self.winfo_width(), self.winfo_height()
        if h <= 1 or w <= 1:
            self.after(30, self._redraw)  # ยังไม่มีขนาดจริง (layout ยังไม่นิ่ง) ลองใหม่
            return
        # กรอบราง (track) ให้เห็นตำแหน่ง/ขนาดของสโครลบาร์ชัดเจนแม้ธัมบ์จะยังไม่โผล่
        # (กันปัญหาที่คอลัมน์สโครลบาร์กลืนไปกับพื้นหลังจนดูเหมือนไม่มี scrollbar เลย)
        self.delete("track")
        self.create_rectangle(1, 1, w - 1, h - 1, outline=COLORS["text_dim"],
                               width=1, tags="track")
        y0, y1 = self._first * h, self._last * h
        if y1 - y0 < 24:  # ขั้นต่ำ 24px กันธัมบ์เล็กเกินไปจนแตะโดนยากบน touchscreen
            mid = (y0 + y1) / 2
            y0, y1 = max(0, mid - 12), min(h, mid + 12)
        pad = 3
        radius = max(0, (w - 2 * pad) / 2)
        self._round_rect(pad, y0 + pad, w - pad, y1 - pad, radius, fill=self.thumb_color)

    def _round_rect(self, x0, y0, x1, y1, r, **kw):
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        points = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
                  x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        self.create_polygon(points, smooth=True, tags="thumb", outline="", **kw)

    def _on_press(self, event):
        h = self.winfo_height()
        y0, y1 = self._first * h, self._last * h
        if y0 <= event.y <= y1:
            self._drag_offset = event.y - y0
        else:
            span = self._last - self._first
            frac = max(0.0, min(1.0 - span, event.y / h - span / 2))
            self._drag_offset = event.y - frac * h
            if self.command:
                self.command("moveto", frac)

    def _on_drag(self, event):
        h = self.winfo_height()
        if h <= 1:
            return
        span = self._last - self._first
        frac = max(0.0, min(1.0 - span, (event.y - self._drag_offset) / h))
        if self.command:
            self.command("moveto", frac)


# ─── Main Application ─────────────────────────────────────────────────────────
# EMG range (โวลต์) สำหรับสเกลกราฟ/แถบระดับสัญญาณ — ค่า EMG จริงจาก SEN0240 + ADS1115
# หลังผ่าน rectify + RMS envelope จะอยู่ราวๆ 0–0.5V (คนละหน่วยกับโค้ดเดิมที่จำลอง 0–500 ลอยๆ)
EMG_RANGE_V = 0.5
class ProstheticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("มือเทียม EMG Control System")
        self.geometry("1400x860")
        self.minsize(1150, 700)
        self.configure(bg=COLORS["bg"])

        # Offline-first Firebase: เขียนไฟล์ .jsonl ในเครื่องเสมอ แล้วซิงก์ขึ้น
        # Firebase Realtime Database เป็นพื้นหลังเมื่อออนไลน์
        # auto_start=False: ไม่เริ่มเธรด sync ขึ้น Firebase ทันทีตอนเปิดแอป — รอจนกว่า
        # ผู้ใช้จะกด "บันทึก" หรือ "เริ่มบันทึกแบบ Real-time" เองก่อน (ดู _save_calibration
        # และ _toggle_realtime_logging) กันไม่ให้ backlog ที่ค้าง sync จากรอบก่อนถูกส่งขึ้น
        # Firebase อัตโนมัติ ทั้งที่ผู้ใช้ยังไม่ทันกดอะไรในรอบนี้เลย — ข้อมูลใน local ไฟล์
        # ยังปลอดภัยเสมอไม่ว่าจะเริ่ม sync หรือยัง (เขียนไฟล์ในเครื่องแยกจากการ sync)
        self.firebase  = FirebaseSyncManager(auto_start=False)

        self.recording = False
        self.calib_samples: list[tuple[float, str, dict]] = []  # (rms, gesture, features)
        # features คือ dict 7 ค่า (rms/mav/variance/waveform_length/zero_crossing/
        # slope_sign_change/iemg) จาก sliding window ล่าสุด ณ ตอนที่เก็บ sample นี้ — อาจ
        # เป็น None ได้ถ้า window ยังไม่ทันคำนวณเสร็จ (เช่นเพิ่งเริ่มสตรีมสัญญาณ)
        self._current_calib_gesture = "คลายมือ"

        # ── สถานะสำหรับ "จำลองเก็บข้อมูลอัตโนมัติ" ในแท็บ Calibration — สลับนับถอยหลัง
        # คลายมือ/กำมือ อัตโนมัติทีละรอบผ่าน after() (ดู _start_auto_simulate ด้านล่าง)
        self._auto_sim_running = False
        self._auto_sim_after_id = None
        self._auto_sim_cycle = 0
        self._auto_sim_total_cycles = 0
        self._auto_sim_base_uid = None       # uid ตอนเริ่มจำลอง — ใช้ตั้งชื่อ user ปลายทาง '{uid}_label'
        self._auto_sim_start_index = 0       # ตำแหน่งเริ่มต้นใน calib_samples ตอนเริ่มจำลองรอบนี้
        self._auto_sim_collecting = False    # True เฉพาะช่วงที่ timer เก็บ 50 samples/ช่วงกำลังทำงานอยู่
        self._auto_sim_sample_after_id = None
        self._auto_sim_warmup_after_id = None  # timer ช่วง warmup ก่อนเริ่มเก็บ sample จริงของแต่ละ phase
        self.emg_history: list[float]   = [0.0] * 200
        self.current_emg  = tk.DoubleVar(value=0)
        self.gesture_var  = tk.StringVar(value="—")
        self.user_id_var  = tk.StringVar(value="")  # ว่างไว้ก่อน — บังคับให้ผู้ใช้ "เพิ่ม User" เอง
        # ประเภทบุคคลของ user ที่กำลังบันทึก/ทำนายอยู่ตอนนี้ (ปกติ/เด็ก/คนแก่/ผู้ป่วยกล้ามเนื้อ/
        # ผู้พิการ — ดู emg_features.PERSON_CATEGORIES) ใช้เป็น feature เพิ่มเติมป้อนเข้าโมเดล
        # กำ-คลายมือเดิม (ดู ml_online.OnlineTrainer.add_sample/predict_with_confidence) และ
        # บันทึกคู่ไปกับทุก sample ที่ส่งขึ้น Firebase (ดู _on_emg_sample) — Combobox โชว์เป็น
        # ภาษาไทย (ค่าตรงกับ PERSON_CATEGORY_VALUES) ส่วน self._current_person_category ที่ใช้
        # จริงทั่วแอปเก็บเป็น key ภาษาอังกฤษเสมอ (แปลงตอนเลือกจาก Combobox — ดู
        # _on_person_category_change) ดีฟอลต์เป็น "ปกติ" (DEFAULT_PERSON_CATEGORY)
        self.person_category_var = tk.StringVar(value=PERSON_CATEGORY_LABELS[DEFAULT_PERSON_CATEGORY])
        self.threshold    = 0.05  # หน่วยโวลต์ (ตรงกับ RMS envelope จริง / calibration.json) —
        # เส้น "บน" (activate/FIST) ของ hysteresis 2 เส้น
        self.hysteresis_margin = 0.015  # ช่องว่าง (โวลต์) ระหว่างเส้นบน/ล่าง ตอนคำนวณเอง
        # จาก slider เดี่ยว (ยังไม่เคยกด "คำนวณ Threshold อัตโนมัติ") — ค่านี้จะถูกแทนที่ด้วย
        # ช่องว่างจริงจากข้อมูล calibration ของ user คนนั้นๆ ทันทีที่กดปุ่มคำนวณอัตโนมัติ
        self.release_threshold = max(0.0, self.threshold - self.hysteresis_margin)  # เส้น
        # "ล่าง" (release/REST) — ระหว่างสองเส้นนี้คือช่วง noise ที่ยังไม่ตัดสินใจเปลี่ยนสถานะ
        self.status_var   = tk.StringVar(value="⬤  พร้อมใช้งาน")

        # ── Machine Learning tab state ──────────────────────────────────────
        # เปลี่ยนตามที่ทีมสรุปกัน: "ไม่ train realtime แล้ว เพราะค่าวิ่งอยู่ตลอด"
        # เทรนได้เฉพาะจากข้อมูลย้อนหลังที่บันทึกไว้ในเครื่องเท่านั้น (Offline เท่านั้น
        # ไม่มีโหมด live/real-time train อีกต่อไป) โดยเลือกช่วงย้อนหลังได้ 3 / 6 / 12
        # เดือนนับจากวันนี้ (ตามที่วิเคราะห์ข้อมูลกันปกติ) หรือจะอัพโหลดไฟล์ JSON เองก็ได้
        self.ml_base_var     = tk.StringVar(value="Random Forest")
        self.ml_ensemble_var = tk.StringVar(value="None")
        self.ml_range_var    = tk.StringVar(value="3")  # "3" / "6" / "12" เดือนย้อนหลัง
        # ── เลือก user สำหรับ "ดึงข้อมูลย้อนหลังมาเทรน" — แยกจาก self.user_id_var โดย
        # ตั้งใจ เพราะ self.user_id_var (หน้า Dashboard/Calibration) ใช้กำหนดว่ากำลัง
        # "บันทึก" สัญญาณเข้า Firebase ให้ใคร ส่วนอันนี้ใช้กำหนดว่าจะ "ดึง" ข้อมูลของใคร
        # ออกมาเทรน คนละทิศทางกัน ไม่ควรผูกกันเป็นตัวแปรเดียว
        self.ml_user_mode_var = tk.StringVar(value="specific")  # "specific" หรือ "all"
        # เลือกได้ทีละหลายคนพร้อมกัน (ไม่ใช่แค่คนเดียว) — เก็บเป็น list ของ user_id ที่ติ๊กไว้
        # ผ่าน dialog เลือกผู้ใช้ (ดู _open_ml_user_select_dialog) ส่วน ml_user_summary_var
        # ใช้แค่โชว์สรุปบนหน้าจอ (เช่น "เลือกแล้ว 3 คน: A, B, C")
        self.ml_selected_users: list[str] = []
        self.ml_user_summary_var = tk.StringVar(value="ยังไม่ได้เลือกผู้ใช้")
        # normalize RMS แยกตามคนก่อนรวม dataset — แต่ละคนแรงกล้ามเนื้อ/ตำแหน่ง electrode ไม่
        # เท่ากัน ค่า RMS ดิบเทียบกันข้ามคนตรงๆ ไม่ได้ ดีฟอลต์เปิดไว้ (มีผลจริงเฉพาะตอนเลือก
        # มากกว่า 1 user เท่านั้น — ดู _load_ml_history)
        self.ml_normalize_var = tk.BooleanVar(value=True)
        # เปิด/ปิดแสดง panel Feature (RMS/MAV/Variance/Waveform Length/Zero Crossing/
        # Slope Sign Change/IEMG) บน Dashboard — ดีฟอลต์เปิด (ข้อกำหนดเรื่อง GUI: ต้องแสดง
        # feature ได้ และเปิด/ปิดได้)
        self.show_features_var = tk.BooleanVar(value=True)

        # แหล่งสั่งงาน servo จริง — "threshold" (ดีฟอลต์ เหมือนระบบเดิมทุกประการ) เทียบ RMS
        # กับ threshold ตรงๆ แบบเดิม หรือ "ml_model" ให้โมเดล multi-feature ที่เทรนไว้ในแท็บ
        # Machine Learning เป็นคนตัดสินใจแทน — ดีฟอลต์เป็น threshold โดยตั้งใจ (ไม่ใช่ ml_model)
        # เพราะเป็นอุปกรณ์ที่ต้องมั่นใจว่าทำงานได้แน่ๆ ตั้งแต่เปิดเครื่อง แม้ผู้ใช้ยังไม่เคยเทรน
        # โมเดลเลยก็ตาม ต้องกดเลือกเปลี่ยนเป็น ML เองเท่านั้น ไม่ auto-switch ให้ (ดู
        # _on_servo_source_change ที่มี dialog ยืนยันก่อนสลับไปโหมด ML ด้วย)
        self.servo_source_var = tk.StringVar(value="threshold")
        # ระดับความมั่นใจขั้นต่ำ (%) ที่โมเดลต้องมีก่อนจะยอมให้ทำนายไปสั่ง servo — ถ้าต่ำกว่านี้
        # (โมเดลลังเล) จะ fallback ไปใช้ RMS threshold แทนชั่วคราวสำหรับ sample นั้น กันมือเทียม
        # ค้างท่าผิดตอนโมเดลไม่มั่นใจ (ดู _on_emg_sample)
        self._ML_SERVO_MIN_CONFIDENCE = 60.0
        self.trainer = OnlineTrainer(self.ml_base_var.get(), self.ml_ensemble_var.get())
        self._ml_replay_queue: list[tuple[dict, str, str]] = []  # (features_dict, gesture, person_category)
        # เก็บสำเนา dataset ล่าสุดที่โหลดสำเร็จไว้เต็มๆ (ไม่ถูก drain เหมือน _ml_replay_queue)
        # ใช้ตอนกด "Export Dataset เป็น CSV" — เป็น None ถ้ายังไม่เคยโหลดอะไรเลยในเซสชันนี้
        self._ml_last_loaded_samples: list[tuple[dict, str, str]] | None = None
        self._ml_replay_total = 0
        # เก็บผลลัพธ์ล่าสุดของโมเดล/ensemble แต่ละคอมโบ้ที่เคยเทรนไปแล้วในเซสชันนี้
        # (key = display_name เช่น "Random Forest", "Bagging + SVM") ไว้เทียบกันในหน้า
        # "สรุปผลโมเดล" — อัพเดตทุกครั้งที่ _refresh_ml_metrics() มีผลประเมินใหม่
        self.model_results: dict[str, dict] = {}
        self.compare_metric_var = tk.StringVar(value="Accuracy")
        self.compare_sort_var = tk.StringVar(value="สูง → ต่ำ")
        self.compare_selected_model: str | None = None
        self._ml_last_seen_error_count = 0    # ใช้เทียบกับ snap['n_errors'] เพื่อรู้ว่ามี error ใหม่
        self._compare_last_updated = None    # datetime ของผลลัพธ์ล่าสุดที่บันทึกเข้า model_results
        self._compare_show_all = False        # False = โชว์แค่ 10 อันดับแรกในตาราง

        # ค่า EMG ล่าสุดจากเธรดอ่านสัญญาณ (เขียน/อ่านข้าม thread ผ่าน lock)
        self._latest_lock = threading.Lock()
        self._latest = {"voltage": 0.0, "rms": 0.0, "gesture": "OPEN", "features": None}
        self._current_user_id = self.user_id_var.get().strip() or "user_01"
        # key ภาษาอังกฤษ (ไม่ใช่ label ไทยที่โชว์ใน Combobox) — ค่าที่ใช้จริงทุกจุดในแอป
        # (log_sample/log_labeled_sample/log_calibration/trainer.predict_with_confidence/
        # trainer.add_sample) อัพเดตทันทีที่เลือกจาก Combobox (ดู _on_person_category_change
        # — เลือกจาก dropdown เป็น action ที่ atomic อยู่แล้ว ไม่มีปัญหา "พิมพ์ทีละตัวอักษร"
        # แบบ user_id_var จึงไม่ต้องรอ "ยืนยัน" เหมือน _confirm_user_id)
        self._current_person_category = DEFAULT_PERSON_CATEGORY
        self._realtime_logging_active = False  # ต้องกด "เริ่มบันทึกแบบ Real-time" เองก่อนถึงจะเขียนข้อมูล
        self._current_session_id = None   # ตั้งตอนกด "เริ่มบันทึกแบบ Real-time" — ดู _toggle_realtime_logging
        self._current_session_start_iso = None

        # ── ตัวบันทึก "เหตุการณ์กำมือ" แบบเป็นก้อนๆ (event-based clip) ──────────
        # แยกจาก _realtime_logging_active (ที่บันทึกทุก sample ต่อเนื่อง) โดยเจตนา —
        # ตัวนี้จะรวบรวมสัญญาณของการกำมือ 1 ครั้ง (รวม pre-roll ก่อนกำมือ 5 วิ + ช่วง
        # กำมือ/คลายมือสลับกันจนคลายมือค้างครบ 5 วิ) แล้วบันทึกเป็นก้อนเดียวขึ้น Firebase
        # node "datasets" — ปิดไว้เป็นค่าเริ่มต้นเสมอ (enabled=False) ต้องกดเปิดเองจาก
        # Dashboard ตามหลักการเดียวกับฟีเจอร์บันทึกอื่นๆ ในแอปนี้
        self.event_recorder = GestureEventRecorder(
            pre_roll_sec=5.0, release_hold_sec=5.0,
            on_event_saved=self._on_gesture_event_saved, enabled=False)

        # เธรดอ่านสัญญาณ EMG แบบ real-time (Hardware จริงถ้ามี ไม่งั้น fallback เป็นจำลอง)
        self.stream = EMGStreamThread(on_sample=self._on_emg_sample, threshold=self.threshold)
        self.hardware_mode = self.stream.hardware_mode
        self.threshold = self.stream.threshold  # ใช้ค่าจาก calibration.json ถ้ามี
        self.release_threshold = max(0.0, self.threshold - self.hysteresis_margin)

        # Servo ผ่าน PCA9685 — ขยับตาม RMS เหมือนกับใน main.py
        # ใช้ self.threshold (ค่าที่ปรับได้จากแอป — manual slider หรือ calibration.json)
        # แทนค่า hardcode เดิม เพื่อให้ threshold เดียวกันทั้งระบบ ไม่มีค่าซ้อนกันสองชุด
        self.servo = EMGServoController(channel=0, threshold=self.threshold)

        self.firebase_var = tk.StringVar(
            value="⬤  Online" if self.firebase.online else "⬤  Offline"
        )

        # ── สถานะสำหรับหน้า Dashboard ใหม่ ───────────────────────────────────
        self.dash_window_var = tk.StringVar(value="2 นาที")  # ตัวเลือกช่วงเวลาบนกราฟ
        self._dash_window_options = {"30 วินาที": 30, "1 นาที": 60,
                                      "2 นาที": 120, "5 นาที": 300}
        # ใช้ deque(maxlen=...) แทน list ธรรมดา — ตอนก่อนหน้านี้ trim buffer ด้วย
        # del list[0:k] ทุกเฟรม ซึ่งเป็น O(n) (ต้องขยับสมาชิกที่เหลือทั้งหมด) พอ buffer
        # ยาวเป็นพันจุด (ช่วงเวลานาน ๆ อย่าง 5 นาที = 7,500 จุด) ทำแบบนี้ 25 ครั้ง/วิ
        # แล้วหน่วงเห็นได้ชัด — deque(maxlen=) evict อัตโนมัติแบบ O(1) แทน
        self._dash_window_maxlen = self._dash_window_options[self.dash_window_var.get()] * 25
        self.dash_chart_history = collections.deque(maxlen=self._dash_window_maxlen)
        self._gesture_confidence = None  # % ความมั่นใจของโมเดลล่าสุด (None ถ้ายังไม่มีโมเดล)
        self._last_fb_online_ts = "—"    # เวลาล่าสุดที่เห็น Firebase Online (ประมาณจาก client)
        self.current_page = "dashboard"

        self._build_ui()
        self.stream.start()
        # หมายเหตุ: ไม่ต้อง pause self.stream เพื่อ ML อีกต่อไป — เธรดอ่านสัญญาณ EMG
        # นี้จำเป็นสำหรับ Dashboard/Calibration/Servo เสมอ ส่วน ML ฝั่ง Machine Learning
        # เทรนจากไฟล์ log ย้อนหลัง (Offline) เท่านั้น ไม่แตะสัญญาณสดเลย จึงไม่ชนกัน
        self._start_loop()

    # ─── Real-time EMG callback (รันบนเธรดของ EMGStreamThread ไม่ใช่ main thread) ──
    def _on_emg_sample(self, voltage, rms, gesture, t):
        # ดึง feature ล่าสุดที่ sliding window คำนวณเสร็จ (mav/variance/waveform_length/
        # zero_crossing) — อาจเป็น None ถ้ายังไม่มี window ไหนคำนวณเสร็จเลย (เช่นเพิ่งเริ่ม
        # สตรีมสัญญาณ ข้อมูลยังไม่ครบ window_size) ไม่บล็อกรอ เพราะ on_sample ต้องเร็ว
        # ที่สุดเท่าที่จะทำได้ (เรียกทุก sample ที่ ~500Hz)
        features = self.stream.get_latest_features()
        with self._latest_lock:
            self._latest = {"voltage": voltage, "rms": rms, "gesture": gesture,
                             "features": features}
        # บันทึกลงไฟล์ .jsonl ในเครื่อง + sync ขึ้น Firebase เฉพาะตอนที่ผู้ใช้กด
        # "เริ่มบันทึกแบบ Real-time" ยืนยันแล้วเท่านั้น (self._realtime_logging_active)
        # — ก่อนหน้านี้บันทึกอัตโนมัติตลอดตั้งแต่เปิดแอป (fallback เป็น 'user_01')
        # ทำให้พอปิดแล้วเปิดโปรแกรมใหม่ มันเผลอบันทึกทับ/ต่อของ user เดิมไปเรื่อยๆ
        # โดยยังไม่ทันกด "บันทึก" เลย ตอนนี้ต้องกดยืนยัน user + กดเริ่มบันทึกเองก่อน
        # ถึงจะเขียนข้อมูลจริง
        if self._realtime_logging_active:
            # features เป็น None ได้ (ช่วงแรกที่ window ยังไม่เต็ม) — log_sample() รองรับ
            # keyword พวกนี้เป็น None ได้อยู่แล้ว (จะไม่เขียน key นั้นลง record เลย ดู
            # firebase_sync._add_extra_features) ไม่ต้องเช็ค None เองตรงนี้ซ้ำ
            f = features or {}
            self.firebase.log_sample(
                self._current_user_id, voltage, rms, gesture, t,
                session_id=self._current_session_id,
                mav=f.get("mav"), variance=f.get("variance"),
                waveform_length=f.get("waveform_length"), zero_crossing=f.get("zero_crossing"),
                slope_sign_change=f.get("slope_sign_change"), iemg=f.get("iemg"),
                person_category=self._current_person_category)
        # ป้อนเข้าตัวบันทึก Event กำมือด้วยเสมอ — เช็ค enabled ภายในตัวเอง (no-op ถ้าปิดอยู่)
        self.event_recorder.add_sample(t, voltage, rms, gesture, datetime.now(), features)
        self._drive_servo(rms, features)

    def _drive_servo(self, rms, features):
        """ตัดสินใจว่าจะสั่ง servo ด้วยอะไร — เรียกทุก sample จาก _on_emg_sample (เธรด
        EMGStreamThread) งานเบามาก ไม่บล็อก (predict() ของโมเดลเล็กๆ ใช้เวลาไม่กี่ไมโครวินาที)

        โหมดเริ่มต้น ('threshold'): ใช้ RMS เทียบ threshold ตรงๆ เหมือนระบบเดิมทุกประการ —
        ไม่มีการเปลี่ยนพฤติกรรมเดิมเลยถ้าผู้ใช้ไม่ได้กดสลับโหมดเองที่ Dashboard

        โหมด ML ('ml_model'): ให้โมเดล multi-feature ที่เทรนไว้ทำนาย gesture จาก feature
        ล่าสุด แล้วส่งไปสั่ง servo ผ่าน update_gesture() แทน — มี fallback กลับไปใช้ RMS
        threshold ให้อัตโนมัติเฉพาะ sample นั้นๆ ถ้าเข้าเงื่อนไขใดเงื่อนไขหนึ่งต่อไปนี้ (ไม่ใช่
        all-or-nothing แบบเปลี่ยนโหมดทั้งระบบ): ยังไม่มี window feature พร้อม (features is
        None), โมเดลทำนายไม่ได้ (exception/โมเดลไม่พร้อม), หรือโมเดลความมั่นใจต่ำกว่า
        self._ML_SERVO_MIN_CONFIDENCE — กันมือเทียมค้างท่าผิดตอนโมเดลลังเล ดีกว่าปล่อยให้
        ทำตามคำทำนายที่ไม่มั่นใจไปเลย

        โหมด threshold — ใช้ features['rms'] (RMS แบบ windowed จาก sliding-window feature
        extractor ตัวเดียวกับที่ป้อนเข้า ML — คำนวณทุก 0.1วิ จาก window 0.2วิ) แทน raw rms
        รายแซมเปิลที่แกว่งเยอะ (จาก per-sample moving RMS filter ที่ 500Hz) ถ้า features
        ยังไม่พร้อม (None — เช่นเพิ่งเริ่มสตรีม window แรกยังไม่ครบ) fallback ไปใช้ raw rms
        ชั่วคราวเหมือนเดิม กันมือค้างไม่ตอบสนองตอนเริ่มสตรีมใหม่ๆ"""
        if self.servo_source_var.get() == "ml_model" and features is not None:
            try:
                pred, conf = self.trainer.predict_with_confidence(
                    features, self._current_person_category)
            except Exception:
                pred, conf = None, None
            if pred is not None and (conf is None or conf >= self._ML_SERVO_MIN_CONFIDENCE):
                self.servo.update_gesture(pred)
                return
        # sync threshold ล่าสุดจากแอปเข้า servo ก่อนสั่งจริงทุกครั้ง — เผื่อผู้ใช้เพิ่งปรับ
        # slider หรือโหลด calibration.json ใหม่ ระหว่างที่แอปกำลังรันอยู่ ส่งทั้ง 2 เส้น
        # ตรงๆ (set_thresholds) แทน set_threshold ตัวเดียว เพราะตอนนี้แอปคำนวณเส้นบน/ล่าง
        # จากข้อมูล calibration จริงของ user แต่ละคนแยกกันแล้ว ไม่ใช่แค่ threshold - margin
        # คงที่เสมอไป
        self.servo.set_thresholds(self.threshold, self.release_threshold)
        smoothed_rms = features["rms"] if features is not None else rms
        self.servo.update(smoothed_rms)

    def _on_gesture_event_saved(self, event):
        """Callback จาก GestureEventRecorder — รันบนเธรดของ EMGStreamThread (ไม่ใช่ main
        thread) เมื่อ 'เหตุการณ์กำมือ' 1 ครั้งบันทึกจบแล้ว ปลอดภัยที่จะทำ I/O หนักๆ
        (เขียนไฟล์ + คิว sync Firebase) ตรงนี้ได้เลยโดยไม่บล็อก UI — แต่การแตะ Tk widget
        ต้อง marshal กลับ main thread ผ่าน self.after() เสมอ"""
        try:
            self.firebase.log_dataset_event(self._current_user_id, event,
                                             sample_rate=self.stream.fs, threshold=self.threshold)
            self.firebase.start()  # เผื่อยังไม่เคยเริ่มเธรด sync — เรียกซ้ำได้ (idempotent)
            ok, err = True, None
        except Exception as e:
            ok, err = False, str(e)

        def _update_ui():
            n = self.event_recorder.n_events_recorded
            if ok:
                self._log(self.dash_activity_log,
                           f"🎬 บันทึก Event กำมือ #{n} สำเร็จ — {event['n_samples']} samples "
                           f"({event['start_time']} → {event['end_time']}) ของ [{self._current_user_id}]")
            else:
                self._log(self.dash_activity_log, f"❌ บันทึก Event กำมือ #{n} ไม่สำเร็จ: {err}")
            if hasattr(self, "dash_event_count_lbl"):
                self.dash_event_count_lbl.config(text=str(n))
        self.after(0, _update_ui)

    # ─── UI Builder ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # ttk 'clam' theme ยังจำเป็นสำหรับ Combobox/Progressbar ที่ใช้ในหน้าอื่น
        # (Notebook ไม่ได้ใช้แล้วหลังเปลี่ยนมาเป็น sidebar navigation)
        style = ttk.Style(self)
        style.theme_use("clam")

        # ── Top header bar
        header = tk.Frame(self, bg=COLORS["surface"], height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        logo_row = tk.Frame(header, bg=COLORS["surface"])
        logo_row.pack(side="left", padx=18, pady=8)
        tk.Label(logo_row, text="🦾", bg=COLORS["surface"],
                 font=("TH Sarabun New", 26)).pack(side="left", padx=(0, 8))
        title_col = tk.Frame(logo_row, bg=COLORS["surface"])
        title_col.pack(side="left")
        tk.Label(title_col, text="มือเทียม EMG Control System", bg=COLORS["surface"],
                 fg=COLORS["text"], font=FONT_H1).pack(anchor="w")
        tk.Label(title_col, text="Machine Learning & Real-time Control", bg=COLORS["surface"],
                 fg=COLORS["text_dim"], font=("TH Sarabun New", 10)).pack(anchor="w")

        fb_pill = tk.Frame(header, bg=COLORS["card"])
        fb_pill.pack(side="right", padx=18, pady=16, ipadx=12, ipady=4)
        tk.Label(fb_pill, text="Firebase:", bg=COLORS["card"],
                 fg=COLORS["text_dim"], font=FONT_TH).pack(side="left")
        self.fb_pill_lbl = tk.Label(fb_pill, textvariable=self.firebase_var, bg=COLORS["card"],
                                     fg=COLORS["success"], font=FONT_TH_B)
        self.fb_pill_lbl.pack(side="left", padx=(4, 0))

        # ── Body: sidebar + page area
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, bg=COLORS["surface"], width=230)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        content = tk.Frame(body, bg=COLORS["bg"])
        content.pack(side="left", fill="both", expand=True)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self.pages = {}
        for key in ("dashboard", "calib", "ml", "compare", "datalog", "settings"):
            f = tk.Frame(content, bg=COLORS["bg"])
            f.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = f

        self._build_sidebar_nav(sidebar)
        self._build_dashboard(self.pages["dashboard"])
        self._build_calibration(self.pages["calib"])
        self._build_ml_tab(self.pages["ml"])
        self._build_model_comparison_page(self.pages["compare"])
        self._build_data_log_page(self.pages["datalog"])
        self._build_settings_page(self.pages["settings"])
        self._show_page("dashboard")

        # ── Status bar
        sbar = tk.Frame(self, bg=COLORS["surface"], height=28)
        sbar.pack(fill="x", side="bottom")
        sbar.pack_propagate(False)
        tk.Label(sbar, textvariable=self.status_var,
                 bg=COLORS["surface"], fg=COLORS["accent2"],
                 font=FONT_TH).pack(side="left", padx=12)

    def _build_sidebar_nav(self, parent):
        """แถบเมนูซ้าย — สลับหน้าโดยไม่ต้องใช้ ttk.Notebook (เพื่อให้ได้หน้าตาแบบ
        sidebar navigation ตามดีไซน์ใหม่) พร้อมการ์ดผู้ใช้งาน + สถานะระบบด้านล่าง"""
        nav_items = [
            ("dashboard", "🏠", "Dashboard"),
            ("calib",     "🎯", "Calibration Tool"),
            ("ml",        "🧠", "Machine Learning"),
            ("compare",   "📊", "สรุปผลโมเดล"),
            ("datalog",   "📋", "Data Log"),
            ("settings",  "⚙️", "Settings"),
        ]
        tk.Frame(parent, bg=COLORS["surface"], height=10).pack()
        self.nav_buttons = {}
        for key, icon, label in nav_items:
            btn = tk.Button(parent, text=f"   {icon}   {label}", anchor="w",
                             bg=COLORS["surface"], fg=COLORS["text_dim"],
                             activebackground=COLORS["card"], activeforeground=COLORS["text"],
                             relief="flat", font=FONT_TH_B, cursor="hand2", bd=0,
                             command=lambda k=key: self._show_page(k))
            btn.pack(fill="x", padx=10, pady=2, ipady=8)
            self.nav_buttons[key] = btn

        # ── การ์ดผู้ใช้งาน (แสดงชื่อ user ที่ยืนยันแล้ว — self._current_user_id)
        self._card_label(parent, "👥 ผู้ใช้งาน")
        user_card = tk.Frame(parent, bg=COLORS["card"])
        user_card.pack(fill="x", padx=16, pady=(2, 6), ipady=8)
        row1 = tk.Frame(user_card, bg=COLORS["card"])
        row1.pack(fill="x", padx=10)
        tk.Label(row1, text="👤", bg=COLORS["card"], font=("TH Sarabun New", 16)).pack(side="left")
        self.sidebar_user_lbl = tk.Label(row1, text="—", bg=COLORS["card"],
                                          fg=COLORS["text"], font=FONT_TH_B)
        self.sidebar_user_lbl.pack(side="left", padx=6)
        self.sidebar_user_badge = tk.Label(row1, text="ไม่ได้ยืนยัน", bg=COLORS["danger"],
                                            fg="#ffffff", font=("TH Sarabun New", 9, "bold"))
        self.sidebar_user_badge.pack(side="right", ipadx=4)
        # แสดงประเภทบุคคลที่กำลังเลือกไว้คู่กับ user (ดู self._current_person_category /
        # _on_person_category_change) — อัพเดตทุกเฟรมใน _tick() เช่นเดียวกับ sidebar_user_lbl
        self.sidebar_person_category_lbl = tk.Label(
            user_card, text="ประเภท: —", bg=COLORS["card"], fg=COLORS["text_dim"],
            font=("TH Sarabun New", 10), anchor="w")
        self.sidebar_person_category_lbl.pack(fill="x", padx=10, pady=(4, 0))
        tk.Button(user_card, text="เปลี่ยนผู้ใช้งาน", bg=COLORS["surface"], fg=COLORS["text"],
                  activebackground=COLORS["surface"], relief="flat", font=("TH Sarabun New", 10),
                  cursor="hand2", command=self._open_change_user_dialog
                  ).pack(fill="x", padx=10, pady=(6, 0), ipady=3)

        # ── สถานะระบบ
        self._card_label(parent, "🩺 สถานะระบบ")
        self.status_rows = {}
        for key, label in (("emg", "เซ็นเซอร์ EMG"), ("servo", "Servo Motor"),
                            ("firebase", "Firebase"), ("system", "ระบบ")):
            row = tk.Frame(parent, bg=COLORS["surface"])
            row.pack(fill="x", padx=16, pady=1)
            tk.Label(row, text=label, bg=COLORS["surface"], fg=COLORS["text_dim"],
                     font=("TH Sarabun New", 10)).pack(side="left")
            lbl = tk.Label(row, text="—", bg=COLORS["surface"], fg=COLORS["text_dim"],
                           font=("TH Sarabun New", 10, "bold"))
            lbl.pack(side="right")
            self.status_rows[key] = lbl

        tk.Label(parent, text="v1.2.0", bg=COLORS["surface"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 8)).pack(side="bottom", pady=8)

    def _show_page(self, key):
        self.pages[key].tkraise()
        for k, btn in self.nav_buttons.items():
            active = (k == key)
            btn.config(bg=COLORS["accent"] if active else COLORS["surface"],
                       fg="#ffffff" if active else COLORS["text_dim"])
        self.current_page = key

    # ─── Calibration Tab ─────────────────────────────────────────────────────
    def _build_calibration(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True, padx=14, pady=12)

        # ── แถบตั้งค่าด้านบน: 4 ช่องเรียงแนวนอน (ผู้ใช้ / ประเภทบุคคล+บันทึก /
        # จัดการข้อมูล / จำลองเก็บข้อมูลอัตโนมัติ) แทนพาเนลซ้ายแนวตั้งเดิม — เห็นการตั้งค่า
        # ทั้งหมดในแถบเดียวก่อนลงไปที่กราฟ/ปุ่มบันทึกด้านล่าง (ทุกตัวแปร/วิดเจ็ตยังชื่อเดิม
        # ทุกจุด มีแค่ตำแหน่งบนจอที่เปลี่ยน โค้ดส่วนอื่นที่ .config() ใส่วิดเจ็ตเหล่านี้จึง
        # ทำงานเหมือนเดิมทุกประการ)
        top_strip = tk.Frame(outer, bg=COLORS["surface"], highlightthickness=1,
                              highlightbackground=COLORS["accent2"])
        top_strip.pack(fill="x", pady=(0, 14))

        col_user = self._top_card(
            top_strip, "👤 ข้อมูลผู้ใช้",
            help_text="วิธีใช้:\n"
                       "1. กด 'กำมือ' แล้วบีบมือค้าง\n"
                       "2. กด 'คลายมือ' แล้วคลายมือ\n"
                       "3. ใส่รหัสผู้ใช้ (User) แล้วกด 'บันทึก' "
                       "เพื่อยืนยัน user + เก็บชุด Calibration ที่กำมือ/คลายมือ "
                       "ไว้ส่งขึ้น Firebase\n"
                       "4. กด 'เริ่มบันทึกแบบ Real-time' ถ้าต้องการบันทึก "
                       "สัญญาณ EMG ทุก sample ต่อเนื่องขึ้น Firebase ด้วย "
                       "(ปิดไว้เป็นค่าเริ่มต้นเสมอ ต้องกดเองทุกครั้งที่เปิดแอป)")
        col_person = self._top_card(top_strip, "🏷 ประเภทบุคคล & บันทึก")
        col_manage = self._top_card(top_strip, "🗂 จัดการข้อมูลผู้ใช้")
        col_auto = self._top_card(
            top_strip, "🤖 จำลองเก็บข้อมูลอัตโนมัติ",
            help_text="กดเริ่มแล้วโปรแกรมจะนับถอยหลังสลับท่าให้เองทีละรอบ:\n"
                       "  1) คลายมือ (พัก) ค้างไว้ 5 วินาที\n"
                       "  2) กำมือ ค้างไว้ 5 วินาที\n"
                       "ทำซ้ำตามจำนวนรอบที่ตั้งไว้ โดย label ของแต่ละช่วง "
                       "จะถูกกำหนดให้อัตโนมัติตามจังหวะนับถอยหลัง (ไม่ต้องกด "
                       "'กำมือ'/'คลายมือ' เองทีละครั้ง) — ต้องใส่/เลือกรหัส "
                       "ผู้ใช้ก่อนเริ่มเสมอ และควรขยับมือจริงตามจังหวะบนจอ "
                       "ระหว่างที่นับถอยหลัง เพื่อให้ label ตรงกับสัญญาณจริง", last=True)

        # -- คอลัมน์ 1: ข้อมูลผู้ใช้ ---------------------------------------------
        tk.Label(col_user, text="กรุณาเพิ่ม User", bg=COLORS["surface"],
                 fg=COLORS["warning"], font=FONT_TH_B).pack(anchor="w", pady=(2, 2))
        self.user_id_entry = self._inline_field(col_user, "รหัสผู้ใช้:", self.user_id_var)
        # ยืนยันรหัสผู้ใช้ตอนกด Enter หรือคลิกออกจากช่องเท่านั้น (ไม่ใช่ทุกตัวอักษรที่พิมพ์)
        # ก่อนหน้านี้ _tick() อ่าน self.user_id_var ตรงๆ ทุกเฟรม (~25fps) แล้วส่งให้
        # _on_emg_sample ไปบันทึกไฟล์/sync ขึ้น Firebase ทันที ทำให้ระหว่างพิมพ์ทีละตัว
        # เช่น "T" -> "Te" -> "Tes" -> "Test" แต่ละตัวอักษรกลายเป็น user_id แยกกันที่ถูก
        # เก็บ/sync ขึ้น Firebase ไปด้วย (เห็นเป็น node แยกกันเต็มไปหมด) ทั้งที่ยังพิมพ์ไม่เสร็จ
        self.user_id_entry.bind("<Return>", self._confirm_user_id)
        self.user_id_entry.bind("<FocusOut>", self._confirm_user_id)
        # ปุ่มเลือกรหัสผู้ใช้จากรายชื่อที่เคยบันทึกไว้บน Firebase — ใช้ dialog เดียวกับ
        # "เปลี่ยนผู้ใช้งาน" ที่ sidebar (เขียนลง self.user_id_var ตัวเดียวกัน) จึงพิมพ์เอง
        # ในช่องด้านบน หรือกดปุ่มนี้เพื่อเลือก/เพิ่มผู้ใช้จากรายชื่อก็ได้ทั้งสองแบบ
        tk.Button(col_user, text="📋  เลือกจากรายชื่อ Firebase", bg=COLORS["surface"],
                  fg=COLORS["accent2"], activebackground=COLORS["surface"], relief="flat",
                  font=("TH Sarabun New", 10), cursor="hand2",
                  command=self._open_change_user_dialog
                  ).pack(anchor="w", pady=(4, 0))

        # -- คอลัมน์ 2: ประเภทบุคคล + บันทึก --------------------------------------
        # ประเภทบุคคล — บันทึกคู่กับทุก sample ขึ้น Firebase และป้อนเข้าโมเดลกำ-คลายมือเป็น
        # feature เพิ่มเติม (ดู self._current_person_category / _on_person_category_change)
        # เลือกจาก dropdown ได้เลย มีผลทันที ไม่ต้องกดยืนยันแบบรหัสผู้ใช้ (เลือกจาก dropdown
        # เป็น action เดียวจบ ไม่มีสถานะ "พิมพ์ค้างอยู่กลางคัน" แบบช่องพิมพ์ข้อความ)
        self.person_category_combo = self._inline_field(
            col_person, "ประเภทบุคคล:", self.person_category_var, values=PERSON_CATEGORY_VALUES,
            command=self._on_person_category_change)
        self.btn_save = self._btn(col_person, "💾  บันทึก",
                                   COLORS["accent"], self._save_calibration)
        self.btn_save.pack(pady=(8, 0), ipady=6, fill="x")

        # -- คอลัมน์ 3: จัดการข้อมูลผู้ใช้ ------------------------------------------
        # ปุ่มเริ่ม/หยุดบันทึกสัญญาณ EMG แบบต่อเนื่องขึ้น Firebase — แยกจากปุ่ม "บันทึก"
        # ด้านบนโดยตั้งใจ: "บันทึก" ยืนยันตัวตน user + เซฟชุด calibration ที่กำมือ/คลายมือ
        # ไว้ (ทำครั้งเดียวจบ) ส่วนปุ่มนี้คือสวิตช์เปิด/ปิดการบันทึกสัญญาณสดต่อเนื่อง — ปิด
        # ไว้เป็นค่าเริ่มต้นเสมอทุกครั้งที่เปิดแอปใหม่ ต้องกดเองถึงจะเริ่มเขียนข้อมูลจริง
        # (แก้ปัญหาที่เคยบันทึกทับ user คนก่อนอัตโนมัติตั้งแต่เปิดโปรแกรม โดยไม่ทันกดอะไรเลย)
        self.btn_toggle_realtime = self._btn(col_manage, "▶  เริ่มบันทึกแบบ Real-time",
                                              COLORS["success"], self._toggle_realtime_logging)
        self.btn_toggle_realtime.config(state="disabled")
        self.btn_toggle_realtime.pack(pady=(2, 6), ipady=6, fill="x")

        # ปุ่มลบข้อมูลผู้ใช้แบบเด็ดขาด (ทั้ง Firebase + local) — เพิ่มมาเพราะ Firebase
        # Console ลบ node ที่มีข้อมูลเยอะๆ ไม่ได้เอง (ล็อกเป็น read-only ให้ 'เลือก key ที่มี
        # record น้อยกว่านี้') ต้องยิงลบผ่าน API แทน ดู FirebaseSyncManager.delete_user_data
        # กดแล้วเปิด dialog แสดงรายชื่อผู้ใช้ทั้งหมดบน Firebase ให้ติ๊กเลือกลบได้ทีละคน
        # หรือหลายคนพร้อมกัน (ไม่ผูกกับช่องรหัสผู้ใช้ด้านบนอีกต่อไป)
        self.btn_delete_user = self._btn(col_manage, "🗑  ลบข้อมูลผู้ใช้ (เลือกได้หลายคน)",
                                          COLORS["danger"], self._open_delete_users_dialog)
        self.btn_delete_user.pack(ipady=6, fill="x")

        # -- คอลัมน์ 4: จำลองเก็บข้อมูลอัตโนมัติ — นับถอยหลังสลับคลายมือ/กำมือให้เอง ------
        cycles_row = tk.Frame(col_auto, bg=COLORS["surface"])
        cycles_row.pack(fill="x", pady=(2, 4))
        tk.Label(cycles_row, text="จำนวนรอบ:", bg=COLORS["surface"],
                 fg=COLORS["text"], font=FONT_TH).pack(side="left")
        self.auto_sim_cycles_var = tk.StringVar(value="5")
        tk.Spinbox(cycles_row, from_=1, to=100, textvariable=self.auto_sim_cycles_var,
                   width=5, bg=COLORS["card"], fg=COLORS["text"], relief="flat",
                   buttonbackground=COLORS["card"], insertbackground=COLORS["text"],
                   font=FONT_TH).pack(side="left", padx=6)

        self.auto_sim_status_lbl = tk.Label(
            col_auto, text="พร้อมเริ่ม — พัก 5 วิ / กำมือ 5 วิ ต่อรอบ",
            bg=COLORS["surface"], fg=COLORS["text_dim"], font=("TH Sarabun New", 10),
            wraplength=230, justify="left")
        self.auto_sim_status_lbl.pack(anchor="w", pady=(0, 6))

        auto_btn_row = tk.Frame(col_auto, bg=COLORS["surface"])
        auto_btn_row.pack(fill="x")
        self.btn_auto_sim = self._btn(auto_btn_row, "▶  เริ่มจำลองเก็บข้อมูล",
                                       COLORS["accent2"], self._start_auto_simulate)
        self.btn_auto_sim.pack(side="left", padx=(0, 4), ipady=6, fill="x", expand=True)
        self.btn_auto_sim_stop = self._btn(auto_btn_row, "⏹", COLORS["text_dim"],
                                            lambda: self._stop_auto_simulate(finished=False))
        self.btn_auto_sim_stop.config(state="disabled")
        self.btn_auto_sim_stop.pack(side="left", ipady=6)

        # ── กราฟ EMG + สถิติ + threshold + ปุ่มบันทึก + log ของการทำงาน ──────────
        # เต็มความกว้างหน้าจอ (เดิมอยู่ในพาเนล "right" แคบๆ ข้างพาเนลซ้าย 280px)
        top_r = tk.Frame(outer, bg=COLORS["bg"])
        top_r.pack(fill="x", pady=(0, 4))
        tk.Label(top_r, text="สัญญาณ EMG แบบ Real-time",
                 bg=COLORS["bg"], fg=COLORS["text"], font=FONT_H2).pack(side="left")
        tk.Label(top_r, text="(เส้นแดง = กำมือ (บน) / เส้นเหลือง = พัก (ล่าง) — ตรงกลางคือช่วง noise)",
                 bg=COLORS["bg"], fg=COLORS["text_dim"], font=FONT_TH).pack(side="left", padx=8)

        self.calib_canvas = tk.Canvas(outer, bg=COLORS["card"],
                                      height=240, highlightthickness=1,
                                      highlightbackground=COLORS["border"])
        self.calib_canvas.pack(fill="x", pady=4)

        # Stats row
        stats_row = tk.Frame(outer, bg=COLORS["bg"])
        stats_row.pack(fill="x", pady=4)
        self.calib_emg_lbl  = self._stat_box(stats_row, "EMG ปัจจุบัน", "0.0", COLORS["emg_line"])
        self.calib_count_lbl= self._stat_box(stats_row, "Samples ที่บันทึก", "0", COLORS["accent2"])

        # Threshold slider
        thr_row = tk.Frame(outer, bg=COLORS["bg"])
        thr_row.pack(fill="x", pady=(4, 0))
        tk.Label(thr_row, text="Threshold:", bg=COLORS["bg"],
                 fg=COLORS["text"], font=FONT_TH).pack(side="left")
        # หน่วยเป็นโวลต์ (RMS envelope จริงจาก SEN0240+ADS1115 อยู่ราวๆ 0.01–0.5V)
        self.thr_slider = tk.Scale(thr_row, from_=0.01, to=EMG_RANGE_V, resolution=0.005,
                                   orient="horizontal", bg=COLORS["bg"],
                                   fg=COLORS["text"], troughcolor=COLORS["card"],
                                   highlightthickness=0, font=FONT_TH,
                                   command=self._on_threshold)
        self.thr_slider.set(self.threshold)
        self.thr_slider.pack(side="left", fill="x", expand=True, padx=8)
        self.thr_lbl = tk.Label(thr_row, text=f"{self.threshold:.3f}V",
                                 bg=COLORS["bg"], fg=COLORS["warning"],
                                 font=FONT_TH_B, width=7)
        self.thr_lbl.pack(side="left")

        # Buttons
        btn_row = tk.Frame(outer, bg=COLORS["bg"])
        btn_row.pack(fill="x", pady=10)

        self.btn_grip = self._btn(btn_row, "✊  กำมือ",    COLORS["grip"],
                                  lambda: self._start_record("กำมือ"))
        self.btn_grip.pack(side="left", padx=4, ipady=6, fill="x", expand=True)

        self.btn_release = self._btn(btn_row, "🖐  คลายมือ", COLORS["release"],
                                     lambda: self._start_record("คลายมือ"))
        self.btn_release.pack(side="left", padx=4, ipady=6, fill="x", expand=True)

        self.btn_stop = self._btn(btn_row, "⏹  หยุด", COLORS["text_dim"],
                                   self._stop_record)
        self.btn_stop.pack(side="left", padx=4, ipady=6, fill="x", expand=True)
        self.btn_stop.config(state="disabled")

        self.btn_clear = self._btn(btn_row, "🗑  ล้างข้อมูล", COLORS["danger"],
                                    self._clear_calibration)
        self.btn_clear.pack(side="left", padx=4, ipady=6, fill="x", expand=True)

        self.btn_auto_thr = self._btn(btn_row, "🎯  คำนวณ Threshold อัตโนมัติ",
                                       COLORS["accent"], self._suggest_threshold_from_calibration)
        self.btn_auto_thr.pack(side="left", padx=4, ipady=6, fill="x", expand=True)

        # Log box
        tk.Label(outer, text="📋 บันทึกการทำงาน", bg=COLORS["bg"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w")
        log_frame = tk.Frame(outer, bg=COLORS["card"])
        log_frame.pack(fill="both", expand=True, pady=(2, 0))
        self.log_box = tk.Text(log_frame, bg=COLORS["card"], fg=COLORS["text"],
                                font=FONT_MONO, height=6, state="disabled",
                                relief="flat", wrap="word")
        sb = tk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sb.pack(side="right", fill="y")

    def _top_card(self, parent, title, help_text=None, last=False):
        """ช่องหนึ่งในแถบตั้งค่าด้านบนของหน้า Calibration Tool — หัวข้อสีฟ้าไซแอน
        (+ปุ่ม ❔ ถ้ามี help_text) ตามด้วยเนื้อหาที่ผู้เรียกจะ .pack() ต่อลงในเฟรมที่คืนกลับ
        ไป ทุกช่องกว้างเท่ากัน (uniform group) และมีเส้นแบ่งบางๆ คั่นระหว่างช่อง ยกเว้น
        ช่องสุดท้าย (last=True) ที่ไม่ต้องมีเส้นแบ่งต่อท้าย"""
        card = tk.Frame(parent, bg=COLORS["surface"])
        card.pack(side="left", fill="both", expand=True, padx=16, pady=12)
        head = tk.Frame(card, bg=COLORS["surface"])
        head.pack(fill="x")
        tk.Label(head, text=title, bg=COLORS["surface"], fg=COLORS["accent2"],
                 font=FONT_TH_B, wraplength=200, justify="left", anchor="w"
                 ).pack(side="left", fill="x", expand=True)
        if help_text:
            self._help_button(head, title, help_text).pack(side="right", anchor="n")
        if not last:
            tk.Frame(parent, bg=COLORS["border"], width=1).pack(side="left", fill="y", pady=14)
        return card

    def _inline_field(self, parent, label, var, values=None, command=None):
        """แถวป้ายกำกับ + ช่องกรอก (หรือ dropdown ถ้าใส่ values) วางป้ายไว้บนช่องกรอก —
        กระชับกว่า _labeled_entry/_labeled_combobox เดิม (ป้ายอยู่ข้างซ้าย) เหมาะกับ
        คอลัมน์แคบๆ ในแถบตั้งค่าด้านบนของหน้า Calibration Tool"""
        row = tk.Frame(parent, bg=COLORS["surface"])
        row.pack(fill="x", pady=(4, 2))
        tk.Label(row, text=label, bg=COLORS["surface"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 10)).pack(anchor="w")
        if values is not None:
            widget = ttk.Combobox(row, textvariable=var, values=values, state="readonly",
                                   font=FONT_TH)
            if command:
                widget.bind("<<ComboboxSelected>>", command)
        else:
            widget = tk.Entry(row, textvariable=var, bg=COLORS["card"], fg=COLORS["text"],
                               insertbackground=COLORS["text"], relief="flat", font=FONT_TH)
        widget.pack(fill="x", pady=(2, 0), ipady=3)
        return widget

    # ─── Dashboard Tab ────────────────────────────────────────────────────────
    def _build_dashboard(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True, padx=14, pady=12)

        # ── Row 1: stat cards ────────────────────────────────────────────
        row1 = tk.Frame(outer, bg=COLORS["bg"])
        row1.pack(fill="x")

        # Card 1: ค่า EMG ปัจจุบัน + ring gauge
        c1 = tk.Frame(row1, bg=COLORS["card"])
        c1.pack(side="left", fill="both", expand=True, padx=(0, 8), ipady=8)
        tk.Label(c1, text="ค่า EMG ปัจจุบัน", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=FONT_TH).pack(anchor="w", padx=12, pady=(6, 0))
        c1_body = tk.Frame(c1, bg=COLORS["card"])
        c1_body.pack(fill="x", padx=12)
        self.dash_emg_val_lbl = tk.Label(c1_body, text="0.000 V", bg=COLORS["card"],
                                          fg=COLORS["emg_line"], font=("Courier New", 18, "bold"))
        self.dash_emg_val_lbl.pack(side="left")
        self.dash_ring_canvas = tk.Canvas(c1_body, width=64, height=64, bg=COLORS["card"],
                                           highlightthickness=0)
        self.dash_ring_canvas.pack(side="right")
        tk.Label(c1, text="RMS Value", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9)).pack(anchor="w", padx=12, pady=(0, 6))

        # Card 2: ท่าทางปัจจุบัน
        c2 = tk.Frame(row1, bg=COLORS["card"])
        c2.pack(side="left", fill="both", expand=True, padx=8, ipady=8)
        tk.Label(c2, text="ท่าทางปัจจุบัน", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=FONT_TH).pack(anchor="w", padx=12, pady=(6, 0))
        c2_body = tk.Frame(c2, bg=COLORS["card"])
        c2_body.pack(fill="x", padx=12)
        self.dash_gesture_icon_lbl = tk.Label(c2_body, text="🖐", bg=COLORS["card"],
                                               font=("TH Sarabun New", 22))
        self.dash_gesture_icon_lbl.pack(side="left")
        self.dash_gesture_text_lbl = tk.Label(c2_body, text="OPEN", bg=COLORS["card"],
                                               fg=COLORS["accent2"], font=FONT_TH_B)
        self.dash_gesture_text_lbl.pack(side="left", padx=6)
        self.dash_gesture_conf_lbl = tk.Label(c2, text="ยังไม่มีโมเดล", bg=COLORS["card"],
                                               fg=COLORS["text_dim"], font=("TH Sarabun New", 9))
        self.dash_gesture_conf_lbl.pack(anchor="w", padx=12, pady=(2, 6))

        # Card 3: โมเดลที่ใช้งาน
        c3 = tk.Frame(row1, bg=COLORS["card"])
        c3.pack(side="left", fill="both", expand=True, padx=8, ipady=8)
        tk.Label(c3, text="โมเดลที่ใช้งาน", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=FONT_TH).pack(anchor="w", padx=12, pady=(6, 0))
        self.dash_model_lbl = tk.Label(c3, text="—", bg=COLORS["card"], fg=COLORS["text"],
                                        font=FONT_TH_B, wraplength=160, justify="left")
        self.dash_model_lbl.pack(anchor="w", padx=12)
        self.dash_model_ensemble_lbl = tk.Label(c3, text="", bg=COLORS["accent"], fg="#ffffff",
                                                 font=("TH Sarabun New", 9, "bold"))
        self.dash_model_ensemble_lbl.pack(anchor="w", padx=12, pady=(2, 6))

        # Card 4: สถานะ Firebase
        c4 = tk.Frame(row1, bg=COLORS["card"])
        c4.pack(side="left", fill="both", expand=True, padx=8, ipady=8)
        tk.Label(c4, text="สถานะ Firebase", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=FONT_TH).pack(anchor="w", padx=12, pady=(6, 0))
        c4_body = tk.Frame(c4, bg=COLORS["card"])
        c4_body.pack(fill="x", padx=12)
        tk.Label(c4_body, text="☁", bg=COLORS["card"], font=("TH Sarabun New", 18)).pack(side="left")
        self.dash_fb_status_lbl = tk.Label(c4_body, text="Offline", bg=COLORS["card"],
                                            fg=COLORS["danger"], font=FONT_TH_B)
        self.dash_fb_status_lbl.pack(side="left", padx=6)
        self.dash_fb_sync_lbl = tk.Label(c4, text="ยังไม่เริ่ม sync", bg=COLORS["card"],
                                          fg=COLORS["text_dim"], font=("TH Sarabun New", 9))
        self.dash_fb_sync_lbl.pack(anchor="w", padx=12, pady=(2, 6))

        # Card 5: บันทึกข้อมูล (samples ที่เทรนอยู่ตอนนี้)
        c5 = tk.Frame(row1, bg=COLORS["card"])
        c5.pack(side="left", fill="both", expand=True, padx=(8, 0), ipady=8)
        tk.Label(c5, text="บันทึกข้อมูล", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=FONT_TH).pack(anchor="w", padx=12, pady=(6, 0))
        c5_body = tk.Frame(c5, bg=COLORS["card"])
        c5_body.pack(fill="x", padx=12)
        tk.Label(c5_body, text="🗄", bg=COLORS["card"], font=("TH Sarabun New", 16)).pack(side="left")
        self.dash_samples_lbl = tk.Label(c5_body, text="0", bg=COLORS["card"], fg=COLORS["accent"],
                                          font=("Courier New", 16, "bold"))
        self.dash_samples_lbl.pack(side="left", padx=6)
        tk.Label(c5, text="Samples (โมเดลปัจจุบัน)", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9)).pack(anchor="w", padx=12, pady=(2, 6))

        # ── Row 1.1: แหล่งสั่งงาน Servo จริง (RMS Threshold เดิม / ML Model ใหม่) ──
        servo_src_row = tk.Frame(outer, bg=COLORS["card"])
        servo_src_row.pack(fill="x", pady=(10, 0))
        tk.Label(servo_src_row, text="⚙ แหล่งสั่งงาน Servo จริง:", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(side="left", padx=(12, 10), pady=8)
        tk.Radiobutton(servo_src_row, text="RMS Threshold (ค่าเริ่มต้น)",
                       variable=self.servo_source_var, value="threshold",
                       command=self._on_servo_source_change, bg=COLORS["card"], fg=COLORS["text"],
                       selectcolor=COLORS["surface"], activebackground=COLORS["card"],
                       font=FONT_TH, cursor="hand2").pack(side="left", padx=(0, 12))
        tk.Radiobutton(servo_src_row, text="ML Model (จากแท็บ Machine Learning)",
                       variable=self.servo_source_var, value="ml_model",
                       command=self._on_servo_source_change, bg=COLORS["card"], fg=COLORS["text"],
                       selectcolor=COLORS["surface"], activebackground=COLORS["card"],
                       font=FONT_TH, cursor="hand2").pack(side="left")
        self.servo_source_status_lbl = tk.Label(
            servo_src_row, text="", bg=COLORS["card"], fg=COLORS["text_dim"],
            font=("TH Sarabun New", 9))
        self.servo_source_status_lbl.pack(side="left", padx=12)

        # ── Row 1.5: Feature panel (RMS/MAV/Variance/WL/ZC/SSC/IEMG) ──
        # เปิด/ปิดแสดงผลได้ผ่าน checkbox (ดีฟอลต์เปิด) — ข้อกำหนดเรื่อง GUI ต้องแสดง
        # feature ที่คำนวณได้ทั้งหมด และเปิด/ปิดการแสดงผลได้
        feat_toggle_row = tk.Frame(outer, bg=COLORS["bg"])
        feat_toggle_row.pack(fill="x", pady=(8, 0))
        tk.Checkbutton(feat_toggle_row, text="แสดง Feature ที่คำนวณได้ (RMS/MAV/Variance/"
                                              "Waveform Length/Zero Crossing/Slope Sign "
                                              "Change/IEMG)",
                       variable=self.show_features_var, command=self._toggle_feature_panel,
                       bg=COLORS["bg"], fg=COLORS["text_dim"], selectcolor=COLORS["card"],
                       activebackground=COLORS["bg"], font=("TH Sarabun New", 10),
                       cursor="hand2").pack(anchor="w")

        self.feature_panel = tk.Frame(outer, bg=COLORS["card"])
        self.feature_panel.pack(fill="x", pady=(4, 0))
        self._feature_value_lbls = {}
        # กริด 4 คอลัมน์ (แต่ก่อนมี 5 feature ใส่แถวเดียวพอ ตอนนี้ 7 ตัวใส่แถวเดียวจะแน่นไป
        # เลยแบ่งเป็น 2 แถว 4 คอลัมน์แทน — ช่องสุดท้ายว่างไว้เฉยๆ)
        _feature_display = [("rms", "RMS (V)"), ("mav", "MAV (V)"), ("variance", "Variance"),
                             ("waveform_length", "Waveform Length"),
                             ("zero_crossing", "Zero Crossing"),
                             ("slope_sign_change", "Slope Sign Change"), ("iemg", "IEMG")]
        for col in range(4):
            self.feature_panel.grid_columnconfigure(col, weight=1, uniform="feat")
        for i, (key, label_text) in enumerate(_feature_display):
            row, col = divmod(i, 4)
            cell = tk.Frame(self.feature_panel, bg=COLORS["card"])
            cell.grid(row=row, column=col, sticky="nsew", padx=10, pady=8)
            tk.Label(cell, text=label_text, bg=COLORS["card"], fg=COLORS["text_dim"],
                     font=("TH Sarabun New", 9)).pack(anchor="w")
            val_lbl = tk.Label(cell, text="—", bg=COLORS["card"], fg=COLORS["accent2"],
                                font=("Courier New", 13, "bold"))
            val_lbl.pack(anchor="w")
            self._feature_value_lbls[key] = val_lbl
        if not self.show_features_var.get():
            self.feature_panel.pack_forget()

        # ── Row 2: กราฟ real-time (ซ้าย) + gauge/ควบคุมการบันทึก (ขวา) ─────
        row2 = tk.Frame(outer, bg=COLORS["bg"])
        row2.pack(fill="both", expand=True, pady=(10, 0))

        chart_card = tk.Frame(row2, bg=COLORS["card"])
        chart_card.pack(side="left", fill="both", expand=True, padx=(0, 8))
        chart_head = tk.Frame(chart_card, bg=COLORS["card"])
        chart_head.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(chart_head, text="กราฟสัญญาณ EMG แบบ Real-time", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(side="left")
        tk.Label(chart_head, text="ช่วงเวลา:", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 10)).pack(side="right", padx=(0, 4))
        dash_window_cb = ttk.Combobox(chart_head, textvariable=self.dash_window_var, width=10,
                                       values=list(self._dash_window_options.keys()),
                                       font=("TH Sarabun New", 10), state="readonly")
        dash_window_cb.pack(side="right")
        self.dash_chart_canvas = tk.Canvas(chart_card, bg=COLORS["bg"], height=280,
                                            highlightthickness=0)
        self.dash_chart_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        right_col = tk.Frame(row2, bg=COLORS["bg"], width=260)
        right_col.pack(side="left", fill="y")
        right_col.pack_propagate(False)

        gauge_card = tk.Frame(right_col, bg=COLORS["card"])
        gauge_card.pack(fill="x")
        tk.Label(gauge_card, text="ระดับสัญญาณ EMG", bg=COLORS["card"], fg=COLORS["text"],
                 font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 0))
        self.dash_gauge_canvas = tk.Canvas(gauge_card, width=230, height=140, bg=COLORS["card"],
                                            highlightthickness=0)
        self.dash_gauge_canvas.pack(padx=10, pady=6)

        rec_card = tk.Frame(right_col, bg=COLORS["card"])
        rec_card.pack(fill="x", pady=(10, 0))
        tk.Label(rec_card, text="ควบคุมการบันทึก", bg=COLORS["card"], fg=COLORS["text"],
                 font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 6))
        self.dash_realtime_btn = self._btn(rec_card, "▶  เริ่มบันทึกแบบ Real-time",
                                            COLORS["success"], self._toggle_realtime_logging)
        self.dash_realtime_btn.pack(fill="x", padx=12, pady=(0, 6), ipady=6)
        self._btn(rec_card, "🗑  ล้างข้อมูล", COLORS["card"], self._clear_calibration
                  ).pack(fill="x", padx=12, pady=(0, 12), ipady=6)

        # ── การ์ดบันทึก Event กำมืออัตโนมัติ (แยกจากการบันทึก Real-time ต่อเนื่องด้านบน)
        event_card = tk.Frame(right_col, bg=COLORS["card"])
        event_card.pack(fill="x", pady=(10, 0))
        tk.Label(event_card, text="🎬 บันทึก Event กำมืออัตโนมัติ", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(event_card, text="พบกำมือ → เริ่มบันทึก (รวม 5 วิก่อนหน้า) จนคลายมือ\n"
                                    "ค้างครบ 5 วิ ถึงจะถือว่าจบ 1 เหตุการณ์", bg=COLORS["card"],
                 fg=COLORS["text_dim"], font=("TH Sarabun New", 9), justify="left"
                 ).pack(anchor="w", padx=12, pady=(0, 6))
        self.dash_event_toggle_btn = self._btn(event_card, "▶  เปิดใช้งาน",
                                                COLORS["success"], self._toggle_event_recorder)
        self.dash_event_toggle_btn.config(state="disabled")
        self.dash_event_toggle_btn.pack(fill="x", padx=12, pady=(0, 6), ipady=6)
        event_status_row = tk.Frame(event_card, bg=COLORS["card"])
        event_status_row.pack(fill="x", padx=12, pady=(0, 12))
        self.dash_event_status_lbl = tk.Label(event_status_row, text="⚪ ปิดอยู่", bg=COLORS["card"],
                                               fg=COLORS["text_dim"], font=("TH Sarabun New", 10))
        self.dash_event_status_lbl.pack(side="left")
        tk.Label(event_status_row, text="บันทึกแล้ว:", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9)).pack(side="right", padx=(0, 4))
        self.dash_event_count_lbl = tk.Label(event_status_row, text="0", bg=COLORS["card"],
                                              fg=COLORS["accent"], font=("Courier New", 10, "bold"))
        self.dash_event_count_lbl.pack(side="right")

        # ── Row 3: ผลการทดสอบโมเดล / Confusion Matrix / Log กิจกรรม ──────
        row3 = tk.Frame(outer, bg=COLORS["bg"])
        row3.pack(fill="both", expand=True, pady=(10, 0))

        metrics_card = tk.Frame(row3, bg=COLORS["card"])
        metrics_card.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(metrics_card, text="🏆 ผลการทดสอบโมเดล (Test Set 20%)", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 6))
        grid = tk.Frame(metrics_card, bg=COLORS["card"])
        grid.pack(fill="x", padx=12, pady=(0, 10))
        self.dash_metric_lbls = {}
        metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "Kappa"]
        for i, name in enumerate(metric_names):
            cell = tk.Frame(grid, bg=COLORS["bg"])
            cell.grid(row=i // 3, column=i % 3, sticky="nsew", padx=4, pady=4)
            grid.grid_columnconfigure(i % 3, weight=1)
            tk.Label(cell, text=name, bg=COLORS["bg"], fg=COLORS["text_dim"],
                     font=("TH Sarabun New", 9)).pack(anchor="w", padx=8, pady=(4, 0))
            lbl = tk.Label(cell, text="—", bg=COLORS["bg"], fg=COLORS["accent2"],
                           font=("Courier New", 13, "bold"))
            lbl.pack(anchor="w", padx=8, pady=(0, 4))
            self.dash_metric_lbls[name] = lbl
        self.dash_metrics_summary_lbl = tk.Label(metrics_card, text="ยังไม่มีข้อมูลพอประเมิน",
                                                  bg=COLORS["card"], fg=COLORS["text_dim"],
                                                  font=("TH Sarabun New", 9))
        self.dash_metrics_summary_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        cm_card = tk.Frame(row3, bg=COLORS["card"])
        cm_card.pack(side="left", fill="both", expand=True, padx=8)
        tk.Label(cm_card, text="🔲 Confusion Matrix", bg=COLORS["card"], fg=COLORS["text"],
                 font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 6))
        self.dash_cm_frame = tk.Frame(cm_card, bg=COLORS["card"])
        self.dash_cm_frame.pack(padx=12, pady=(0, 6))
        self.dash_cm_summary_lbl = tk.Label(cm_card, text="", bg=COLORS["card"],
                                             fg=COLORS["text_dim"], font=("TH Sarabun New", 9))
        self.dash_cm_summary_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        log_card = tk.Frame(row3, bg=COLORS["card"])
        log_card.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(log_card, text="📄 Log การเทรน / ทำนายแบบ Real-time", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 6))
        log_frame = tk.Frame(log_card, bg=COLORS["card"])
        log_frame.pack(fill="both", expand=True, padx=10)
        self.dash_activity_log = tk.Text(log_frame, bg=COLORS["bg"], fg=COLORS["text"],
                                          font=("Courier New", 9), height=6, state="disabled",
                                          relief="flat", wrap="word")
        sb3 = tk.Scrollbar(log_frame, command=self.dash_activity_log.yview)
        self.dash_activity_log.configure(yscrollcommand=sb3.set)
        self.dash_activity_log.pack(side="left", fill="both", expand=True)
        sb3.pack(side="right", fill="y")
        tk.Button(log_card, text="ดู Log ทั้งหมด →", bg=COLORS["card"], fg=COLORS["accent"],
                  activebackground=COLORS["card"], relief="flat", font=("TH Sarabun New", 10),
                  cursor="hand2", command=lambda: self._show_page("datalog")
                  ).pack(anchor="w", padx=12, pady=(4, 10))

        self._draw_ring_gauge(self.dash_ring_canvas, 0)
        self._draw_semi_gauge(self.dash_gauge_canvas, 0, EMG_RANGE_V)
        self._render_dash_confusion(None)
        self._update_servo_source_status()  # ตั้งข้อความสถานะเริ่มต้น ("ใช้ RMS Threshold แบบเดิม")

    # ─── Gauge drawing helpers (Canvas — ไม่ต้องพึ่ง matplotlib) ─────────────
    def _draw_ring_gauge(self, canvas, pct):
        canvas.delete("all")
        w, h = 64, 64
        pad = 6
        color = (COLORS["success"] if pct < 60 else
                 COLORS["warning"] if pct < 85 else COLORS["danger"])
        canvas.create_oval(pad, pad, w - pad, h - pad, outline=COLORS["bg"], width=8)
        extent = -359.9 * max(0.0, min(1.0, pct / 100.0))
        if abs(extent) > 0.5:
            canvas.create_arc(pad, pad, w - pad, h - pad, start=90, extent=extent,
                               style="arc", outline=color, width=8)
        canvas.create_text(w / 2, h / 2, text=f"{pct:.0f}%", fill=COLORS["text"],
                            font=("TH Sarabun New", 12, "bold"))

    def _draw_semi_gauge(self, canvas, value, max_value):
        canvas.delete("all")
        w, h = 230, 140
        cx, cy = w / 2, h - 24
        r = 85
        canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=180,
                           style="arc", outline=COLORS["bg"], width=16)
        frac = max(0.0, min(1.0, value / max_value if max_value else 0))
        color = (COLORS["success"] if frac < 0.6 else
                 COLORS["warning"] if frac < 0.85 else COLORS["danger"])
        if frac > 0.01:
            canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=180 * (1 - frac),
                               extent=180 * frac, style="arc", outline=color, width=16)
        angle = math.radians(180 * (1 - frac))
        nx, ny = cx + r * 0.75 * math.cos(angle), cy - r * 0.75 * math.sin(angle)
        canvas.create_line(cx, cy, nx, ny, fill=COLORS["text"], width=3)
        canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=COLORS["text"], outline="")
        canvas.create_text(cx, cy - r - 16, text=f"{value:.3f} V", fill=COLORS["text"],
                            font=("Courier New", 14, "bold"))
        canvas.create_text(cx - r + 6, cy + 14, text="0", fill=COLORS["text_dim"],
                            font=("TH Sarabun New", 9))
        canvas.create_text(cx + r - 10, cy + 14, text=f"{max_value:.1f}", fill=COLORS["text_dim"],
                            font=("TH Sarabun New", 9))

    def _draw_dash_chart(self):
        canvas = self.dash_chart_canvas
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10 or len(self.dash_chart_history) < 2:
            return
        pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 20
        plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b

        # เส้น threshold บน (activate) + ล่าง (release)
        thr_y_hi = pad_t + plot_h * (1 - min(1.0, self.threshold / EMG_RANGE_V))
        thr_y_lo = pad_t + plot_h * (1 - min(1.0, self.release_threshold / EMG_RANGE_V))
        canvas.create_line(pad_l, thr_y_hi, W - pad_r, thr_y_hi, fill=COLORS["threshold"],
                            dash=(4, 3), width=1)
        canvas.create_line(pad_l, thr_y_lo, W - pad_r, thr_y_lo, fill=COLORS["warning"],
                            dash=(4, 3), width=1)

        # แกน Y (0 / mid / max)
        for frac, label in ((0, "0.0"), (0.5, f"{EMG_RANGE_V/2:.1f}"), (1.0, f"{EMG_RANGE_V:.1f}")):
            y = pad_t + plot_h * (1 - frac)
            canvas.create_line(pad_l, y, W - pad_r, y, fill=COLORS["border"], dash=(2, 4))
            canvas.create_text(pad_l - 6, y, text=label, fill=COLORS["text_dim"],
                               font=("Courier New", 8), anchor="e")

        n = len(self.dash_chart_history)
        # ลดจำนวนจุดที่วาดจริงให้ไม่เกิน MAX_PLOT_POINTS เสมอ ไม่ว่าจะเลือกช่วงเวลานาน
        # แค่ไหน (buffer อาจมีถึง 7,500 จุดตอนเลือก "5 นาที") — วาดทุกจุดจริงๆ ทุกเฟรม
        # คือสาเหตุหลักที่ทำให้โปรแกรมหน่วง ลดจุดที่วาดลงแต่ยังเห็นรูปทรงสัญญาณเหมือนเดิม
        MAX_PLOT_POINTS = 240
        step = max(1, n // MAX_PLOT_POINTS)
        history_list = list(self.dash_chart_history)  # deque ไม่รองรับ slice ตรงๆ ต้องแปลงก่อน
        sampled = history_list[::step]
        n_sampled = len(sampled)
        pts = []
        for i, v in enumerate(sampled):
            x = pad_l + plot_w * (i / max(1, n_sampled - 1))
            y = pad_t + plot_h * (1 - min(1.0, v / EMG_RANGE_V))
            pts.extend([x, y])
        if len(pts) >= 4:
            canvas.create_line(*pts, fill=COLORS["emg_line"], width=2, smooth=True)

    # ─── หน้า สรุปผลโมเดล (เปรียบเทียบผลลัพธ์ของทุกโมเดล/ensemble ที่เคยเทรน) ────
    def _build_model_comparison_page(self, parent):
        # ห่อทั้งหน้าด้วย Canvas+Scrollbar เพราะเนื้อหาเยอะ (5 การ์ดสรุป + ตารางรวมทุก metric
        # + คำอธิบาย metric + การ์ดรายละเอียด 6 อัน + radar chart) อาจไม่พอดีจอเล็ก
        outer_canvas = tk.Canvas(parent, bg=COLORS["bg"], highlightthickness=0)
        outer_scroll = tk.Scrollbar(parent, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=outer_scroll.set)
        outer_canvas.pack(side="left", fill="both", expand=True)
        outer_scroll.pack(side="right", fill="y")

        wrap = tk.Frame(outer_canvas, bg=COLORS["bg"])
        wrap_window = outer_canvas.create_window((0, 0), window=wrap, anchor="nw")
        wrap.bind("<Configure>", lambda e: outer_canvas.configure(scrollregion=outer_canvas.bbox("all")))
        outer_canvas.bind("<Configure>", lambda e: outer_canvas.itemconfig(wrap_window, width=e.width))

        def _on_wheel(e):
            outer_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        outer_canvas.bind("<Enter>", lambda e: outer_canvas.bind_all("<MouseWheel>", _on_wheel))
        outer_canvas.bind("<Leave>", lambda e: outer_canvas.unbind_all("<MouseWheel>"))

        wrap.configure(padx=16, pady=14)
        pad = tk.Frame(wrap, bg=COLORS["bg"])  # inner padding wrapper (Frame ไม่รับ padx/pady โดยตรงจาก configure สำหรับ children)
        pad.pack(fill="both", expand=True, padx=0, pady=0)

        # ── หัวข้อ + ปุ่มรีเฟรช/ล้าง ─────────────────────────────────────────────
        header_row = tk.Frame(pad, bg=COLORS["bg"])
        header_row.pack(fill="x", pady=(0, 12))
        title_col = tk.Frame(header_row, bg=COLORS["bg"])
        title_col.pack(side="left", fill="x", expand=True)
        tk.Label(title_col, text="📊 สรุปผลการเปรียบเทียบโมเดล", bg=COLORS["bg"], fg=COLORS["text"],
                 font=FONT_H2).pack(anchor="w")
        self.compare_subtitle_lbl = tk.Label(
            title_col, text="เปรียบเทียบประสิทธิภาพของโมเดลทั้งหมด 0 โมเดล", bg=COLORS["bg"],
            fg=COLORS["text_dim"], font=("TH Sarabun New", 10), wraplength=560, justify="left")
        self.compare_subtitle_lbl.pack(anchor="w", pady=(2, 0))

        ctrl_col = tk.Frame(header_row, bg=COLORS["bg"])
        ctrl_col.pack(side="right")
        self._btn(ctrl_col, "🔄 รีเฟรช", COLORS["card"], self._refresh_model_comparison
                  ).pack(side="left", padx=4, ipadx=4, ipady=4)
        self._btn(ctrl_col, "🗑 ล้างผลเปรียบเทียบ", COLORS["danger"], self._clear_model_comparison
                  ).pack(side="left", padx=(4, 0), ipadx=4, ipady=4)

        # ── การ์ดสรุป 5 อัน (Best Model + จำนวนโมเดล + ช่วงคะแนน + ค่าเฉลี่ย + อัปเดตล่าสุด) ──
        stats_row = tk.Frame(pad, bg=COLORS["bg"])
        stats_row.pack(fill="x", pady=(0, 12))

        best_card = tk.Frame(stats_row, bg=COLORS["accent"])
        best_card.pack(side="left", fill="both", expand=True, padx=(0, 8), ipady=8)
        tk.Label(best_card, text="🏆 Best Model", bg=COLORS["accent"], fg="#ffffff",
                 font=("TH Sarabun New", 10, "bold")).pack(anchor="w", padx=14, pady=(8, 2))
        self.compare_best_name_lbl = tk.Label(best_card, text="—", bg=COLORS["accent"],
                                               fg="#ffffff", font=("TH Sarabun New", 16, "bold"))
        self.compare_best_name_lbl.pack(anchor="w", padx=14)
        self.compare_best_metric_lbl = tk.Label(best_card, text="", bg=COLORS["accent"],
                                                 fg="#e5e2ff", font=("TH Sarabun New", 9))
        self.compare_best_metric_lbl.pack(anchor="w", padx=14)
        best_row2 = tk.Frame(best_card, bg=COLORS["accent"])
        best_row2.pack(fill="x", padx=14, pady=(2, 8))
        self.compare_best_score_lbl = tk.Label(best_row2, text="—", bg=COLORS["accent"],
                                                fg="#ffffff", font=("Courier New", 20, "bold"))
        self.compare_best_score_lbl.pack(side="left")
        self.compare_best_delta_lbl = tk.Label(best_row2, text="", bg=COLORS["accent"],
                                                fg="#d4ffd4", font=("TH Sarabun New", 9, "bold"))
        self.compare_best_delta_lbl.pack(side="right")

        def _stat_card(title, icon=""):
            c = tk.Frame(stats_row, bg=COLORS["card"])
            c.pack(side="left", fill="both", expand=True, padx=8, ipady=8)
            title_lbl = tk.Label(c, text=f"{icon} {title}".strip(), bg=COLORS["card"],
                                  fg=COLORS["text_dim"], font=("TH Sarabun New", 10))
            title_lbl.pack(anchor="w", padx=14, pady=(8, 4))
            val_lbl = tk.Label(c, text="—", bg=COLORS["card"], fg=COLORS["text"],
                                font=("TH Sarabun New", 18, "bold"))
            val_lbl.pack(anchor="w", padx=14)
            sub_lbl = tk.Label(c, text="", bg=COLORS["card"], fg=COLORS["text_dim"],
                                font=("TH Sarabun New", 9), wraplength=180, justify="left")
            sub_lbl.pack(anchor="w", padx=14, pady=(2, 8))
            return title_lbl, val_lbl, sub_lbl

        self.compare_total_title_lbl, self.compare_total_lbl, self.compare_total_sub_lbl = \
            _stat_card("จำนวนโมเดลทั้งหมด", "🔢")
        self.compare_range_title_lbl, self.compare_range_lbl, self.compare_range_sub_lbl = \
            _stat_card("ช่วงคะแนน", "📈")
        self.compare_avg_title_lbl, self.compare_avg_lbl, self.compare_avg_sub_lbl = \
            _stat_card("ค่าเฉลี่ย", "💧")
        self.compare_updated_title_lbl, self.compare_updated_lbl, self.compare_updated_sub_lbl = \
            _stat_card("อัปเดตล่าสุด", "🕒")

        # ── ตารางเปรียบเทียบโมเดลรวม (ทุก metric ในตารางเดียว พร้อมแนวโน้ม) ───────
        table_card = tk.Frame(pad, bg=COLORS["card"])
        table_card.pack(fill="both", expand=True, pady=(0, 12))

        table_header = tk.Frame(table_card, bg=COLORS["card"])
        table_header.pack(fill="x", padx=14, pady=(12, 6))
        title_box = tk.Frame(table_header, bg=COLORS["card"])
        title_box.pack(side="left", fill="x", expand=True)
        tk.Label(title_box, text="📶 เปรียบเทียบประสิทธิภาพโมเดล", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w")
        tk.Label(title_box, text="คลิกที่หัวคอลัมน์เพื่อจัดเรียง — คลิกชื่อโมเดลเพื่อดูรายละเอียด"
                                  "ด้านล่าง", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9)).pack(anchor="w")

        ctrl_box = tk.Frame(table_header, bg=COLORS["card"])
        ctrl_box.pack(side="right")
        tk.Label(ctrl_box, text="จัดเรียงตาม", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9)).pack(side="left", padx=(0, 4))
        self.compare_sort_combo_var = tk.StringVar(value=self._sort_combo_text())
        self.compare_sort_combo = ttk.Combobox(
            ctrl_box, textvariable=self.compare_sort_combo_var, width=22,
            values=self._sort_combo_options(), font=FONT_TH, state="readonly")
        self.compare_sort_combo.pack(side="left", padx=(0, 6))
        self.compare_sort_combo.bind("<<ComboboxSelected>>", self._on_sort_combo_change)
        self._btn(ctrl_box, "🔽 Filter", COLORS["surface"], self._toggle_compare_filter
                  ).pack(side="left", padx=(0, 6), ipadx=4, ipady=4)
        self._btn(ctrl_box, "⬇  ส่งออก", COLORS["surface"], self._export_compare_csv
                  ).pack(side="left", ipadx=4, ipady=4)

        # แถบค้นหา/กรองชื่อโมเดล — สร้างไว้เลยแต่ยังไม่ pack (โผล่ตอนกด Filter เท่านั้น)
        self.compare_filter_row = tk.Frame(table_card, bg=COLORS["card"])
        tk.Label(self.compare_filter_row, text="🔍", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 11)).pack(side="left", padx=(14, 4))
        self.compare_filter_var = tk.StringVar(value="")
        self.compare_filter_entry = tk.Entry(
            self.compare_filter_row, textvariable=self.compare_filter_var, bg=COLORS["surface"],
            fg=COLORS["text"], insertbackground=COLORS["text"], relief="flat", font=FONT_TH)
        self.compare_filter_entry.pack(side="left", fill="x", expand=True, ipady=3, padx=(0, 14))
        self.compare_filter_var.trace_add(
            "write", lambda *_: self._render_compare_table(*self._compare_sorted_items()))

        self.compare_table_frame = tk.Frame(table_card, bg=COLORS["card"])
        self.compare_table_frame.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        # ── คำอธิบายเมตริก (การ์ดเล็ก 6 อัน ด้านล่างตาราง) ────────────────────────
        legend_card = tk.Frame(pad, bg=COLORS["card"])
        legend_card.pack(fill="x", pady=(0, 12))
        tk.Label(legend_card, text="คำอธิบายเมตริก", bg=COLORS["card"], fg=COLORS["text"],
                 font=FONT_TH_B).pack(anchor="w", padx=14, pady=(10, 6))
        legend_row = tk.Frame(legend_card, bg=COLORS["card"])
        legend_row.pack(fill="x", padx=14, pady=(0, 14))
        legend_items = [
            ("🎯", "Accuracy", "ความถูกต้องโดยรวม", "ยิ่งสูงยิ่งดี", COLORS["success"]),
            ("🧲", "Precision", "ความแม่นยำของผลบวก", "ยิ่งสูงยิ่งดี", COLORS["success"]),
            ("📥", "Recall", "ความระลึกถึง (ครอบคลุม)", "ยิ่งสูงยิ่งดี", COLORS["success"]),
            ("⚖️", "F1-score", "ค่าเฉลี่ยสมดุลระหว่าง Precision และ Recall", "ยิ่งสูงยิ่งดี", COLORS["success"]),
            ("👁", "AUC", "ความสามารถในการแยกแยะคลาส", "ยิ่งใกล้ 1 ยิ่งดี", COLORS["success"]),
            ("🏅", "Kappa", "ความสอดคล้องของการจำแนก", "ยิ่งใกล้ 1 ยิ่งดี", COLORS["success"]),
        ]
        for icon, name, desc, hint, hint_color in legend_items:
            cell = tk.Frame(legend_row, bg=COLORS["card"])
            cell.pack(side="left", fill="both", expand=True, padx=6)
            tk.Label(cell, text=f"{icon} {name}", bg=COLORS["card"], fg=COLORS["text"],
                     font=("TH Sarabun New", 10, "bold")).pack(anchor="w")
            tk.Label(cell, text=desc, bg=COLORS["card"], fg=COLORS["text_dim"],
                     font=("TH Sarabun New", 9), wraplength=150, justify="left").pack(anchor="w")
            tk.Label(cell, text=hint, bg=COLORS["card"], fg=hint_color,
                     font=("TH Sarabun New", 9, "bold")).pack(anchor="w")

        # ── รายละเอียดเชิงลึกของโมเดลที่เลือก + Radar Chart (ข้อมูลเสริม) ──────────
        bottom_row = tk.Frame(pad, bg=COLORS["bg"])
        bottom_row.pack(fill="x")

        detail_card = tk.Frame(bottom_row, bg=COLORS["card"])
        detail_card.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.compare_detail_title_lbl = tk.Label(detail_card, text="รายละเอียดเชิงลึกของโมเดลที่เลือก: —",
                                                  bg=COLORS["card"], fg=COLORS["text"], font=FONT_TH_B)
        self.compare_detail_title_lbl.pack(anchor="w", padx=14, pady=(10, 8))
        detail_grid = tk.Frame(detail_card, bg=COLORS["card"])
        detail_grid.pack(fill="x", padx=14, pady=(0, 14))
        self.compare_detail_lbls = {}
        metric_colors = {"Accuracy": COLORS["success"], "Precision": COLORS["accent2"],
                          "Recall": COLORS["accent"], "F1-score": COLORS["warning"],
                          "AUC": "#22d3ee", "Kappa": COLORS["danger"]}
        for i, (name, color) in enumerate(metric_colors.items()):
            cell = tk.Frame(detail_grid, bg=COLORS["bg"])
            cell.grid(row=i // 3, column=i % 3, sticky="nsew", padx=4, pady=4)
            detail_grid.grid_columnconfigure(i % 3, weight=1)
            tk.Label(cell, text=name, bg=COLORS["bg"], fg=color,
                     font=("TH Sarabun New", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
            val_lbl = tk.Label(cell, text="—", bg=COLORS["bg"], fg=COLORS["text"],
                                font=("Courier New", 15, "bold"))
            val_lbl.pack(anchor="w", padx=10)
            tk.Frame(cell, bg=color, height=3).pack(fill="x", padx=10, pady=(4, 10))
            self.compare_detail_lbls[name] = val_lbl

        # ⚠️ บั๊ก UI ที่แก้: เดิม Frame นี้ตั้งแค่ width=340 ไม่ได้ตั้ง height เลย พอสั่ง
        # pack_propagate(False) การ์ดเลยไม่ขยายตามเนื้อหาข้างใน (หัว label + Canvas วาด
        # radar สูง 280px + padding) แต่ไปยืดตามความสูงของ detail_card (การ์ดซ้ายมือ) แทน
        # ซึ่งเตี้ยกว่าที่ Canvas ต้องการจริงมาก (~200px < ~340px ที่ต้องใช้) ทำให้ Canvas
        # ล้นออกไปนอกขอบล่างการ์ดจริง (ตามที่เห็นในภาพ — กราฟถูกตัด/ทับซ้อนกับเนื้อหาด้านล่าง)
        # แก้โดยตั้ง height ให้พอกับเนื้อหาจริง (label ~34px + canvas 280px + pady บน-ล่าง 20px)
        radar_card = tk.Frame(bottom_row, bg=COLORS["card"], width=340, height=345)
        radar_card.pack(side="left", fill="y", padx=(8, 0))
        radar_card.pack_propagate(False)
        tk.Label(radar_card, text="Radar Chart (เปรียบเทียบ Metric)", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=14, pady=(10, 0))
        self.compare_radar_canvas = tk.Canvas(radar_card, bg=COLORS["card"], width=300, height=280,
                                               highlightthickness=0)
        self.compare_radar_canvas.pack(padx=10, pady=10)

        # ── กราฟแท่งเปรียบเทียบอัลกอริทึม (ทุก Metric พร้อมกัน) ─────────────────────
        # ต่างจาก Radar Chart ด้านบน (เทียบทีละโมเดลที่เลือก) — อันนี้วาง "ทุกโมเดล" เรียง
        # เทียบกันเป็นแท่ง แยกกลุ่มตาม metric ใช้แทรกในบทที่ 4 (ผลการทดลอง/เปรียบเทียบ
        # อัลกอริทึม) ของรายงานได้โดยตรง — เป็น matplotlib Figure จริง (ไม่ใช่ Canvas วาดเอง
        # แบบ radar) จึงบันทึกเป็นไฟล์รูปด้วย fig.savefig() ได้
        bar_card = tk.Frame(pad, bg=COLORS["card"])
        bar_card.pack(fill="x", pady=(8, 0))
        bar_head = tk.Frame(bar_card, bg=COLORS["card"])
        bar_head.pack(fill="x", padx=14, pady=(10, 0))
        tk.Label(bar_head, text="📊 กราฟแท่งเปรียบเทียบอัลกอริทึมทั้งหมด (ทุก Metric)",
                 bg=COLORS["card"], fg=COLORS["text"], font=FONT_TH_B).pack(side="left")
        self._btn(bar_head, "💾 บันทึกรูป (ใช้ในรายงาน)", COLORS["surface"],
                   self._save_compare_bar_chart).pack(side="right", ipadx=4, ipady=4)
        self._compare_bar_fig = Figure(figsize=(9.6, 3.6), dpi=100)
        self._compare_bar_fig.patch.set_facecolor(COLORS["card"])
        self._compare_bar_ax = self._compare_bar_fig.add_subplot(111)
        self._compare_bar_canvas = FigureCanvasTkAgg(self._compare_bar_fig, master=bar_card)
        self._compare_bar_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self._refresh_model_comparison()

    def _draw_compare_bar_chart(self):
        """วาดกราฟแท่งกลุ่ม (grouped bar) เทียบทุกโมเดลใน self.model_results พร้อมกัน
        แยกกลุ่มตาม 6 metric (Accuracy/Precision/Recall/F1/AUC/Kappa) — ใช้ตอบโจทย์
        'ต้องมีกราฟเปรียบเทียบอัลกอริทึม' ในบทที่ 4 คู่กับตารางที่มีอยู่แล้ว
        (compare_table_frame) เรียกซ้ำทุกครั้งที่ _refresh_model_comparison() ทำงาน"""
        ax = self._compare_bar_ax
        ax.clear()
        self._style_ml_axes(ax, "", "Score")

        items, _key = self._compare_sorted_items()
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "Kappa"]
        metric_keys = self._compare_metric_map()

        if not items:
            ax.text(0.5, 0.5, "ยังไม่มีผลลัพธ์โมเดลให้เปรียบเทียบ\n(เทรนโมเดลอย่างน้อย 1 ตัวก่อน)",
                    ha="center", va="center", color=COLORS["text_dim"], fontsize=9,
                    transform=ax.transAxes)
            self._compare_bar_canvas.draw_idle()
            return

        # จำกัดไม่เกิน 6 โมเดลบนกราฟเดียว กันแท่งซ้อนกันจนอ่านไม่ออก — โมเดลที่เหลือยังดูได้
        # ในตารางเต็มด้านบน (นี่คือกราฟสรุปภาพรวม ไม่ใช่ตัวแทนข้อมูลทั้งหมด)
        shown = items[:6]
        model_names = [name for name, _ in shown]
        palette = [COLORS["accent"], COLORS["accent2"], COLORS["success"],
                   COLORS["warning"], "#22d3ee", COLORS["danger"]]

        n_models = len(shown)
        n_metrics = len(metrics)
        group_width = 0.8
        bar_width = group_width / max(n_models, 1)
        x_base = range(n_metrics)

        for i, (name, res) in enumerate(shown):
            values = [res[metric_keys[m]] or 0 for m in metrics]
            offsets = [xb - group_width / 2 + bar_width * i + bar_width / 2 for xb in x_base]
            ax.bar(offsets, values, bar_width * 0.92, color=palette[i % len(palette)],
                   label=name)

        ax.set_xticks(list(x_base))
        ax.set_xticklabels(metrics, color=COLORS["text_dim"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n_models, 4),
                  fontsize=7, facecolor=COLORS["card"], edgecolor=COLORS["text_dim"],
                  labelcolor=COLORS["text"])
        self._compare_bar_fig.tight_layout()
        self._compare_bar_canvas.draw_idle()

    def _save_compare_bar_chart(self):
        if not self.model_results:
            messagebox.showinfo("ไม่มีข้อมูล", "ยังไม่มีผลเปรียบเทียบโมเดลให้บันทึกเป็นรูป")
            return
        self._save_figure_png(self._compare_bar_fig, "model_comparison_bar_chart.png")

    def _compare_metric_map(self):
        return {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
                "F1-score": "f1", "AUC": "auc", "Kappa": "kappa"}

    def _sort_combo_options(self):
        opts = []
        for m in ("Accuracy", "Precision", "Recall", "F1-score", "AUC", "Kappa"):
            opts.append(f"{m} (มาก → น้อย)")
            opts.append(f"{m} (น้อย → มาก)")
        return opts

    def _sort_combo_text(self):
        direction = "มาก → น้อย" if self.compare_sort_var.get() == "สูง → ต่ำ" else "น้อย → มาก"
        return f"{self.compare_metric_var.get()} ({direction})"

    def _sync_sort_combo_text(self):
        if hasattr(self, "compare_sort_combo_var"):
            self.compare_sort_combo_var.set(self._sort_combo_text())

    def _on_sort_combo_change(self, event=None):
        text = self.compare_sort_combo_var.get()
        try:
            metric, rest = text.split(" (", 1)
            direction = rest.rstrip(")")
        except ValueError:
            return
        self.compare_metric_var.set(metric)
        self.compare_sort_var.set("สูง → ต่ำ" if direction == "มาก → น้อย" else "น้อย → มาก")
        self._refresh_model_comparison()

    def _set_sort_from_column(self, metric_name):
        """คลิกที่หัวคอลัมน์ metric ในตาราง — ถ้าคลิกซ้ำ metric เดิม สลับทิศทางเรียง
        ถ้าคลิก metric ใหม่ เริ่มที่ 'มาก → น้อย' เสมอ"""
        if self.compare_metric_var.get() == metric_name:
            self.compare_sort_var.set(
                "ต่ำ → สูง" if self.compare_sort_var.get() == "สูง → ต่ำ" else "สูง → ต่ำ")
        else:
            self.compare_metric_var.set(metric_name)
            self.compare_sort_var.set("สูง → ต่ำ")
        self._sync_sort_combo_text()
        self._refresh_model_comparison()

    def _toggle_compare_filter(self):
        if self.compare_filter_row.winfo_ismapped():
            self.compare_filter_row.pack_forget()
            self.compare_filter_var.set("")
        else:
            self.compare_filter_row.pack(fill="x", pady=(0, 6), before=self.compare_table_frame)
            self.compare_filter_entry.focus_set()

    def _toggle_compare_show_all(self):
        self._compare_show_all = not self._compare_show_all
        self._render_compare_table(*self._compare_sorted_items())

    def _export_compare_csv(self):
        items, _key = self._compare_sorted_items()
        if not items:
            messagebox.showinfo("ไม่มีข้อมูล", "ยังไม่มีผลเปรียบเทียบให้ส่งออก")
            return
        path = filedialog.asksaveasfilename(
            title="ส่งออกผลเปรียบเทียบโมเดล", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], initialfile="model_comparison.csv")
        if not path:
            return
        import csv
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "Kappa"]
        metric_keys = self._compare_metric_map()
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["อันดับ", "โมเดล"] + metrics)
                for i, (name, res) in enumerate(items):
                    row = [i + 1, name]
                    for m in metrics:
                        v = res[metric_keys[m]]
                        row.append("" if v is None else round(v, 4))
                    writer.writerow(row)
            self._log(self.dash_activity_log, f"⬇ ส่งออกผลเปรียบเทียบโมเดล {len(items)} รายการ → {path}")
            messagebox.showinfo("สำเร็จ", f"ส่งออกไฟล์ไปที่:\n{path}")
        except Exception as e:
            messagebox.showerror("ส่งออกไม่สำเร็จ", str(e))

    def _save_figure_png(self, fig, default_name, log_widget=None):
        """บันทึกกราฟ matplotlib (Figure) เป็นไฟล์รูป .png ความละเอียดสูง (dpi=200) —
        ใช้สำหรับดึงรูปผลการทดลอง/กราฟเปรียบเทียบอัลกอริทึมไปแทรกในบทที่ 4 ของรายงาน
        โดยไม่ต้องแคปหน้าจอเอง (แคปจอจะได้ความละเอียดต่ำและติดพื้นหลัง UI)"""
        path = filedialog.asksaveasfilename(
            title="บันทึกรูปภาพสำหรับรายงาน", defaultextension=".png",
            filetypes=[("PNG image", "*.png")], initialfile=default_name)
        if not path:
            return
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
            self._log(log_widget or self.dash_activity_log, f"💾 บันทึกรูปภาพ → {path}")
            messagebox.showinfo("สำเร็จ", f"บันทึกรูปไปที่:\n{path}")
        except Exception as e:
            messagebox.showerror("บันทึกไม่สำเร็จ", str(e))

    def _compare_sorted_items(self):
        """คืนค่า (items, key) — items เรียงตาม metric+ทิศทางที่เลือกไว้ รายการที่ไม่มีค่า
        (None) จะอยู่ท้ายสุดเสมอไม่ว่าจะเลือกเรียงทิศไหน"""
        key = self._compare_metric_map()[self.compare_metric_var.get()]
        reverse = self.compare_sort_var.get() == "สูง → ต่ำ"
        items = list(self.model_results.items())

        def sort_key(kv):
            val = kv[1][key]
            if val is None:
                return (1, 0)
            return (0, -val if reverse else val)
        items.sort(key=sort_key)
        return items, key

    def _clear_model_comparison(self):
        self.model_results.clear()
        self.compare_selected_model = None
        self._compare_last_updated = None
        self._refresh_model_comparison()
        self._log(self.dash_activity_log, "🗑 ล้างผลเปรียบเทียบโมเดลทั้งหมดแล้ว")

    def _select_compare_model(self, name):
        self.compare_selected_model = name
        items, key = self._compare_sorted_items()
        self._render_compare_table(items, key)
        self._render_compare_detail()
        self._draw_radar_chart()

    def _refresh_model_comparison(self):
        """เรียกทุกครั้งที่ผลลัพธ์เปลี่ยน (เทรนโมเดลใหม่เสร็จ) หรือผู้ใช้เปลี่ยน metric/sort
        — อัพเดตทุก widget ในหน้านี้ให้ตรงกันหมด"""
        if not hasattr(self, "compare_table_frame"):
            return  # หน้านี้ยังสร้างไม่เสร็จ (ป้องกันเผื่อถูกเรียกก่อนเวลา)
        items, key = self._compare_sorted_items()
        if self.compare_selected_model not in self.model_results:
            self.compare_selected_model = items[0][0] if items else None
        self.compare_subtitle_lbl.config(
            text=f"เปรียบเทียบประสิทธิภาพของโมเดลทั้งหมด {len(items)} โมเดล")
        self._sync_sort_combo_text()
        self._render_compare_stats(items, key)
        self._render_compare_table(items, key)
        self._render_compare_detail()
        self._draw_radar_chart()
        self._draw_compare_bar_chart()

    def _render_compare_stats(self, items, key):
        is_pct = key in ("accuracy", "precision", "recall", "f1")
        metric_name = self.compare_metric_var.get()
        self.compare_range_title_lbl.config(text=f"📈 ช่วงคะแนน ({metric_name})")
        self.compare_avg_title_lbl.config(text=f"💧 ค่าเฉลี่ย ({metric_name})")

        def fmt(v):
            return "—" if v is None else (f"{v*100:.2f}%" if is_pct else f"{v:.3f}")

        if self._compare_last_updated:
            self.compare_updated_lbl.config(text=self._compare_last_updated.strftime("%d %b %Y"))
            self.compare_updated_sub_lbl.config(text=self._compare_last_updated.strftime("%H:%M:%S น."))
        else:
            self.compare_updated_lbl.config(text="—")
            self.compare_updated_sub_lbl.config(text="")

        if not items:
            self.compare_best_name_lbl.config(text="—")
            self.compare_best_metric_lbl.config(text="")
            self.compare_best_score_lbl.config(text="—")
            self.compare_best_delta_lbl.config(text="")
            self.compare_total_lbl.config(text="0")
            self.compare_total_sub_lbl.config(text="ยังไม่มีผลเปรียบเทียบ")
            self.compare_range_lbl.config(text="—")
            self.compare_range_sub_lbl.config(text="")
            self.compare_avg_lbl.config(text="—")
            self.compare_avg_sub_lbl.config(text="")
            return

        valid = sorted(((n, r[key]) for n, r in items if r[key] is not None),
                        key=lambda x: -x[1])
        best_name, best_val = valid[0] if valid else (items[0][0], None)
        second_val = valid[1][1] if len(valid) > 1 else None

        self.compare_best_name_lbl.config(text=best_name)
        self.compare_best_metric_lbl.config(text=metric_name)
        self.compare_best_score_lbl.config(text=fmt(best_val))
        if best_val is not None and second_val is not None:
            delta = (best_val - second_val) * (100 if is_pct else 1)
            self.compare_best_delta_lbl.config(text=f"↑ +{delta:.2f}{'%' if is_pct else ''} ดีกว่าอันดับ 2")
        else:
            self.compare_best_delta_lbl.config(text="")

        self.compare_total_lbl.config(text=str(len(items)))
        self.compare_total_sub_lbl.config(text="โมเดล")

        if valid:
            vals = [v for _, v in valid]
            self.compare_range_lbl.config(text=f"{fmt(min(vals))} - {fmt(max(vals))}")
            self.compare_range_sub_lbl.config(text="ช่วงคะแนน")
            avg = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            self.compare_avg_lbl.config(text=fmt(avg))
            self.compare_avg_sub_lbl.config(text=f"ส่วนเบี่ยงเบนมาตรฐาน {fmt(std)}")
        else:
            self.compare_range_lbl.config(text="—")
            self.compare_range_sub_lbl.config(text="")
            self.compare_avg_lbl.config(text="—")
            self.compare_avg_sub_lbl.config(text="")

    _COMPARE_METRIC_ICONS = {"Accuracy": "🎯", "Precision": "🧲", "Recall": "📥",
                              "F1-score": "⚖️", "AUC": "👁", "Kappa": "🏅"}
    _COMPARE_BAR_COLORS = {"Accuracy": "#4f8fff", "Precision": "#a78bfa", "Recall": "#4ade80",
                            "F1-score": "#fb923c", "AUC": "#22d3ee", "Kappa": "#facc15"}

    def _render_compare_table(self, items, key):
        for w in self.compare_table_frame.winfo_children():
            w.destroy()
        if not items:
            tk.Label(self.compare_table_frame, text="ยังไม่มีผลลัพธ์ — ไปเทรนโมเดลในแท็บ Machine "
                                                      "Learning ก่อน", bg=COLORS["card"],
                     fg=COLORS["text_dim"], font=("TH Sarabun New", 10)).pack(anchor="w", pady=20)
            return

        filter_text = self.compare_filter_var.get().strip().lower() if hasattr(self, "compare_filter_var") else ""
        filtered = [(n, r) for n, r in items if filter_text in n.lower()] if filter_text else items
        if not filtered:
            tk.Label(self.compare_table_frame, text=f"ไม่พบโมเดลที่ตรงกับคำค้นหา '{filter_text}'",
                     bg=COLORS["card"], fg=COLORS["text_dim"], font=("TH Sarabun New", 10)
                     ).pack(anchor="w", pady=20)
            return

        display_items = filtered if self._compare_show_all else filtered[:10]

        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "Kappa"]
        metric_keys = self._compare_metric_map()
        is_pct_map = {"Accuracy": True, "Precision": True, "Recall": True,
                      "F1-score": True, "AUC": False, "Kappa": False}
        best_per_metric = {}
        for m in metrics:
            vals = [r[metric_keys[m]] for _, r in items if r[metric_keys[m]] is not None]
            best_per_metric[m] = max(vals) if vals else None

        METRIC_COL_W = 118

        # ── หัวตาราง (คลิกได้เพื่อจัดเรียง) ──
        header = tk.Frame(self.compare_table_frame, bg=COLORS["card"])
        header.pack(fill="x", pady=(0, 4))
        tk.Label(header, text="#", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9, "bold"), width=3, anchor="w").pack(side="left")
        tk.Label(header, text="โมเดล", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9, "bold"), width=24, anchor="w").pack(side="left")
        for m in metrics:
            col_lbl = tk.Label(
                header, text=f"{self._COMPARE_METRIC_ICONS[m]} {m}", bg=COLORS["card"],
                fg=(COLORS["accent2"] if m == self.compare_metric_var.get() else COLORS["text_dim"]),
                font=("TH Sarabun New", 9, "bold"), width=14, anchor="w", cursor="hand2")
            col_lbl.pack(side="left")
            col_lbl.bind("<Button-1>", lambda e, mm=m: self._set_sort_from_column(mm))
        tk.Label(header, text="แนวโน้ม", bg=COLORS["card"], fg=COLORS["text_dim"],
                 font=("TH Sarabun New", 9, "bold"), width=9, anchor="w").pack(side="left")
        tk.Frame(self.compare_table_frame, bg=COLORS["border"], height=1).pack(fill="x", pady=(0, 4))

        medals = ["🥇", "🥈", "🥉"]
        for i, (name, res) in enumerate(display_items):
            row = tk.Frame(self.compare_table_frame, bg=COLORS["card"], cursor="hand2")
            row.pack(fill="x", pady=3)
            rank_text = medals[i] if (i < 3 and not filter_text) else str(i + 1)
            tk.Label(row, text=rank_text, bg=COLORS["card"], fg=COLORS["text"],
                     font=("TH Sarabun New", 11), width=3).pack(side="left")

            is_selected = (name == self.compare_selected_model)
            name_lbl = tk.Label(row, text=name, bg=COLORS["card"],
                                 fg=COLORS["accent2"] if is_selected else COLORS["text"],
                                 font=("TH Sarabun New", 10, "bold" if is_selected else "normal"),
                                 width=24, anchor="w", cursor="hand2")
            name_lbl.pack(side="left")

            for m in metrics:
                cell = tk.Frame(row, bg=COLORS["card"], width=METRIC_COL_W, height=22)
                cell.pack(side="left")
                self._render_metric_bar_cell(cell, res[metric_keys[m]], is_pct_map[m],
                                              self._COMPARE_BAR_COLORS[m], best_per_metric[m])

            spark = tk.Canvas(row, width=76, height=24, bg=COLORS["card"], highlightthickness=0)
            spark.pack(side="left", padx=(4, 0))
            self._draw_sparkline(spark, res.get("history", []))

            for widget in (row, name_lbl):
                widget.bind("<Button-1>", lambda e, n=name: self._select_compare_model(n))

        if len(filtered) > 10:
            more_row = tk.Frame(self.compare_table_frame, bg=COLORS["card"])
            more_row.pack(fill="x", pady=(8, 2))
            btn_text = (f"▲  แสดงแค่ 10 อันดับแรก" if self._compare_show_all
                        else f"▼  ดูทั้งหมด {len(filtered)} โมเดล")
            tk.Button(more_row, text=btn_text, bg=COLORS["surface"], fg=COLORS["accent"],
                      activebackground=COLORS["surface"], relief="flat", font=("TH Sarabun New", 10),
                      cursor="hand2", command=self._toggle_compare_show_all
                      ).pack(anchor="center", ipady=4, ipadx=10)

    def _render_metric_bar_cell(self, parent, val, is_pct, color, best_val):
        """สีของแท่ง (bar) ต้องตรงกับสีประจำ metric column นั้นเสมอ (self._COMPARE_BAR_COLORS
        — เช่น Accuracy = สีฟ้าเสมอทุกแถว) ⚠️ เดิมแถวที่ได้คะแนนดีที่สุดในคอลัมน์นั้นจะถูก
        เปลี่ยนแท่งเป็นสีทอง (gold) แทนสีประจำคอลัมน์ ทำให้สีในคอลัมน์เดียวกันไม่ตรงกันหมด
        ตามที่อาจารย์ทักท้วง — ตัดการเปลี่ยนสีแท่งออก ใช้สีประจำคอลัมน์เสมอ ส่วนการไฮไลต์ค่า
        ที่ดีที่สุดในคอลัมน์ ย้ายไปทำที่ตัวเลข (ตัวหนา + สีทอง) แทน ไม่แตะสีแท่งอีกต่อไป"""
        parent.pack_propagate(False)
        inner = tk.Frame(parent, bg=COLORS["card"])
        inner.pack(fill="both", expand=True)
        bar_bg = tk.Frame(inner, bg=COLORS["bg"], width=44, height=8)
        bar_bg.pack(side="left", pady=7)
        bar_bg.pack_propagate(False)
        is_best = val is not None and best_val is not None and val == best_val
        if val is not None:
            frac = max(0.0, min(1.0, val))
            tk.Frame(bar_bg, bg=color).place(relx=0, rely=0, relwidth=max(0.02, frac), relheight=1)
        val_text = "N/A" if val is None else (f"{val*100:.2f}%" if is_pct else f"{val:.3f}")
        tk.Label(inner, text=val_text, bg=COLORS["card"],
                 fg=COLORS["gold"] if is_best else COLORS["text"],
                 font=("Courier New", 9, "bold" if is_best else "bold")).pack(side="left", padx=(6, 0))

    def _draw_sparkline(self, canvas, history):
        """วาดกราฟเส้นเล็กๆ (sparkline) แสดงแนวโน้มคะแนนของโมเดลนี้ในการเทรนแต่ละรอบล่าสุด
        (เก็บย้อนหลังไว้ใน model_results[name]['history'] — ดู _refresh_ml_metrics)"""
        canvas.delete("all")
        w, h = 76, 24
        pts = [v for v in history if v is not None]
        if len(pts) < 2:
            canvas.create_line(4, h / 2, w - 4, h / 2, fill=COLORS["text_dim"], width=1)
            return
        lo, hi = min(pts), max(pts)
        span = (hi - lo) or 1.0
        n = len(pts)
        coords = []
        for i, v in enumerate(pts):
            x = 4 + (w - 8) * (i / (n - 1))
            y = h - 4 - (h - 8) * ((v - lo) / span)
            coords.extend([x, y])
        color = COLORS["success"] if pts[-1] >= pts[0] else COLORS["danger"]
        canvas.create_line(*coords, fill=color, width=2, smooth=True)

    def _render_compare_detail(self):
        name = self.compare_selected_model
        self.compare_detail_title_lbl.config(text=f"รายละเอียดเชิงลึกของโมเดลที่เลือก: {name or '—'}")
        key_map = self._compare_metric_map()
        pct_names = {"Accuracy", "Precision", "Recall", "F1-score"}
        if not name or name not in self.model_results:
            for lbl in self.compare_detail_lbls.values():
                lbl.config(text="—")
            return
        res = self.model_results[name]
        for label_name, lbl in self.compare_detail_lbls.items():
            val = res[key_map[label_name]]
            if val is None:
                lbl.config(text="N/A")
            elif label_name in pct_names:
                lbl.config(text=f"{val*100:.2f}%")
            else:
                lbl.config(text=f"{val:.3f}")

    def _draw_radar_chart(self):
        """วาด radar chart (hexagon) เทียบ 6 metric ของโมเดลที่เลือกอยู่ในตาราง"""
        canvas = self.compare_radar_canvas
        canvas.delete("all")
        cx, cy, r = 150, 140, 100
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "Kappa"]
        keys = ["accuracy", "precision", "recall", "f1", "auc", "kappa"]
        n = len(metrics)
        angle_step = 2 * math.pi / n

        # กริดพื้นหลัง (วงแหวน 25/50/75/100%) + แกนเส้นตรง + label
        for frac in (0.25, 0.5, 0.75, 1.0):
            pts = []
            for i in range(n):
                angle = -math.pi / 2 + i * angle_step
                pts.extend([cx + r * frac * math.cos(angle), cy + r * frac * math.sin(angle)])
            canvas.create_polygon(*pts, outline=COLORS["border"], fill="", width=1)
        for i, name in enumerate(metrics):
            angle = -math.pi / 2 + i * angle_step
            x, y = cx + r * math.cos(angle), cy + r * math.sin(angle)
            canvas.create_line(cx, cy, x, y, fill=COLORS["border"])
            lx, ly = cx + (r + 20) * math.cos(angle), cy + (r + 20) * math.sin(angle)
            canvas.create_text(lx, ly, text=name, fill=COLORS["text_dim"], font=("TH Sarabun New", 9))

        name = self.compare_selected_model
        if not name or name not in self.model_results:
            canvas.create_text(cx, cy, text="เลือกโมเดล\nจากตารางด้านบน", fill=COLORS["text_dim"],
                                font=("TH Sarabun New", 10), justify="center")
            return

        res = self.model_results[name]
        values = [max(0.0, min(1.0, res[k])) if res[k] is not None else 0.0 for k in keys]
        pts = []
        for i, val in enumerate(values):
            angle = -math.pi / 2 + i * angle_step
            pts.extend([cx + r * val * math.cos(angle), cy + r * val * math.sin(angle)])
        # ไม่ใช้ stipple (เพราะ render ไม่แน่นอนข้าม platform โดยเฉพาะ Windows) — ใช้แค่
        # เส้นขอบ + จุดที่มุมแทน อ่านง่ายกว่าและแสดงผลเหมือนกันทุกเครื่อง
        canvas.create_polygon(*pts, outline=COLORS["success"], fill="", width=2)
        for i in range(0, len(pts), 2):
            canvas.create_oval(pts[i] - 3, pts[i + 1] - 3, pts[i] + 3, pts[i + 1] + 3,
                                fill=COLORS["success"], outline="")

    # ─── หน้า Data Log (รายการ log กิจกรรมทั้งหมด) ───────────────────────────
    def _build_data_log_page(self, parent):
        wrap = tk.Frame(parent, bg=COLORS["bg"])
        wrap.pack(fill="both", expand=True, padx=16, pady=14)
        tk.Label(wrap, text="📋 Data Log", bg=COLORS["bg"], fg=COLORS["text"],
                 font=FONT_H2).pack(anchor="w")
        tk.Label(wrap, text="ประวัติกิจกรรมทั้งหมดของแอป (บันทึก/เทรน/ทำนาย/sync ฯลฯ)",
                 bg=COLORS["bg"], fg=COLORS["text_dim"], font=("TH Sarabun New", 10)
                 ).pack(anchor="w", pady=(0, 8))
        frame = tk.Frame(wrap, bg=COLORS["card"])
        frame.pack(fill="both", expand=True)
        self.datalog_text = tk.Text(frame, bg=COLORS["card"], fg=COLORS["text"],
                                     font=("Courier New", 10), state="disabled",
                                     relief="flat", wrap="word")
        sb = tk.Scrollbar(frame, command=self.datalog_text.yview)
        self.datalog_text.configure(yscrollcommand=sb.set)
        self.datalog_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        sb.pack(side="right", fill="y")

    # ─── หน้า Settings ────────────────────────────────────────────────────────
    def _build_settings_page(self, parent):
        wrap = tk.Frame(parent, bg=COLORS["bg"])
        wrap.pack(fill="both", expand=True, padx=16, pady=14)
        tk.Label(wrap, text="⚙️ Settings", bg=COLORS["bg"], fg=COLORS["text"],
                 font=FONT_H2).pack(anchor="w", pady=(0, 10))

        thr_card = tk.Frame(wrap, bg=COLORS["card"])
        thr_card.pack(fill="x", pady=(0, 10))
        tk.Label(thr_card, text="Threshold (ค่า RMS ขั้นต่ำที่ถือว่า FIST)", bg=COLORS["card"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 4))
        thr_row = tk.Frame(thr_card, bg=COLORS["card"])
        thr_row.pack(fill="x", padx=12, pady=(0, 12))
        self.settings_thr_scale = tk.Scale(thr_row, from_=0.0, to=EMG_RANGE_V, resolution=0.001,
                                            orient="horizontal", bg=COLORS["card"],
                                            fg=COLORS["text"], troughcolor=COLORS["bg"],
                                            highlightthickness=0, font=("TH Sarabun New", 9),
                                            command=self._on_threshold)
        self.settings_thr_scale.set(self.threshold)
        self.settings_thr_scale.pack(fill="x")

        hw_card = tk.Frame(wrap, bg=COLORS["card"])
        hw_card.pack(fill="x", pady=(0, 10))
        tk.Label(hw_card, text="ฮาร์ดแวร์", bg=COLORS["card"], fg=COLORS["text"],
                 font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 4))
        hw_mode = "เชื่อมต่อ Hardware จริง (ADS1115 + PCA9685)" if self.hardware_mode \
            else "โหมดจำลอง (dev — ไม่พบ ADS1115)"
        tk.Label(hw_card, text=f"เซ็นเซอร์ EMG: {hw_mode}", bg=COLORS["card"],
                 fg=COLORS["text_dim"], font=("TH Sarabun New", 10)
                 ).pack(anchor="w", padx=12, pady=(0, 4))
        if not self.hardware_mode:
            self._btn(hw_card, "🖐 / ✊  สลับมือ (โหมดจำลอง)", COLORS["warning"],
                       self._toggle_grip).pack(anchor="w", padx=12, pady=(0, 12), ipady=4)
        else:
            tk.Label(hw_card, text="", bg=COLORS["card"]).pack(pady=(0, 8))

        about_card = tk.Frame(wrap, bg=COLORS["card"])
        about_card.pack(fill="x")
        tk.Label(about_card, text="เกี่ยวกับ", bg=COLORS["card"], fg=COLORS["text"],
                 font=FONT_TH_B).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Label(about_card, text="มือเทียม EMG Control System v1.2.0",
                 bg=COLORS["card"], fg=COLORS["text_dim"], font=("TH Sarabun New", 10)
                 ).pack(anchor="w", padx=12, pady=(0, 10))

    def _render_dash_confusion(self, conf):
        for w in self.dash_cm_frame.winfo_children():
            w.destroy()
        labels = list(CLASSES)
        if conf is None:
            tk.Label(self.dash_cm_frame, text="ยังไม่มีข้อมูลพอประเมิน", bg=COLORS["card"],
                      fg=COLORS["text_dim"], font=("TH Sarabun New", 10)).grid(row=0, column=0)
            self.dash_cm_summary_lbl.config(text="")
            return
        tk.Label(self.dash_cm_frame, text="", bg=COLORS["card"]).grid(row=0, column=0)
        for j, lbl in enumerate(labels):
            tk.Label(self.dash_cm_frame, text=f"ทำนาย: {lbl}", bg=COLORS["card"],
                      fg=COLORS["text_dim"], font=("TH Sarabun New", 9)
                      ).grid(row=0, column=j + 1, padx=4, pady=2)
        total = int(conf.sum()) or 1
        correct = int(conf.trace())
        for i, actual in enumerate(labels):
            tk.Label(self.dash_cm_frame, text=f"จริง: {actual}", bg=COLORS["card"],
                      fg=COLORS["text_dim"], font=("TH Sarabun New", 9)
                      ).grid(row=i + 1, column=0, padx=4, pady=2, sticky="e")
            for j in range(len(labels)):
                val = int(conf[i][j])
                bg = COLORS["success"] if i == j else COLORS["danger"]
                tk.Label(self.dash_cm_frame, text=str(val), bg=bg, fg="#ffffff",
                          font=("Courier New", 11, "bold"), width=6
                          ).grid(row=i + 1, column=j + 1, padx=2, pady=2, sticky="nsew")
        self.dash_cm_summary_lbl.config(
            text=f"Total: {total} samples   |   Accuracy: {correct/total*100:.2f}%")

    # ─── Machine Learning Tab ─────────────────────────────────────────────────
    def _build_ml_tab(self, parent):
        outer = tk.Frame(parent, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True, padx=14, pady=12)

        # ── แถบตั้งค่าด้านบน: จัดผังเดียวกับหน้า Calibration Tool (การ์ดเรียงแนวนอน
        # ในกรอบเน้นเดียวกัน) แทนพาเนลซ้ายแนวตั้งเดิม — ใช้ตัวช่วย _top_card เดียวกัน
        top_strip = tk.Frame(outer, bg=COLORS["surface"], highlightthickness=1,
                              highlightbackground=COLORS["accent2"])
        top_strip.pack(fill="x", pady=(0, 14))

        col_user = self._top_card(
            top_strip, "👤 เลือกผู้ใช้",
            help_text="แยกจาก 'ผู้ใช้งาน' ในหน้า Dashboard โดยตั้งใจ — "
                       "Dashboard ใช้เลือกว่ากำลัง 'บันทึก' สัญญาณเข้า Firebase "
                       "ให้ใคร ส่วนอันนี้ใช้เลือกว่าจะ 'ดึง' ข้อมูลของใครออกมา "
                       "เทรนโมเดล เลือกได้หลายคนพร้อมกัน (ติ๊กได้ไม่จำกัดจำนวน) "
                       "หรือรวมทุกคนเข้าด้วยกัน\n\n"
                       "'ทุก user (Label)' ต่างจาก 'ทุก user (รวมทุกคน)' ตรงที่จะ "
                       "ตรวจก่อนว่า user คนนั้นมี sample ที่ label (gesture) แล้วจริง "
                       "ในช่วงย้อนหลังที่เลือกหรือไม่ — คนที่ยังไม่มีข้อมูล label เลย "
                       "จะถูกข้ามไปโดยอัตโนมัติ ไม่นับรวมเข้า dataset")
        col_source = self._top_card(
            top_strip, "📊 แหล่งข้อมูลเทรน (Offline)",
            help_text="ไม่เทรนจากสัญญาณสด (Real-time) แล้ว เพราะค่าวิ่งเข้ามาตลอดเวลา "
                       "ทำให้เทรนถี่เกินไปและหนักเครื่องมาก\n\n"
                       "แทนที่ด้วยการดึงข้อมูลย้อนหลังจาก Firebase Realtime "
                       "Database โดยตรง (node sessions/<user_id>) แทนไฟล์ในเครื่อง "
                       "— ดึงได้แม้ข้อมูลนั้นมาจากเครื่องอื่น (เช่น sync มาจาก "
                       "Raspberry Pi หลายตัว) เลือกช่วงย้อนหลังนับจากวันนี้ "
                       "3 / 6 / 12 เดือน ตามที่ใช้วิเคราะห์ข้อมูลกันปกติ "
                       "(ต้องเชื่อมต่อ Firebase อยู่ถึงจะโหลดได้)\n\n"
                       "ใช้ค่า RMS เป็น feature เดียวในการจำแนก FIST/OPEN")
        col_model = self._top_card(
            top_strip, "🤖 Base Model",
            help_text="โมเดลเดี่ยวที่รองรับ (ของจริงทั้งหมด ไม่มีตัวประมาณ):\n"
                       "ANN, k-NN, SVM, Random Forest, C4.5 Decision Tree, Random Tree, "
                       "Naive Bayes — จาก scikit-learn ตรงๆ\n"
                       "XGBoost, CatBoost — จากไลบรารีต้นฉบับ (ต้องติดตั้งเพิ่มเอง: "
                       "pip install xgboost catboost --break-system-packages "
                       "ถ้าไม่ได้ติดตั้งจะไม่โผล่ในรายการนี้)")
        col_ensemble = self._top_card(
            top_strip, "🧩 Ensemble Method",
            help_text="รวม Base Model ด้านบนเข้ากับเทคนิค ensemble ของจริงจาก "
                       "scikit-learn ได้ทุกคู่ (Bagging, AdaBoost)\n\n"
                       "⚠️ ยิ่งซ้อน ensemble หนักเท่าไหร่ (โดยเฉพาะคู่กับ ANN/SVM/"
                       "XGBoost/CatBoost) ยิ่งเทรนช้าลง ระบบจะเทรนถี่น้อยลงอัตโนมัติ"
                       "ให้เหมาะสม แต่ยังไม่ทำให้โปรแกรมค้าง เพราะเทรนอยู่เบื้องหลัง"
                       "แยกจากหน้าจอ")
        col_train = self._top_card(top_strip, "🎛 ควบคุมการเทรน", last=True)

        # -- คอลัมน์ 1: เลือกผู้ใช้ -------------------------------------------------
        tk.Radiobutton(col_user, text="เลือก user รายคน (เลือกได้หลายคน)",
                       variable=self.ml_user_mode_var, value="specific",
                       command=self._on_ml_user_mode_change,
                       bg=COLORS["surface"], fg=COLORS["text"], selectcolor=COLORS["accent"],
                       activebackground=COLORS["surface"], font=FONT_TH, cursor="hand2",
                       wraplength=230, justify="left"
                       ).pack(anchor="w", pady=1)
        user_row = tk.Frame(col_user, bg=COLORS["surface"])
        user_row.pack(fill="x", pady=(0, 2))
        self.btn_ml_select_users = self._btn(user_row, "👥  เลือกผู้ใช้...", COLORS["card"],
                                              self._open_ml_user_select_dialog)
        self.btn_ml_select_users.pack(side="left", fill="x", expand=True, ipady=2)
        self.ml_user_summary_lbl = tk.Label(col_user, textvariable=self.ml_user_summary_var,
                                             bg=COLORS["surface"], fg=COLORS["text_dim"],
                                             font=("TH Sarabun New", 10), wraplength=250,
                                             justify="left", anchor="w")
        self.ml_user_summary_lbl.pack(anchor="w", pady=(2, 2), fill="x")
        tk.Radiobutton(col_user, text="ทุก user (รวมทุกคน)", variable=self.ml_user_mode_var,
                       value="all", command=self._on_ml_user_mode_change,
                       bg=COLORS["surface"], fg=COLORS["text"], selectcolor=COLORS["accent"],
                       activebackground=COLORS["surface"], font=FONT_TH, cursor="hand2",
                       wraplength=230, justify="left"
                       ).pack(anchor="w", pady=1)
        # "ทุก user (Label)" — เหมือน "all" แต่กรองเอาเฉพาะ user ที่ตรวจสอบแล้วว่ามี sample
        # ที่ label (gesture) จริงในช่วงย้อนหลังที่เลือก คนที่ยังไม่มีข้อมูล label เลย (เช่น
        # user ทดสอบที่สร้างไว้แต่ไม่เคย calibrate/บันทึกท่าทาง) จะถูกข้ามไปเอง ไม่ต้องมานั่ง
        # ไล่เลือกทีละคนเอง — ดู self._load_ml_history (ค้นหา "all_labeled")
        tk.Radiobutton(col_user, text="ทุก user (Label)", variable=self.ml_user_mode_var,
                       value="all_labeled", command=self._on_ml_user_mode_change,
                       bg=COLORS["surface"], fg=COLORS["text"], selectcolor=COLORS["accent"],
                       activebackground=COLORS["surface"], font=FONT_TH, cursor="hand2",
                       wraplength=230, justify="left"
                       ).pack(anchor="w", pady=(1, 6))
        tk.Checkbutton(col_user, text="ปรับสเกล RMS ต่อคนก่อนรวม (Normalize per user)",
                       variable=self.ml_normalize_var, bg=COLORS["surface"], fg=COLORS["text"],
                       selectcolor=COLORS["card"], activebackground=COLORS["surface"],
                       font=("TH Sarabun New", 10), cursor="hand2", wraplength=230, justify="left"
                       ).pack(anchor="w", pady=(0, 2))
        tk.Label(col_user, text="มีผลเฉพาะตอนรวมมากกว่า 1 user — แต่ละคนแรงกล้ามเนื้อ/ตำแหน่ง "
                            "electrode ไม่เท่ากัน เทียบ RMS ดิบข้ามคนตรงๆ ไม่ได้",
                 bg=COLORS["surface"], fg=COLORS["text_dim"], font=("TH Sarabun New", 9),
                 wraplength=250, justify="left").pack(anchor="w", pady=(0, 4))

        # -- คอลัมน์ 2: แหล่งข้อมูลเทรน ---------------------------------------------
        tk.Label(col_source, text="เลือกช่วงข้อมูลย้อนหลัง:", bg=COLORS["surface"],
                 fg=COLORS["text_dim"], font=FONT_TH).pack(anchor="w", pady=(2, 0))
        for label, months in (("3 เดือนย้อนหลัง", "3"),
                               ("6 เดือนย้อนหลัง", "6"),
                               ("12 เดือนย้อนหลัง", "12")):
            tk.Radiobutton(col_source, text=label, variable=self.ml_range_var, value=months,
                           bg=COLORS["surface"], fg=COLORS["text"],
                           selectcolor=COLORS["accent"], activebackground=COLORS["surface"],
                           font=FONT_TH, cursor="hand2", wraplength=230, justify="left"
                           ).pack(anchor="w", pady=1)

        self.btn_ml_load_range = self._btn(col_source, "📅  โหลดข้อมูลย้อนหลัง & เทรน",
                                            COLORS["accent2"], self._load_ml_history)
        self.btn_ml_load_range.pack(pady=(6, 4), ipady=4, fill="x")

        self.btn_ml_load = self._btn(col_source, "📂  หรืออัพโหลดไฟล์ JSON เอง (ขั้นสูง)",
                                      COLORS["card"], self._load_ml_file)
        self.btn_ml_load.pack(pady=(0, 4), ipady=4, fill="x")

        self.btn_ml_export_csv = self._btn(col_source, "⬇  Export Dataset เป็น CSV",
                                            COLORS["card"], self._export_ml_dataset_csv)
        self.btn_ml_export_csv.pack(pady=(0, 4), ipady=4, fill="x")

        # -- คอลัมน์ 3: Base Model --------------------------------------------------
        ml_base_cb = ttk.Combobox(col_model, textvariable=self.ml_base_var, width=16,
                                   values=BASE_MODELS, font=FONT_TH, state="readonly")
        ml_base_cb.pack(pady=4, fill="x")
        ml_base_cb.bind("<<ComboboxSelected>>", self._on_ml_model_change)
        tk.Label(col_model, text="⚠️ เปลี่ยนแล้วรีเซ็ตโมเดล ต้องโหลดข้อมูลเทรนใหม่",
                 bg=COLORS["surface"], fg=COLORS["text_dim"], font=("TH Sarabun New", 9),
                 wraplength=250, justify="left").pack(anchor="w", pady=(0, 2))

        # -- คอลัมน์ 4: Ensemble Method ----------------------------------------------
        ml_ens_cb = ttk.Combobox(col_ensemble, textvariable=self.ml_ensemble_var, width=16,
                                  values=ENSEMBLE_METHODS, font=FONT_TH, state="readonly")
        ml_ens_cb.pack(pady=4, fill="x")
        ml_ens_cb.bind("<<ComboboxSelected>>", self._on_ml_model_change)

        self.ml_model_note_lbl = tk.Label(col_ensemble, text="", bg=COLORS["surface"],
                                           fg=COLORS["text_dim"], font=("TH Sarabun New", 10),
                                           wraplength=250, justify="left")
        self.ml_model_note_lbl.pack(anchor="w", pady=(0, 4))

        # -- คอลัมน์ 5: ควบคุมการเทรน -------------------------------------------------
        self.btn_ml_reset = self._btn(col_train, "🗑  รีเซ็ตโมเดล",
                                       COLORS["warning"], self._reset_ml_model)
        self.btn_ml_reset.pack(pady=2, ipady=4, fill="x")

        # ปุ่มเทรนทุกชุดค่าผสม (Base Model ทุกตัว x Ensemble Method ทุกแบบ รวมถึง Base
        # เดี่ยวๆ ที่ไม่มี Ensemble คือ ens="None") กับ dataset ล่าสุดที่โหลดไว้ในครั้งเดียว
        # — ประหยัดเวลาเทียบกับต้องมาเปลี่ยน dropdown 2 ช่องแล้วกด 'โหลดข้อมูลย้อนหลัง & เทรน'
        # เองทีละคู่ (ดู _train_all_models) ผลลัพธ์ของทุกโมเดลจะถูกบันทึกลง self.model_results
        # เหมือนเทรนปกติทุกครั้ง จึงไปโผล่ให้เทียบกันได้ทันทีที่หน้า 'สรุปผลโมเดล'
        self.btn_ml_train_all = self._btn(col_train, "🚀  เทรนทุกโมเดล (Base + Ensemble)",
                                           COLORS["accent2"], self._train_all_models)
        self.btn_ml_train_all.pack(pady=(6, 2), ipady=4, fill="x")
        tk.Label(col_train, text="เทรนทุก Base Model x Ensemble Method ทีเดียว "
                                  "(รวม Base เดี่ยวๆ ด้วย) — ใช้ dataset ล่าสุดที่โหลดไว้",
                 bg=COLORS["surface"], fg=COLORS["text_dim"], font=("TH Sarabun New", 9),
                 wraplength=250, justify="left").pack(anchor="w", pady=(0, 2))

        # ── ผลการเทรน/กราฟ/log — เต็มความกว้างหน้าจอด้านล่าง (เหมือนหน้า Calibration
        # Tool ที่กราฟ+ปุ่ม+log อยู่เต็มความกว้างใต้แถบตั้งค่า)
        #
        # ⚠️ ของเดิมห่อทุกอย่าง (หัวข้อ/สถานะ/สรุป split/metrics/confusion + กราฟ 3 อัน + log)
        # ไว้ใน _make_sidebar_scrollable กล่องเดียวกันหมด — ตัว Canvas ของ scrollable
        # คำนวณ scrollregion จาก inner.bbox("all") ตอน <Configure> ซึ่งเกิดพร้อมๆ กับตอนที่
        # matplotlib FigureCanvasTkAgg (กราฟ 3 อัน) กำลังสร้าง/วาดตัวเองอยู่ (มีน้ำหนักและ
        # เหตุการณ์ resize ภายในเยอะกว่า widget ปกติมาก) ทำให้บางเครื่อง/บาง timing คำนวณ
        # bbox ผิดพลาดตั้งแต่รอบแรก จนกล่องสถานะ/metrics/confusion ที่อยู่บนสุดในนั้นไม่โผล่
        # ให้เห็นเลย (เหมือนหายไปทั้งที่โค้ดสร้างมันขึ้นมาแล้วจริงๆ)
        #
        # แก้โดยแยกเป็น 2 ส่วน: ส่วนสรุปผล (หัวข้อ/สถานะ/split/metrics/confusion) แสดงตรงๆ
        # ไม่ผ่าน scrollable เลย การันตีว่าเห็นเสมอทันทีที่เปิดแท็บ ไม่ขึ้นกับการคำนวณ
        # scrollregion ใดๆ ส่วนกราฟ 3 อัน + log ที่หนักและกินพื้นที่แนวตั้งเยอะเท่านั้นที่ยัง
        # อยู่ใน scrollable กล่องแยกต่างหากด้านล่าง
        right_outer = tk.Frame(outer, bg=COLORS["bg"])
        right_outer.pack(fill="both", expand=True)

        top_r = tk.Frame(right_outer, bg=COLORS["bg"])
        top_r.pack(fill="x", pady=(0, 4))
        tk.Label(top_r, text="ผลการเทรนโมเดล", bg=COLORS["bg"],
                 fg=COLORS["text"], font=FONT_H2).pack(side="left")

        self.ml_status_lbl = tk.Label(right_outer, text="Samples: 0   |   เทรนไปแล้ว: 0 ครั้ง",
                                       bg=COLORS["bg"], fg=COLORS["text_dim"], font=FONT_TH)
        self.ml_status_lbl.pack(anchor="w", pady=(0, 2))

        # สรุปสัดส่วน train/validate/test จริง ณ ตอนนี้ เทียบกับเป้าหมาย 70/10/20% ที่ตั้งไว้
        # ใน OnlineTrainer (ดู _render_ml_split_summary) — ตอบข้อ 1: "เพิ่ม validate...
        # สรุป train70 test20 validate10"
        self.ml_split_lbl = tk.Label(right_outer, text="Train/Validate/Test: —",
                                      bg=COLORS["card"], fg=COLORS["text_dim"], font=FONT_MONO,
                                      justify="left", anchor="w", wraplength=1000)
        self.ml_split_lbl.pack(fill="x", pady=(0, 8), ipady=6, ipadx=8)

        tk.Label(right_outer, text="📈 ผลการประเมิน (Test Set 20% ที่สุ่มแยกไว้)", bg=COLORS["bg"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", pady=(4, 2))
        self.ml_metrics_lbl = tk.Label(right_outer, text="ยังไม่มีข้อมูลพอประเมิน",
                                        bg=COLORS["card"], fg=COLORS["text"],
                                        font=FONT_MONO, justify="left", anchor="w")
        self.ml_metrics_lbl.pack(fill="x", pady=(0, 8), ipady=8, ipadx=8)

        tk.Label(right_outer, text="📐 Confusion Matrix", bg=COLORS["bg"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", pady=(0, 2))
        self.ml_confusion_lbl = tk.Label(right_outer, text="ยังไม่มีข้อมูลพอประเมิน",
                                          bg=COLORS["card"], fg=COLORS["text"],
                                          font=FONT_MONO, justify="left", anchor="w")
        self.ml_confusion_lbl.pack(fill="x", pady=(0, 8), ipady=8, ipadx=8)

        # ── Confusion Matrix แบบรูปภาพ (heatmap) — เพิ่มคู่กับตัวอักษรด้านบน เพราะตัวอักษร
        # แทรกลงรายงาน (Word/บทที่ 4) ไม่สวย ส่วนอันนี้เป็น matplotlib Figure จริง บันทึก
        # เป็น .png ความละเอียดสูงได้ตรง ๆ ด้วย _save_figure_png()
        cm_img_head = tk.Frame(right_outer, bg=COLORS["bg"])
        cm_img_head.pack(fill="x", pady=(2, 2))
        tk.Label(cm_img_head, text="🖼 Confusion Matrix (รูปภาพ — สำหรับแทรกในรายงาน)",
                 bg=COLORS["bg"], fg=COLORS["text"], font=FONT_TH_B).pack(side="left")
        self._btn(cm_img_head, "💾 บันทึกรูป", COLORS["surface"],
                   lambda: self._save_figure_png(self._ml_fig_cm, "confusion_matrix.png", self.ml_log)
                   ).pack(side="right")
        self._ml_fig_cm = Figure(figsize=(4.2, 3.6), dpi=90)
        self._ml_fig_cm.patch.set_facecolor(COLORS["card"])
        self._ml_ax_cm = self._ml_fig_cm.add_subplot(111)
        self._ml_canvas_cm = FigureCanvasTkAgg(self._ml_fig_cm, master=right_outer)
        self._ml_canvas_cm.get_tk_widget().pack(fill="x", pady=(0, 8))

        # ── กราฟ 3 อัน + log — หนักและกินพื้นที่แนวตั้งเยอะสุด แยก scrollable ของตัวเอง
        # ต่างหากจากส่วนสรุปผลด้านบน (เหตุผลดู comment ก้อนใหญ่ด้านบน)
        graphs_outer = tk.Frame(right_outer, bg=COLORS["bg"])
        graphs_outer.pack(fill="both", expand=True)
        graphs_scroll, right = self._make_sidebar_scrollable(graphs_outer, bg=COLORS["bg"])
        graphs_scroll.pack(fill="both", expand=True)


        # ─── กราฟระหว่างเทรน — แยกเป็น "Loss" (Error) กับ "Performance" (Accuracy) คนละ
        # กราฟ (เดิมรวมกันเป็นกราฟเดียวแบบ dual-axis แต่พอ validate set ยังไม่มีข้อมูลพอ
        # (เช่น เพิ่งเทรนรอบแรก) กราฟทั้งก้อนจะไม่โชว์อะไรเลยทั้งที่บางเส้นมีข้อมูลพอแล้ว —
        # แยกเป็นคนละกราฟให้แต่ละอันโชว์ได้อิสระจากกัน) แถวที่ 2 เป็นกราฟเปรียบเทียบ predict
        # อัพเดตพร้อมกับ metrics/confusion ทุกครั้งที่มีผลเทรนรอบใหม่ (ดู _refresh_ml_metrics
        # -> _render_ml_loss_graph / _render_ml_performance_graph / _render_ml_predict_graph)
        graphs_row1 = tk.Frame(right, bg=COLORS["bg"])
        graphs_row1.pack(fill="x", padx=2, pady=(0, 4))

        loss_box = tk.Frame(graphs_row1, bg=COLORS["card"])
        loss_box.pack(side="left", fill="both", expand=True, padx=(0, 4))
        loss_head = tk.Frame(loss_box, bg=COLORS["card"])
        loss_head.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(loss_head, text="📉 Loss (Error) ระหว่างเทรน — Train vs Validate",
                 bg=COLORS["card"], fg=COLORS["text"], font=("TH Sarabun New", 10, "bold")
                 ).pack(side="left")
        self._btn(loss_head, "💾", COLORS["surface"],
                   lambda: self._save_figure_png(self._ml_fig_loss, "loss_graph.png", self.ml_log)
                   ).pack(side="right")
        self._ml_fig_loss = Figure(figsize=(4.6, 2.3), dpi=90)
        self._ml_fig_loss.patch.set_facecolor(COLORS["card"])
        self._ml_ax_loss = self._ml_fig_loss.add_subplot(111)
        self._ml_canvas_loss = FigureCanvasTkAgg(self._ml_fig_loss, master=loss_box)
        self._ml_canvas_loss.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._style_ml_axes(self._ml_ax_loss, "Retrain round", "Error (1 - Accuracy)")

        perf_box = tk.Frame(graphs_row1, bg=COLORS["card"])
        perf_box.pack(side="left", fill="both", expand=True, padx=(4, 0))
        perf_head = tk.Frame(perf_box, bg=COLORS["card"])
        perf_head.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(perf_head, text="📈 Performance (Accuracy) ระหว่างเทรน — Train vs Validate",
                 bg=COLORS["card"], fg=COLORS["text"], font=("TH Sarabun New", 10, "bold")
                 ).pack(side="left")
        self._btn(perf_head, "💾", COLORS["surface"],
                   lambda: self._save_figure_png(self._ml_fig_perf, "performance_graph.png", self.ml_log)
                   ).pack(side="right")
        self._ml_fig_perf = Figure(figsize=(4.6, 2.3), dpi=90)
        self._ml_fig_perf.patch.set_facecolor(COLORS["card"])
        self._ml_ax_perf = self._ml_fig_perf.add_subplot(111)
        self._ml_canvas_perf = FigureCanvasTkAgg(self._ml_fig_perf, master=perf_box)
        self._ml_canvas_perf.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._style_ml_axes(self._ml_ax_perf, "Retrain round", "Accuracy")

        graphs_row2 = tk.Frame(right, bg=COLORS["bg"])
        graphs_row2.pack(fill="x", padx=2, pady=(0, 8))

        pred_box = tk.Frame(graphs_row2, bg=COLORS["card"])
        pred_box.pack(side="left", fill="both", expand=True)
        pred_head = tk.Frame(pred_box, bg=COLORS["card"])
        pred_head.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(pred_head, text="🎯 เปรียบเทียบ Predict — Actual vs Predicted (Test Set)",
                 bg=COLORS["card"], fg=COLORS["text"], font=("TH Sarabun New", 10, "bold")
                 ).pack(side="left")
        self._btn(pred_head, "💾", COLORS["surface"],
                   lambda: self._save_figure_png(self._ml_fig_pred, "predict_vs_actual_graph.png", self.ml_log)
                   ).pack(side="right")
        self._ml_fig_pred = Figure(figsize=(9.6, 2.3), dpi=90)
        self._ml_fig_pred.patch.set_facecolor(COLORS["card"])
        self._ml_ax_pred = self._ml_fig_pred.add_subplot(111)
        self._ml_canvas_pred = FigureCanvasTkAgg(self._ml_fig_pred, master=pred_box)
        self._ml_canvas_pred.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._style_ml_axes(self._ml_ax_pred, "Class", "Samples")

        # ─── กราฟ Decision Boundary 2 มิติ (RMS vs MAV) — ใช้โมเดลตัวช่วยที่เทรนใหม่บน 2
        # features นี้โดยเฉพาะ ด้วยสถาปัตยกรรม/hyperparameter เดียวกับโมเดลที่เลือกใช้งานจริง
        # (ดู OnlineTrainer.get_boundary_data) เพื่อวาดขอบเขตการตัดสินใจ (contourf) พร้อม
        # จุดข้อมูลจริงทับ แบบเดียวกับตัวอย่าง k-NN decision boundary ของ scikit-learn ที่ใช้
        # อ้างอิงในบทที่ 2 — เป็นภาพประกอบบนระนาบ 2 มิติเท่านั้น ไม่ใช่ขอบเขตจริงของโมเดล 7
        # มิติที่ใช้งานอยู่กับ servo จริง (ดู docstring ของเมธอดนั้นสำหรับรายละเอียด)
        graphs_row3 = tk.Frame(right, bg=COLORS["bg"])
        graphs_row3.pack(fill="x", padx=2, pady=(0, 8))

        boundary_box = tk.Frame(graphs_row3, bg=COLORS["card"])
        boundary_box.pack(side="left", fill="both", expand=True)
        boundary_head = tk.Frame(boundary_box, bg=COLORS["card"])
        boundary_head.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(boundary_head, text="🗺️ Decision Boundary — RMS vs MAV",
                 bg=COLORS["card"], fg=COLORS["text"], font=("TH Sarabun New", 10, "bold")
                 ).pack(side="left")
        self._btn(boundary_head, "💾", COLORS["surface"],
                   lambda: self._save_figure_png(self._ml_fig_boundary, "decision_boundary_graph.png", self.ml_log)
                   ).pack(side="right")
        self._ml_fig_boundary = Figure(figsize=(9.6, 3.2), dpi=90)
        self._ml_fig_boundary.patch.set_facecolor(COLORS["card"])
        self._ml_ax_boundary = self._ml_fig_boundary.add_subplot(111)
        self._ml_canvas_boundary = FigureCanvasTkAgg(self._ml_fig_boundary, master=boundary_box)
        self._ml_canvas_boundary.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._style_ml_axes(self._ml_ax_boundary, "RMS (V)", "MAV (V)")

        tk.Label(right, text="📋 Log การเทรน / ทำนายแบบ Real-time", bg=COLORS["bg"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=2)
        ml_log_frame = tk.Frame(right, bg=COLORS["card"])
        ml_log_frame.pack(fill="both", expand=True, padx=2, pady=(2, 8))
        self.ml_log = tk.Text(ml_log_frame, bg=COLORS["card"], fg=COLORS["text"],
                               font=FONT_MONO, height=6, state="disabled",
                               relief="flat", wrap="word")
        sb3 = tk.Scrollbar(ml_log_frame, command=self.ml_log.yview)
        self.ml_log.configure(yscrollcommand=sb3.set)
        self.ml_log.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sb3.pack(side="right", fill="y")

        self._update_ml_model_note()  # ตั้งค่า note label ครั้งแรกให้ตรงกับดีฟอลต์
        self._update_ml_user_summary()  # ตั้งข้อความสรุปเริ่มต้น ("ยังไม่ได้เลือกผู้ใช้")

    def _style_ml_axes(self, ax, xlabel, ylabel):
        """ตั้งสี/ฟอนต์ของแกนกราฟ matplotlib ให้เข้ากับธีมมืดของแอป (COLORS) — เรียกครั้งแรก
        ตอนสร้างแกน และเรียกซ้ำทุกครั้งที่ ax.clear() (clear() ล้างสไตล์ที่ตั้งไว้ทิ้งด้วย)"""
        ax.set_facecolor(COLORS["card"])
        ax.set_xlabel(xlabel, color=COLORS["text_dim"], fontsize=8)
        ax.set_ylabel(ylabel, color=COLORS["text_dim"], fontsize=8)
        ax.tick_params(colors=COLORS["text_dim"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(COLORS["text_dim"])
        ax.grid(True, color=COLORS["surface"], linewidth=0.6, alpha=0.6)

    def _render_ml_split_summary(self, snap):
        """แสดงจำนวน/สัดส่วนจริงของ train/validate/test ณ ตอนนี้ เทียบกับเป้าหมาย
        70/10/20% ที่ตั้งไว้ใน OnlineTrainer — ข้อ 1: เพิ่ม validate เข้าสมการคำนวณ"""
        s = snap["split_summary"]
        total = s["n_train"] + s["n_val"] + s["n_test"]
        if total == 0:
            self.ml_split_lbl.config(text="Train/Validate/Test: ยังไม่มีข้อมูล")
            return
        self.ml_split_lbl.config(
            text=f"Train      {s['n_train']:>6,}  ({s['pct_train']:5.1f}%  เป้าหมาย {s['target_train']:.0f}%)\n"
                 f"Validate   {s['n_val']:>6,}  ({s['pct_val']:5.1f}%  เป้าหมาย {s['target_val']:.0f}%)\n"
                 f"Test       {s['n_test']:>6,}  ({s['pct_test']:5.1f}%  เป้าหมาย {s['target_test']:.0f}%)")

    def _render_ml_loss_graph(self, snap):
        """วาดกราฟ Loss (Error = 1-Accuracy) ของ train vs validate ต่อรอบเทรน
        (snap['history']) — แยกออกจากกราฟ Performance ต่างหาก (ข้อ 3: บางครั้ง validate
        set ยังไม่มีข้อมูลพอ ถ้ารวมกราฟเดียวกันทั้งก้อนจะไม่โชว์อะไรเลย ทั้งที่ train error
        มีข้อมูลพอวาดแล้ว) — โชว์ได้ตั้งแต่มีข้อมูล 1 รอบเทรน (จุดเดียว ไม่มีเส้นเชื่อม)
        ไม่ต้องรอครบ 2 รอบเหมือนเดิม"""
        ax = self._ml_ax_loss
        ax.clear()
        self._style_ml_axes(ax, "Retrain round", "Error (1 - Accuracy)")

        history = snap["history"]
        if len(history) < 1:
            ax.text(0.5, 0.5, "Not enough data yet\n(need at least 1 retrain round)",
                    ha="center", va="center", color=COLORS["text_dim"], fontsize=8,
                    transform=ax.transAxes)
            self._ml_canvas_loss.draw_idle()
            return

        rounds = [h["round"] for h in history]
        train_err = [h["train_error"] for h in history]
        val_err = [h["val_error"] for h in history]

        lines = []
        l1, = ax.plot(rounds, train_err, color=COLORS["danger"], linewidth=1.4,
                       marker="o", markersize=3, label="Train Error")
        lines.append(l1)
        # val_error เป็น None ได้ตอนยังไม่มี validate sample พอ (เช่นเพิ่งเริ่มเทรนรอบแรกๆ)
        # — กรองจุด None ออกก่อนวาด ไม่งั้น matplotlib error
        val_rounds = [r for r, v in zip(rounds, val_err) if v is not None]
        val_err_clean = [v for v in val_err if v is not None]
        if val_err_clean:
            l2, = ax.plot(val_rounds, val_err_clean, color=COLORS["warning"], linewidth=1.4,
                           marker="o", markersize=3, linestyle="--", label="Validate Error")
            lines.append(l2)

        ax.set_ylim(-0.02, 1.02)
        if len(rounds) == 1:
            ax.set_xlim(rounds[0] - 1, rounds[0] + 1)
        ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=6,
                  facecolor=COLORS["card"], edgecolor=COLORS["text_dim"],
                  labelcolor=COLORS["text"])
        self._ml_fig_loss.tight_layout()
        self._ml_canvas_loss.draw_idle()

    def _render_ml_performance_graph(self, snap):
        """วาดกราฟ Performance (Accuracy) ของ train vs validate ต่อรอบเทรน — คู่กับ
        _render_ml_loss_graph แต่แยกเป็นกราฟของตัวเอง (ข้อ 3)"""
        ax = self._ml_ax_perf
        ax.clear()
        self._style_ml_axes(ax, "Retrain round", "Accuracy")

        history = snap["history"]
        if len(history) < 1:
            ax.text(0.5, 0.5, "Not enough data yet\n(need at least 1 retrain round)",
                    ha="center", va="center", color=COLORS["text_dim"], fontsize=8,
                    transform=ax.transAxes)
            self._ml_canvas_perf.draw_idle()
            return

        rounds = [h["round"] for h in history]
        train_acc = [h["train_accuracy"] for h in history]
        val_acc = [h["val_accuracy"] for h in history]

        lines = []
        l1, = ax.plot(rounds, train_acc, color=COLORS["accent"], linewidth=1.4,
                       marker="o", markersize=3, label="Train Accuracy")
        lines.append(l1)
        val_rounds = [r for r, v in zip(rounds, val_acc) if v is not None]
        val_acc_clean = [v for v in val_acc if v is not None]
        if val_acc_clean:
            l2, = ax.plot(val_rounds, val_acc_clean, color=COLORS["success"], linewidth=1.4,
                           marker="o", markersize=3, linestyle="--", label="Validate Accuracy")
            lines.append(l2)

        ax.set_ylim(-0.02, 1.02)
        if len(rounds) == 1:
            ax.set_xlim(rounds[0] - 1, rounds[0] + 1)
        ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right", fontsize=6,
                  facecolor=COLORS["card"], edgecolor=COLORS["text_dim"],
                  labelcolor=COLORS["text"])
        self._ml_fig_perf.tight_layout()
        self._ml_canvas_perf.draw_idle()

    def _render_ml_predict_graph(self, snap):
        """วาดกราฟแท่งเปรียบเทียบจำนวน Actual vs Predicted ต่อ class บน test set 20%
        ล่าสุด (snap['last_test_true']/['last_test_pred']) — ตอบข้อ 4"""
        ax = self._ml_ax_pred
        ax.clear()
        self._style_ml_axes(ax, "Class", "Samples")

        y_true = snap["last_test_true"]
        y_pred = snap["last_test_pred"]
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            ax.text(0.5, 0.5, "Not enough data yet\n(need a Test Set evaluation first)",
                    ha="center", va="center", color=COLORS["text_dim"], fontsize=8,
                    transform=ax.transAxes)
            self._ml_canvas_pred.draw_idle()
            return

        labels = list(CLASSES)
        actual_counts = [y_true.count(c) for c in labels]
        pred_counts = [y_pred.count(c) for c in labels]

        x = range(len(labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], actual_counts, width, color=COLORS["accent"],
               label="Actual")
        ax.bar([i + width / 2 for i in x], pred_counts, width, color=COLORS["accent2"],
               label="Predicted")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, color=COLORS["text_dim"], fontsize=8)
        ax.legend(loc="upper right", fontsize=6, facecolor=COLORS["card"],
                  edgecolor=COLORS["text_dim"], labelcolor=COLORS["text"])
        self._ml_fig_pred.tight_layout()
        self._ml_canvas_pred.draw_idle()

    def _render_ml_boundary_graph(self, snap):
        """วาดกราฟ Decision Boundary 2 มิติ (RMS vs MAV) ของโมเดลที่เลือกใช้งานอยู่ตอนนี้ —
        เรียก self.trainer.get_boundary_data() ซึ่งเทรน "โมเดลตัวช่วย" แยกต่างหาก (ดู
        docstring ของเมธอดนั้นใน ml_online.py) บน 2 features นี้โดยเฉพาะ แล้ววาดพื้นที่การ
        ตัดสินใจ (contourf) ทับด้วยจุดข้อมูล train จริง — สไตล์เดียวกับตัวอย่าง k-NN decision
        boundary ของ scikit-learn ที่ใช้อ้างอิงในบทที่ 2 (ข้อ 2.2.16) เป็นภาพประกอบขอบเขตการ
        ตัดสินใจแบบ 2 มิติเท่านั้น ไม่ใช่ขอบเขตจริงของโมเดล 7 มิติที่ใช้งานจริงกับ servo"""
        ax = self._ml_ax_boundary
        ax.clear()
        self._style_ml_axes(ax, "RMS (V)", "MAV (V)")

        data = self.trainer.get_boundary_data("rms", "mav")
        if data is None:
            ax.text(0.5, 0.5, "Not enough data yet\n(need samples from both classes)",
                    ha="center", va="center", color=COLORS["text_dim"], fontsize=8,
                    transform=ax.transAxes)
            self._ml_canvas_boundary.draw_idle()
            return

        pipe, X2, y = data["pipeline"], data["X2"], data["y"]
        x_min, x_max = float(X2[:, 0].min()), float(X2[:, 0].max())
        y_min, y_max = float(X2[:, 1].min()), float(X2[:, 1].max())
        x_pad = (x_max - x_min) * 0.1 or 0.01
        y_pad = (y_max - y_min) * 0.1 or 0.01
        xx, yy = np.meshgrid(
            np.linspace(x_min - x_pad, x_max + x_pad, 200),
            np.linspace(y_min - y_pad, y_max + y_pad, 200))
        try:
            Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()])
        except Exception:
            ax.text(0.5, 0.5, "Boundary model error", ha="center", va="center",
                    color=COLORS["text_dim"], fontsize=8, transform=ax.transAxes)
            self._ml_canvas_boundary.draw_idle()
            return
        Z = (Z == "FIST").astype(float).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                    colors=[COLORS["surface"], COLORS["accent"]], alpha=0.35)
        for label, color in (("OPEN", COLORS["success"]), ("FIST", COLORS["danger"])):
            mask = (y == label)
            if mask.any():
                ax.scatter(X2[mask, 0], X2[mask, 1], s=10, color=color,
                           edgecolors="none", label=label, alpha=0.85)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.legend(loc="upper right", fontsize=6, facecolor=COLORS["card"],
                  edgecolor=COLORS["text_dim"], labelcolor=COLORS["text"])
        ax.set_title(f"โมเดล: {data['display_name']}  (n={len(y)} — เทรนใหม่บน 2 features นี้)",
                     color=COLORS["text_dim"], fontsize=7)
        self._ml_fig_boundary.tight_layout()
        self._ml_canvas_boundary.draw_idle()

    def _update_ml_user_summary(self):
        """อัพเดตข้อความสรุปผู้ใช้ที่เลือกไว้ (self.ml_selected_users) ให้โชว์บนหน้าจอ
        เรียกทุกครั้งที่รายชื่อที่เลือกเปลี่ยน (ปิด dialog เลือกผู้ใช้ / เปลี่ยนโหมด)"""
        n = len(self.ml_selected_users)
        if n == 0:
            self.ml_user_summary_var.set("ยังไม่ได้เลือกผู้ใช้ — กดปุ่ม 'เลือกผู้ใช้...' ด้านบน")
        elif n <= 4:
            self.ml_user_summary_var.set(f"เลือกแล้ว {n} คน: " + ", ".join(self.ml_selected_users))
        else:
            shown = ", ".join(self.ml_selected_users[:4])
            self.ml_user_summary_var.set(f"เลือกแล้ว {n} คน: {shown}, ... (+{n - 4})")

    def _make_sidebar_scrollable(self, parent, bg=None):
        """คล้าย _make_touch_scrollable_list แต่บางกว่ามาก (ไม่มีปุ่ม ▲/▼ ใหญ่ + scrollbar
        กว้างแค่ ~10px แทน 48px) ออกแบบมาสำหรับพาเนลควบคุมด้านซ้ายที่แคบอยู่แล้ว (280px)
        ไม่อยากให้ scrollbar กินพื้นที่แนวนอนไปเยอะเหมือน dialog เลือกผู้ใช้ (ซึ่งกว้างกว่า
        มาก) — ยังรองรับ mouse wheel (dev บน PC) และลาก scrollbar บางๆ ได้บนจอสัมผัส

        เหตุผลที่ต้องมี: พอเพิ่ม wraplength ให้หัวข้อยาวๆ ขึ้นบรรทัดใหม่แทนที่จะโดนตัดขอบ
        (ดู _card_label) ความสูงรวมของเนื้อหาในพาเนลซ้ายเพิ่มขึ้น พอเจอเครื่อง/ฟอนต์/ขนาด
        หน้าจอที่ต่างจากตอน dev (เช่น ความสูงหน้าต่างใกล้ minsize 700px หรือฟอนต์ TH Sarabun
        New ตัวจริงมี line-height สูงกว่าฟอนต์ fallback ตอน dev) เนื้อหาส่วนล่าง (เช่น
        Ensemble Method, ปุ่มรีเซ็ตโมเดล) อาจโดนดันตกขอบล่างของพาเนล (fixed height ตาม
        พื้นที่หน้าต่าง) จนกดไม่ถึงเลย — ห่อด้วย Canvas+Scrollbar ให้เลื่อนดูได้เสมอ ไม่ว่า
        เนื้อหาจะสูงแค่ไหนก็ตาม กันปัญหานี้ทั้งหมดที่ต้นตอ แทนที่จะพยายามเดาความสูงให้พอดี
        เป๊ะๆ ซึ่งเปราะบางมาก (ขึ้นกับฟอนต์/ขนาดจอที่ควบคุมไม่ได้)

        คืนค่า (outer, inner) — ใส่ widget ต่างๆ ลงใน inner ตามปกติ (แทนที่จะใส่ลง parent
        ตรงๆ) แล้ว pack(outer) ลง parent อีกที"""
        bg = bg or COLORS["surface"]
        outer = tk.Frame(parent, bg=bg)

        # ใช้ grid แทน pack(side=left/right) สำหรับคู่ canvas+scrollbar นี้ — เจอบั๊กเดียวกับ
        # ตอนแก้ _make_touch_scrollable_list: pack(side="left", fill="both", expand=True)
        # ให้ canvas ไปก่อน ทำให้มันชิงพื้นที่ทั้งหมดของ outer ไปหมด พอ scrollbar
        # pack(side="right") ตามมาทีหลัง คอลัมน์ของมันโดนบีบเหลือ 0px (ไม่โผล่เลย แม้แต่งเป็น
        # สีที่ต่างจากพื้นหลังแล้วก็ตาม) grid + columnconfigure กำหนดสัดส่วนคอลัมน์ชัดเจน
        # ตายตัวกว่า ไม่ขึ้นกับลำดับการวาง
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(1, weight=0, minsize=12)

        canvas = tk.Canvas(outer, bg=bg, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        # เดิม trough_color=COLORS["surface"] ซึ่งเป็นสีเดียวกับพื้นหลังพาเนลเป๊ะ (#2a2a3e)
        # ทำให้รางเลื่อนกลืนหายไปกับพื้นหลังจนดูเหมือนไม่มี scrollbar เลย เปลี่ยนเป็น
        # COLORS["card"] (อ่อนกว่าชัดเจน) ให้ตัดกับพื้นหลัง + ขยับกว้างจาก 10 -> 12px
        scrollbar = CustomScrollbar(outer, command=canvas.yview, width=12,
                                     thumb_color=COLORS["accent"], trough_color=COLORS["card"])
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        inner = tk.Frame(canvas, bg=bg)
        inner_win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig(inner_win, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        return outer, inner

    def _make_touch_scrollable_list(self, parent, list_bg=None):
        """สร้างพื้นที่รายการที่เลื่อนได้ (Canvas+Scrollbar) แบบใช้งานได้จริงบน Raspberry Pi
        ที่ต่อจอ touchscreen (ไม่มี mouse wheel เลย) — ปัญหาที่เจอจริง: scrollbar เดิม
        (tk.Scrollbar ค่าเริ่มต้น) บางมาก (~14px) กดโดนยากด้วยนิ้ว พอมี user เยอะขึ้น (เช่น
        17+ คน) รายการที่อยู่นอกจอเลยกดเลือกไม่ได้เลยสักคน

        แก้โดยเพิ่ม 2 อย่าง (ไม่ใช้วิธีลากนิ้วบนตัว list โดยตรง เพราะจะชนกับการแตะติ๊ก
        checkbox ที่อยู่ในนั้น แยกยากว่าผู้ใช้ตั้งใจจะลากเลื่อนหรือจะแตะติ๊ก):
          1. Scrollbar กว้างขึ้นชัดเจน + สีตัดกับพื้นหลัง — วาดเองด้วย CustomScrollbar
             (Canvas) แทน tk.Scrollbar เดิม เพราะค่า width ของ tk.Scrollbar ไม่ถูกเคารพ
             เท่ากันทุก theme/เครื่อง (บางเครื่องยังได้แถบบางๆ ~14px ทั้งที่ตั้ง width=28 แล้ว)
          2. ปุ่มลูกศร ▲/▼ ขนาดใหญ่ ด้านบน/ล่างของ scrollbar กดทีเดียวเลื่อนเป็นก้อนได้เลย
             ไม่ต้องเล็งไปโดนแถบเลื่อนบางๆ พอดี
        ยัง bind mouse wheel ไว้เหมือนเดิมด้วยสำหรับตอน dev บน PC

        คืนค่า (canvas, inner_frame, cleanup_fn) — ผู้เรียกใส่ widget ต่างๆ ลงใน inner_frame
        ตามปกติ แล้วเรียก cleanup_fn() ตอนปิด dialog (unbind mouse wheel กันหลุดค้าง
        กระทบหน้าต่างอื่นหลังปิด dialog นี้ไปแล้ว)"""
        outer = tk.Frame(parent, bg=COLORS["surface"])
        bg = list_bg or COLORS["card"]

        # ใช้ grid แทน pack สำหรับ 2 คอลัมน์นี้ (canvas เนื้อหา + คอลัมน์ scrollbar) —
        # เดิมใช้ pack(side=left/right) แล้วเจอปัญหาบางเครื่อง/บาง layout คอลัมน์
        # scrollbar โดนบีบจนแทบไม่เหลือพื้นที่แสดงผล (ดูเหมือนไม่มี scrollbar เลย)
        # grid + columnconfigure กำหนดสัดส่วนชัดเจนแน่นอนกว่า ไม่ขึ้นกับลำดับการ pack
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(1, weight=0, minsize=48)

        canvas = tk.Canvas(outer, bg=bg, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        scroll_col = tk.Frame(outer, bg=COLORS["surface"], width=48)
        scroll_col.grid(row=0, column=1, sticky="ns")
        scroll_col.grid_propagate(False)
        scroll_col.grid_columnconfigure(0, weight=1)
        scroll_col.grid_rowconfigure(1, weight=1)

        tk.Button(scroll_col, text="▲", font=("TH Sarabun New", 13, "bold"),
                  bg=COLORS["card"], fg=COLORS["accent"], activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  command=lambda: canvas.yview_scroll(-3, "units")
                  ).grid(row=0, column=0, sticky="ew", ipady=8)

        scrollbar = CustomScrollbar(scroll_col, command=canvas.yview, width=42,
                                     thumb_color=COLORS["accent"], trough_color=COLORS["border"])
        scrollbar.grid(row=1, column=0, sticky="nsew")
        canvas.configure(yscrollcommand=scrollbar.set)

        tk.Button(scroll_col, text="▼", font=("TH Sarabun New", 13, "bold"),
                  bg=COLORS["card"], fg=COLORS["accent"], activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  command=lambda: canvas.yview_scroll(3, "units")
                  ).grid(row=2, column=0, sticky="ew", ipady=8)

        inner = tk.Frame(canvas, bg=bg)
        inner_win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig(inner_win, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        def cleanup():
            canvas.unbind_all("<MouseWheel>")

        return outer, canvas, inner, cleanup

    def _open_ml_user_select_dialog(self):
        """Dialog ติ๊กเลือกผู้ใช้ได้หลายคนพร้อมกัน สำหรับดึงข้อมูลย้อนหลังมาเทรนโมเดล —
        ใช้รูปแบบเดียวกับ _open_delete_users_dialog (checkbox ในกล่องเลื่อนได้) แต่กด
        'ยืนยันการเลือก' แล้วเก็บผลลงใน self.ml_selected_users แทนที่จะลบข้อมูล"""
        dialog = tk.Toplevel(self)
        dialog.title("เลือกผู้ใช้สำหรับข้อมูลย้อนหลัง")
        dialog.configure(bg=COLORS["surface"])
        dialog.geometry("360x520")
        dialog.minsize(320, 420)
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(dialog, text="👥 เลือกผู้ใช้ที่ต้องการดึงข้อมูล (ติ๊กได้หลายคน)",
                 bg=COLORS["surface"], fg=COLORS["text"], font=FONT_TH_B,
                 wraplength=290, justify="left"
                 ).pack(anchor="w", padx=14, pady=(14, 6), side="top")

        bottom_section = tk.Frame(dialog, bg=COLORS["surface"])
        bottom_section.pack(side="bottom", fill="x")

        status_lbl = tk.Label(bottom_section, text="กำลังดึงรายชื่อจาก Firebase...",
                               bg=COLORS["surface"], fg=COLORS["text_dim"],
                               font=("TH Sarabun New", 9))
        status_lbl.pack(anchor="w", padx=14, pady=(4, 8))

        user_vars: dict[str, tk.BooleanVar] = {}

        def _checked_users():
            return [uid for uid, var in user_vars.items() if var.get()]

        def _update_confirm_btn_label(*_):
            n = len(_checked_users())
            confirm_btn.config(text=f"✅  ยืนยันการเลือก ({n} คน)" if n else "✅  ยืนยันการเลือก")

        def _confirm_selection():
            selected = _checked_users()
            if not selected:
                messagebox.showinfo("แจ้งเตือน", "กรุณาติ๊กเลือกผู้ใช้อย่างน้อย 1 คนก่อน",
                                     parent=dialog)
                return
            self.ml_selected_users = selected
            self._update_ml_user_summary()
            self._log(self.ml_log, f"👥 เลือกผู้ใช้สำหรับข้อมูลย้อนหลัง: {', '.join(selected)}")
            cleanup()
            dialog.destroy()

        confirm_btn = self._btn(bottom_section, "✅  ยืนยันการเลือก", COLORS["accent"],
                                 _confirm_selection)
        confirm_btn.pack(fill="x", padx=14, pady=(0, 8), ipady=6)

        select_row = tk.Frame(bottom_section, bg=COLORS["surface"])
        select_row.pack(fill="x", padx=14, pady=(0, 8))

        def _select_all():
            for var in user_vars.values():
                var.set(True)
            _update_confirm_btn_label()

        def _select_none():
            for var in user_vars.values():
                var.set(False)
            _update_confirm_btn_label()

        tk.Button(select_row, text="เลือกทั้งหมด", bg=COLORS["surface"], fg=COLORS["accent"],
                  activebackground=COLORS["surface"], relief="flat",
                  font=("TH Sarabun New", 10), cursor="hand2", command=_select_all
                  ).pack(side="left")
        tk.Button(select_row, text="ยกเลิกที่เลือกทั้งหมด", bg=COLORS["surface"],
                  fg=COLORS["text_dim"], activebackground=COLORS["surface"], relief="flat",
                  font=("TH Sarabun New", 10), cursor="hand2", command=_select_none
                  ).pack(side="left", padx=(12, 0))

        tk.Frame(bottom_section, bg=COLORS["border"], height=1).pack(fill="x", padx=14, pady=(0, 8))

        list_frame = tk.Frame(dialog, bg=COLORS["surface"])
        list_frame.pack(fill="both", expand=True, padx=14, pady=(0, 4), side="top")

        scroll_area, canvas, inner, cleanup = self._make_touch_scrollable_list(list_frame)
        scroll_area.pack(fill="both", expand=True)

        def _load_users():
            if not (self.firebase.online and self.firebase.db is not None):
                status_lbl.config(text="⚠️ ไม่ได้เชื่อมต่อ Firebase — ไม่สามารถดึงรายชื่อผู้ใช้ได้")
                return
            try:
                users = self.firebase.list_users()
            except Exception as e:
                status_lbl.config(text=f"❌ ดึงรายชื่อไม่สำเร็จ: {e}")
                return
            if not users:
                status_lbl.config(text="ยังไม่มีผู้ใช้คนไหนบันทึกข้อมูลไว้เลย")
                return
            for u in users:
                var = tk.BooleanVar(value=(u in self.ml_selected_users))  # คงค่าที่เคยติ๊กไว้
                row = tk.Frame(inner, bg=COLORS["card"])
                row.pack(fill="x", padx=8, pady=2)
                tk.Checkbutton(row, text=u, variable=var, bg=COLORS["card"], fg=COLORS["text"],
                              selectcolor=COLORS["surface"], activebackground=COLORS["card"],
                              activeforeground=COLORS["text"], font=FONT_TH, anchor="w",
                              command=_update_confirm_btn_label).pack(fill="x")
                user_vars[u] = var
            status_lbl.config(text=f"พบ {len(users)} ผู้ใช้ — ติ๊กเลือกแล้วกด 'ยืนยันการเลือก' ด้านล่าง")
            _update_confirm_btn_label()

        dialog.after(50, _load_users)

        def _on_dialog_close():
            cleanup()
            dialog.destroy()
        dialog.protocol("WM_DELETE_WINDOW", _on_dialog_close)


    # ─── Helpers ─────────────────────────────────────────────────────────────
    def _card_label(self, parent, text, help_text=None):
        f = tk.Frame(parent, bg=COLORS["border"], height=1)
        f.pack(fill="x", padx=12, pady=(10, 0))
        row = tk.Frame(parent, bg=COLORS["surface"])
        row.pack(fill="x", padx=16, pady=(4, 2))
        # ⚠️ เดิม Label นี้ไม่มี wraplength เลย — พอ parent (พาเนลซ้าย) เป็น fixed-width
        # 280px + pack_propagate(False) ข้อความหัวข้อยาวๆ เลยถูกตัดโผล่พ้นขอบพาเนลไปเฉยๆ
        # แทนที่จะขึ้นบรรทัดใหม่ — ใส่ wraplength ให้ขึ้นบรรทัดใหม่แทน (195 เผื่อพื้นที่ปุ่ม
        # ❔ ด้านขวาด้วย ไม่ใช่แค่ความกว้าง 280px เต็มๆ)
        lbl = tk.Label(row, text=text, bg=COLORS["surface"], fg=COLORS["accent"],
                        font=FONT_TH_B, wraplength=180, justify="left", anchor="w")
        lbl.pack(side="left", anchor="w", fill="x", expand=True)
        if help_text:
            self._help_button(row, text, help_text).pack(side="left", padx=(6, 0), anchor="n")

    def _help_button(self, parent, title, message):
        """ปุ่ม ❔ เล็กๆ กดแล้วเด้ง popup อธิบาย — ใช้แทนข้อความอธิบายยาวๆ ที่ค้างอยู่บนจอตลอด
        เพื่อให้หน้าจอโล่งขึ้น แต่ยังกดดูคำอธิบายได้เวลาต้องการ"""
        return tk.Button(parent, text="❔", bg=COLORS["surface"], fg=COLORS["text_dim"],
                          activebackground=COLORS["surface"], relief="flat",
                          font=("TH Sarabun New", 10, "bold"), cursor="hand2", width=2,
                          command=lambda: messagebox.showinfo(title, message))

    def _labeled_entry(self, parent, label, var):
        row = tk.Frame(parent, bg=COLORS["surface"])
        row.pack(fill="x", padx=16, pady=4)
        tk.Label(row, text=label, bg=COLORS["surface"],
                 fg=COLORS["text"], font=FONT_TH, width=10, anchor="w").pack(side="left")
        entry = tk.Entry(row, textvariable=var, bg=COLORS["card"],
                          fg=COLORS["text"], insertbackground=COLORS["text"],
                          relief="flat", font=FONT_TH)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _labeled_combobox(self, parent, label, var, values, command=None):
        """เหมือน _labeled_entry แต่เป็น dropdown เลือกได้จากรายการที่กำหนดตายตัว
        (state="readonly" กันผู้ใช้พิมพ์ค่าเองที่ไม่อยู่ในลิสต์) — ใช้กับ Combobox
        ประเภทบุคคล (ดู self.person_category_var)"""
        row = tk.Frame(parent, bg=COLORS["surface"])
        row.pack(fill="x", padx=16, pady=4)
        tk.Label(row, text=label, bg=COLORS["surface"],
                 fg=COLORS["text"], font=FONT_TH, width=10, anchor="w").pack(side="left")
        combo = ttk.Combobox(row, textvariable=var, values=values, state="readonly",
                              font=FONT_TH)
        combo.pack(side="left", fill="x", expand=True)
        if command:
            combo.bind("<<ComboboxSelected>>", command)
        return combo

    def _stat_box(self, parent, label, init, color):
        card = tk.Frame(parent, bg=COLORS["card"], relief="flat")
        card.pack(side="left", expand=True, fill="x", padx=4, ipadx=6, ipady=6)
        tk.Label(card, text=label, bg=COLORS["card"],
                 fg=COLORS["text_dim"], font=("TH Sarabun New", 10)).pack()
        lbl = tk.Label(card, text=init, bg=COLORS["card"],
                       fg=color, font=("Courier New", 16, "bold"))
        lbl.pack()
        return lbl

    def _btn(self, parent, text, color, cmd):
        # เดิม wraplength=185 คงที่ ตอนนี้ปุ่มเดียวกันถูกใช้ในหลายบริบทที่ความกว้างต่างกันมาก
        # (การ์ดแคบๆ ในแถบตั้งค่าด้านบน ปุ่มเต็มแถวด้านล่าง ฯลฯ) ค่าคงที่ตัวเดียวเลยทำให้
        # บางที่ตัวอักษรขึ้นบรรทัดใหม่ทั้งที่มีที่พอ จนบรรทัดสุดท้ายไปชิดขอบล่างปุ่ม (เช่น
        # "(เลือกได้หลายคน)" ในกรอบสีแดง) — ผูก wraplength เข้ากับความกว้างจริงของปุ่มเอง
        # ผ่าน <Configure> แทน ให้ตัดบรรทัดเฉพาะตอนพื้นที่ไม่พอจริงๆ เท่านั้น ไม่ว่าปุ่มจะอยู่
        # ในคอนเทนเนอร์แคบหรือกว้างแค่ไหนก็ตาม
        btn = tk.Button(parent, text=text, bg=color, fg="#ffffff",
                         activebackground=color, activeforeground="#ffffff",
                         relief="flat", font=FONT_TH_B, cursor="hand2",
                         command=cmd, padx=8, justify="center")
        btn.bind("<Configure>", lambda e: btn.config(wraplength=max(e.width - 14, 40)))
        return btn

    def _log(self, widget, msg, color=None):
        now = datetime.now().strftime("%H:%M:%S")
        line = f"[{now}] {msg}\n"
        self._append_to_log_widget(widget, line)
        # Mirror เข้า Dashboard + Data Log เสมอ (เว้นแต่ widget ที่ส่งมาคืออันใดอันหนึ่ง
        # ในสองอันนี้อยู่แล้ว กัน log ซ้ำ 2 บรรทัด) — รวมกิจกรรมทั้งแอปไว้จุดเดียว
        for mirror in (getattr(self, "dash_activity_log", None),
                       getattr(self, "datalog_text", None)):
            if mirror is not None and mirror is not widget:
                self._append_to_log_widget(mirror, line)

    @staticmethod
    def _append_to_log_widget(widget, line):
        widget.config(state="normal")
        widget.insert("end", line)
        widget.see("end")
        widget.config(state="disabled")

    def _set_status(self, msg):
        self.status_var.set(f"⬤  {msg}")

    # ─── Calibration Logic ────────────────────────────────────────────────────
    def _on_threshold(self, val):
        self.threshold = float(val)
        self.release_threshold = max(0.0, self.threshold - self.hysteresis_margin)
        self.thr_lbl.config(text=f"{self.threshold:.3f}V")
        self.stream.set_threshold(self.threshold)

    def _suggest_threshold_from_calibration(self):
        """คำนวณ threshold ที่เหมาะกับ user ปัจจุบันจากข้อมูล Calibration ที่เพิ่งบันทึกไว้
        (self.calib_samples — เก็บตอนกด '✊ กำมือ' / '🖐 คลายมือ' ในหน้านี้) แทนที่จะใช้
        threshold ตายตัวค่าเดียวสำหรับทุกคน เพราะแต่ละคนออกแรงกำมือได้ไม่เท่ากัน (ยิ่งมี
        person_category ต่างกันมาก เช่น เด็ก/ผู้สูงอายุ/ผู้ป่วยกล้ามเนื้อ ยิ่งต่างกันชัด) —
        ต้องมี samples ทั้ง 'คลายมือ' (rest/noise floor) และ 'กำมือ' (contraction) อย่างน้อย
        อย่างละ 10 samples ถึงจะคำนวณได้อย่างมีความหมาย

        วิธีคิด: หาเส้นแบ่งกึ่งกลางระหว่าง rest_high (RMS สูงสุดที่มักเจอตอนพัก — ใช้
        percentile 90 กัน outlier หลุดโด่ง) กับ fist_low (RMS ต่ำสุดที่มักเจอตอนกำมือ —
        percentile 10) ถ้าสองช่วงไม่คาบเกี่ยวกันเลย (fist_low > rest_high) แปลว่าสัญญาณ
        แยกออกจากกันชัดเจน ตั้ง threshold ตรงกลางพอดี ปลอดภัยสุด ถ้าคาบเกี่ยวกัน (fist_low
        <= rest_high) แปลว่าสัญญาณ 2 ช่วงแยกกันไม่เด็ดขาด (มักเกิดจากตำแหน่ง electrode/gain
        ของวงจร ไม่ใช่ปัญหา software) — ยังคำนวณให้ได้ค่าที่ดีที่สุดเท่าที่ทำได้ (กึ่งกลาง
        ของค่าเฉลี่ยทั้งคู่) แต่จะแจ้งเตือนผู้ใช้ให้รู้ตัวชัดเจนว่าอาจไม่เสถียรร้อยเปอร์เซ็นต์"""
        rest_vals = sorted(v for v, g, _ in self.calib_samples if g == "คลายมือ")
        fist_vals = sorted(v for v, g, _ in self.calib_samples if g == "กำมือ")

        if len(rest_vals) < 10 or len(fist_vals) < 10:
            messagebox.showwarning(
                "ข้อมูลไม่พอ",
                f"ต้องมีข้อมูลทั้ง 'คลายมือ' และ 'กำมือ' อย่างน้อยอย่างละ 10 samples\n"
                f"ตอนนี้มี คลายมือ={len(rest_vals)}, กำมือ={len(fist_vals)}\n\n"
                "กรุณากดปุ่ม '✊ กำมือ' และ '🖐 คลายมือ' บันทึกอย่างละสักครู่ก่อน")
            return

        def _pct(vals, p):
            idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
            return vals[idx]

        rest_high = _pct(rest_vals, 0.90)
        fist_low = _pct(fist_vals, 0.10)
        rest_mean = statistics.mean(rest_vals)
        fist_mean = statistics.mean(fist_vals)

        overlap = fist_low <= rest_high
        if overlap:
            # คาบเกี่ยวกัน — ใช้ค่าเฉลี่ยกึ่งกลางแทน แต่บังคับให้มีช่องว่างขั้นต่ำเสมอ
            # (hysteresis_margin) กันเส้นบน/ล่างกลับด้านหรือชนกันพอดี
            mid = (rest_mean + fist_mean) / 2
            half_gap = max(self.hysteresis_margin, 0.005) / 2
            upper, lower = mid + half_gap, max(0.0, mid - half_gap)
        else:
            # แยกกันชัดเจน — ใช้ค่าจริงจากข้อมูลตรงๆ เป็น 2 เส้น: เส้นบน (fist_low) คือจุด
            # ต่ำสุดที่ยังนับว่า "กำมือแน่ๆ" เส้นล่าง (rest_high) คือจุดสูงสุดที่ยังนับว่า
            # "พักแน่ๆ" ช่วงระหว่างสองเส้นนี้คือโซน noise ที่ไม่ตัดสินใจเปลี่ยนสถานะ
            upper, lower = fist_low, rest_high

        upper = max(float(self.thr_slider.cget("from")),
                    min(float(self.thr_slider.cget("to")), upper))
        self.hysteresis_margin = upper - lower  # จำช่องว่างจริงไว้ ใช้ต่อถ้าลาก slider เอง

        self.thr_slider.set(upper)  # trigger _on_threshold() → ได้ release_threshold คร่าวๆ
        self.release_threshold = max(0.0, lower)  # ทับด้วยค่า "ล่าง" จริงจากข้อมูล (แม่นกว่า
        # margin คงที่ที่ _on_threshold คำนวณให้อัตโนมัติ)
        if hasattr(self, "settings_thr_scale"):
            self.settings_thr_scale.set(upper)

        msg = (f"🎯 คำนวณ threshold จากข้อมูลของ [{self._current_user_id}]: "
               f"พัก≈{rest_mean:.3f}V (90%ile={rest_high:.3f}V) / "
               f"กำมือ≈{fist_mean:.3f}V (10%ile={fist_low:.3f}V) → "
               f"เส้นบน(กำมือ)={upper:.3f}V / เส้นล่าง(พัก)={self.release_threshold:.3f}V")
        self._log(self.log_box, msg)
        if overlap:
            messagebox.showwarning(
                "แยกสัญญาณไม่ชัดเจน",
                "ค่า RMS ตอน 'พัก' กับ 'กำมือ' ของ user นี้คาบเกี่ยวกันอยู่บ้าง "
                "(มักมาจากตำแหน่ง electrode หรือ gain ของวงจร ไม่ใช่ปัญหาซอฟต์แวร์)\n\n"
                f"เส้นที่คำนวณให้ (บน={upper:.3f}V / ล่าง={self.release_threshold:.3f}V) "
                "เป็นค่าที่ดีที่สุดเท่าที่ข้อมูลนี้ "
                "ทำได้ แต่ยังอาจไม่เสถียร 100% แนะนำลองปรับตำแหน่ง electrode แล้วเก็บ "
                "Calibration ใหม่อีกรอบ")
        else:
            messagebox.showinfo("สำเร็จ", msg)

    def _on_servo_source_change(self):
        """เรียกตอนผู้ใช้กดสลับ radio button แหล่งสั่งงาน servo — ระหว่าง 'threshold'
        (ค่าเริ่มต้น เดิมทุกประการ) กับ 'ml_model' (ให้โมเดล multi-feature ตัดสินใจแทน)

        ก่อนยอมให้สลับไป ml_model ต้องเช็คว่ามีโมเดลที่เทรนสำเร็จแล้วจริงๆ ก่อน (ดู
        self.trainer._initialized ผ่าน get_snapshot) ไม่งั้น servo จะไม่มีอะไรสั่งงานเลย —
        ถ้ายังไม่มีโมเดล ดีดกลับไป 'threshold' ให้อัตโนมัติพร้อมแจ้งเตือน (fail-safe เสมอ
        ไม่ปล่อยให้อยู่ในสถานะที่ servo ไม่มีแหล่งสั่งงานที่ใช้งานได้จริง)

        สลับไป ml_model สำเร็จแล้ว ยังมี fallback อีกชั้นระหว่างใช้งานจริงใน _on_emg_sample
        (ถ้าโมเดลความมั่นใจต่ำกว่า self._ML_SERVO_MIN_CONFIDENCE ในบาง sample จะสลับไปใช้
        threshold ให้ชั่วคราวเฉพาะ sample นั้น) — ไม่ใช่ all-or-nothing"""
        if self.servo_source_var.get() == "ml_model":
            snap = self.trainer.get_snapshot()
            if not snap.get("initialized"):
                messagebox.showwarning(
                    "ยังไม่มีโมเดลพร้อมใช้งาน",
                    "ยังไม่มีโมเดลที่เทรนสำเร็จในแท็บ Machine Learning — ไปที่แท็บนั้นแล้ว "
                    "'โหลดข้อมูลย้อนหลัง & เทรน' หรือเริ่มบันทึก Real-time ก่อน แล้วค่อยกลับมา "
                    "เลือกโหมดนี้อีกครั้ง\n\nตอนนี้จะใช้ RMS Threshold ต่อไปก่อน")
                self.servo_source_var.set("threshold")
                self._update_servo_source_status()
                return
            if not messagebox.askyesno(
                    "ยืนยันเปลี่ยนแหล่งสั่งงาน Servo",
                    "จะให้โมเดล ML เป็นคนตัดสินใจสั่ง servo จริงแทน RMS Threshold เดิม\n\n"
                    "ระบบจะ fallback กลับไปใช้ RMS Threshold ให้อัตโนมัติเฉพาะตอนที่โมเดล "
                    "ความมั่นใจต่ำ (<{:.0f}%) เพื่อความปลอดภัย แต่ถ้าโมเดลยังไม่แม่นยำพอ "
                    "การเคลื่อนไหวของมือเทียมอาจไม่เสถียรเท่า Threshold เดิม\n\n"
                    "ยืนยันเปลี่ยนหรือไม่?".format(self._ML_SERVO_MIN_CONFIDENCE)):
                self.servo_source_var.set("threshold")
                self._update_servo_source_status()
                return
        self._log(self.dash_activity_log if hasattr(self, "dash_activity_log") else self.ml_log,
                   f"⚙ เปลี่ยนแหล่งสั่งงาน Servo → "
                   f"{'ML Model' if self.servo_source_var.get() == 'ml_model' else 'RMS Threshold'}")
        self._update_servo_source_status()

    def _update_servo_source_status(self):
        if self.servo_source_var.get() == "ml_model":
            self.servo_source_status_lbl.config(
                text=f"🤖 ใช้โมเดล ML สั่งงาน (fallback เป็น Threshold ถ้ามั่นใจ < "
                     f"{self._ML_SERVO_MIN_CONFIDENCE:.0f}%)", fg=COLORS["accent2"])
        else:
            self.servo_source_status_lbl.config(text="🔧 ใช้ RMS Threshold แบบเดิม", fg=COLORS["text_dim"])

    def _start_record(self, gesture):
        self.recording = True
        self._current_calib_gesture = gesture
        # ปุ่มนี้ใช้ "บังคับ" ท่าตอนอยู่โหมดจำลอง (ไม่มี Hardware) เพื่อ dev/ทดสอบ UI
        # ตอนต่อ Hardware จริงแล้ว ผู้ใช้ต้องขยับมือจริงระหว่างกด แทนการกดสลับสถานะ
        self.stream.set_simulated_contracting(gesture == "กำมือ")
        self.btn_stop.config(state="normal")
        self.btn_grip.config(state="disabled")
        self.btn_release.config(state="disabled")
        self._log(self.log_box, f"🔴 เริ่มบันทึก: {gesture}")
        self._set_status(f"กำลังบันทึกสัญญาณ — {gesture}")

    def _stop_record(self):
        self.recording = False
        self.stream.set_simulated_contracting(False)
        self.btn_stop.config(state="disabled")
        self.btn_grip.config(state="normal")
        self.btn_release.config(state="normal")
        n = len(self.calib_samples)
        self._log(self.log_box, f"⏹ หยุดบันทึก — รวม {n} samples")
        self._set_status("หยุดบันทึก")

    # ─── จำลองเก็บข้อมูลอัตโนมัติ (นับถอยหลังสลับคลายมือ/กำมือ) ──────────────────
    _AUTO_SIM_REST_SEC = 5
    _AUTO_SIM_FIST_SEC = 5
    _AUTO_SIM_SAMPLES_PER_PHASE = 50  # เก็บให้ได้ 50 samples ต่อช่วง (คลายมือ/กำมือ) ต่อ 1 รอบ เสมอ
    # ไม่เริ่มเก็บ sample ทันทีที่ label เปลี่ยน — รอ 0.5 วิให้กล้ามเนื้อขยับตามคำสั่งจริงก่อน
    # (electromechanical delay: สั่ง label ใหม่ปุ๊บ RMS ไม่ได้กระโดดตามทันที ยังค้างอยู่แถวๆ
    # ค่าเดิมของ phase ก่อนหน้าอยู่พักนึง) ถ้าเก็บตั้งแต่ t=0 ของ phase เป๊ะๆ จะได้ sample ที่
    # label ผิดจากสัญญาณจริงปนเข้าไปเพียบ (วัดจากข้อมูลจริงที่เก็บได้: ตัด 0.5 วิแรกออก
    # ทำให้ความแม่นยำโมเดลขยับจาก ~84% เป็น ~96% เพราะ error ส่วนใหญ่มาจากช่วงรอยต่อนี้)
    _AUTO_SIM_LABEL_WARMUP_SEC = 0.5

    def _start_auto_simulate(self):
        """เริ่มจำลองเก็บข้อมูลอัตโนมัติ — ต้องมีรหัสผู้ใช้ก่อนเสมอ (เหมือนปุ่ม 'บันทึก')
        กันไม่ให้ label ไปปนกับ user คนอื่นหรือไม่มี user เลย"""
        if self._auto_sim_running:
            return

        self._confirm_user_id()  # ยืนยันรหัสผู้ใช้ในช่องปัจจุบันก่อน (ถ้ายังไม่เคยกด Enter/คลิกออก)
        uid = self.user_id_var.get().strip()
        if not uid:
            messagebox.showwarning("แจ้งเตือน",
                                    "กรุณาใส่หรือเลือกรหัสผู้ใช้ก่อนเริ่มจำลองเก็บข้อมูล",
                                    parent=self)
            return

        try:
            cycles = int(self.auto_sim_cycles_var.get())
        except (ValueError, tk.TclError):
            cycles = 0
        if cycles < 1:
            cycles = 5
            self.auto_sim_cycles_var.set(str(cycles))

        # ปิดปุ่มควบคุมด้วยมือระหว่างรันอัตโนมัติ กันสั่งซ้อนกันจนสถานะปนกัน
        self._auto_sim_running = True
        self._auto_sim_cycle = 0
        self._auto_sim_total_cycles = cycles
        # จำ uid ตอนเริ่ม + ตำแหน่งเริ่มต้นใน calib_samples ไว้ เพื่อตอนจบจะได้รู้ว่า
        # sample ไหนบ้างเป็นของรอบจำลองนี้โดยเฉพาะ (ไม่ปนกับที่เก็บด้วยมือไว้ก่อนหน้า)
        # แล้วเอาไปบันทึกขึ้น Firebase เป็น user ใหม่แยกต่างหาก (ดู _stop_auto_simulate)
        self._auto_sim_base_uid = uid
        self._auto_sim_start_index = len(self.calib_samples)
        self.btn_grip.config(state="disabled")
        self.btn_release.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.btn_auto_sim.config(state="disabled", text="⏳  กำลังจำลอง...")
        self.btn_auto_sim_stop.config(state="normal")
        self._log(self.log_box,
                  f"🤖 เริ่มจำลองเก็บข้อมูลอัตโนมัติสำหรับ [{uid}] — {cycles} รอบ "
                  f"(คลายมือ {self._AUTO_SIM_REST_SEC}วิ / กำมือ {self._AUTO_SIM_FIST_SEC}วิ ต่อรอบ)")
        self._set_status(f"กำลังจำลองเก็บข้อมูล — รอบ 1/{cycles}")
        self._run_auto_phase("คลายมือ", self._AUTO_SIM_REST_SEC)

    def _run_auto_phase(self, gesture, seconds):
        """เริ่ม phase ใหม่ของการจำลอง ('คลายมือ' หรือ 'กำมือ') แล้วเริ่มนับถอยหลัง

        เก็บ sample ด้วยจังหวะคงที่ของตัวเอง (ไม่ผูกกับ _tick ที่วิ่งตาม UI framerate
        ~25fps เหมือนตอนกดปุ่ม 'กำมือ'/'คลายมือ' ด้วยมือ) เพื่อให้ได้จำนวน sample
        เท่ากันเป๊ะทุกรอบ คือ 50 samples ต่อ 5 วินาที (ดู _start_phase_sampling) —
        self.recording ยังตั้งเป็น True ไว้เหมือนเดิมเพื่อโชว์ตัวบ่งชี้ 'กำลังบันทึก'
        (กรอบสีแดง) บนกราฟตามปกติ แต่ _tick จะไม่ append เข้า calib_samples ซ้ำเอง
        ระหว่างนี้ (เช็คผ่าน self._auto_sim_collecting — ดู _tick)

        เริ่มเก็บ sample จริงหลังรอ _AUTO_SIM_LABEL_WARMUP_SEC วิก่อนเสมอ (ดู docstring
        ค่าคงที่ด้านบน) — ตัวนับถอยหลังบนจอ (_auto_sim_tick) ยังเริ่มทันทีตามปกติ ไม่รอ
        warmup ด้วย เพราะแค่โชว์สถานะให้ผู้ใช้ดู ไม่กระทบคุณภาพข้อมูล"""
        self.recording = True
        self._current_calib_gesture = gesture
        # กรณีไม่มี Hardware จริง (โหมดจำลองสัญญาณ dev) ให้สลับสัญญาณจำลองตามจังหวะไปด้วย
        self.stream.set_simulated_contracting(gesture == "กำมือ")
        icon = "🖐" if gesture == "คลายมือ" else "✊"
        self._cancel_auto_sim_sample_timer()  # กันเผื่อมี timer ค้างจาก phase ก่อนหน้า (รวม warmup)
        remaining_sec = max(0.1, seconds - self._AUTO_SIM_LABEL_WARMUP_SEC)
        # ลดจำนวน sample ตามสัดส่วนเวลาที่เหลือหลังตัด warmup ออก เพื่อให้ความถี่ในการเก็บ
        # (~10Hz) เท่าเดิม ไม่ใช่อัดจำนวนเท่าเดิม (50) ลงในเวลาที่สั้นลง
        count_after_warmup = max(1, round(self._AUTO_SIM_SAMPLES_PER_PHASE * remaining_sec / seconds))
        self._auto_sim_warmup_after_id = self.after(
            round(self._AUTO_SIM_LABEL_WARMUP_SEC * 1000),
            lambda: self._start_phase_sampling(gesture, remaining_sec, count_after_warmup))
        self._auto_sim_tick(gesture, seconds, icon)

    def _start_phase_sampling(self, gesture, seconds, count):
        """เก็บ sample ให้ได้ครบ `count` ตัวเท่าๆ กัน กระจายตลอด `seconds` วินาที
        (ห่างกันตัวละ seconds*1000/count มิลลิวินาที) — เก็บตัวแรกทันที แล้วที่เหลือ
        ตามจังหวะ interval ไปเรื่อยๆ จนครบ"""
        self._cancel_auto_sim_sample_timer()  # กันเผื่อมี timer ค้างจาก phase ก่อนหน้า
        self._auto_sim_collecting = True
        interval_ms = max(1, round(seconds * 1000 / count))
        self._auto_sim_sample_tick(gesture, count, interval_ms)

    def _auto_sim_sample_tick(self, gesture, remaining, interval_ms):
        if not self._auto_sim_running:
            return
        with self._latest_lock:
            rms = self._latest["rms"]
            features = self._latest.get("features")
        self.calib_samples.append((rms, gesture, features))
        remaining -= 1
        if remaining <= 0:
            self._auto_sim_collecting = False
            self._auto_sim_sample_after_id = None
            return
        self._auto_sim_sample_after_id = self.after(
            interval_ms, lambda: self._auto_sim_sample_tick(gesture, remaining, interval_ms))

    def _cancel_auto_sim_sample_timer(self):
        self._auto_sim_collecting = False
        if self._auto_sim_sample_after_id is not None:
            try:
                self.after_cancel(self._auto_sim_sample_after_id)
            except Exception:
                pass
            self._auto_sim_sample_after_id = None
        if self._auto_sim_warmup_after_id is not None:
            try:
                self.after_cancel(self._auto_sim_warmup_after_id)
            except Exception:
                pass
            self._auto_sim_warmup_after_id = None

    def _auto_sim_tick(self, gesture, remaining, icon):
        if not self._auto_sim_running:
            return
        self.auto_sim_status_lbl.config(
            text=f"{icon} {gesture} — เหลือ {remaining} วิ  "
                 f"(รอบ {self._auto_sim_cycle + 1}/{self._auto_sim_total_cycles})")
        if remaining <= 0:
            self._auto_sim_next_phase(gesture)
            return
        self._auto_sim_after_id = self.after(
            1000, lambda: self._auto_sim_tick(gesture, remaining - 1, icon))

    def _auto_sim_next_phase(self, finished_gesture):
        if not self._auto_sim_running:
            return
        if finished_gesture == "คลายมือ":
            # จบช่วงพัก -> ต่อด้วยกำมือในรอบเดียวกัน
            self._run_auto_phase("กำมือ", self._AUTO_SIM_FIST_SEC)
        else:
            # จบช่วงกำมือ -> ครบ 1 รอบ
            self._auto_sim_cycle += 1
            self._log(self.log_box,
                      f"   ↳ รอบที่ {self._auto_sim_cycle}/{self._auto_sim_total_cycles} "
                      f"เสร็จแล้ว (รวม {len(self.calib_samples)} samples)")
            if self._auto_sim_cycle >= self._auto_sim_total_cycles:
                self._stop_auto_simulate(finished=True)
            else:
                self._set_status(
                    f"กำลังจำลองเก็บข้อมูล — รอบ {self._auto_sim_cycle + 1}/{self._auto_sim_total_cycles}")
                self._run_auto_phase("คลายมือ", self._AUTO_SIM_REST_SEC)

    def _stop_auto_simulate(self, finished=False):
        """หยุดการจำลอง — ถ้า finished=True แปลว่าจบครบทุกรอบเอง ถ้า False แปลว่าผู้ใช้กด
        หยุดเองกลางคัน จากนั้นจะเอา sample ที่เก็บได้ในรอบนี้ (ตั้งแต่ _auto_sim_start_index
        เป็นต้นไป) บันทึกขึ้น Firebase ให้อัตโนมัติทันที โดยตั้งใจสร้างเป็น "user ใหม่" แยก
        ต่างหากจาก user เดิม — เติมคำว่า '_label' ต่อท้ายรหัสผู้ใช้ (เช่น 'Te' -> 'Te_label')
        เพื่อไม่ให้ข้อมูลที่ผ่านการ label ชัดเจนจากการจำลองนี้ ไปปนกับข้อมูล Real-time ปกติ
        ของ user เดิม — แยกกันไว้ให้เอาไปเปรียบเทียบกันได้ (เช่น เทรน/เทียบผลโมเดลระหว่าง
        ชุดข้อมูล Real-time กับชุดข้อมูลที่มี label ยืนยันแล้ว)"""
        self._auto_sim_running = False
        if self._auto_sim_after_id is not None:
            try:
                self.after_cancel(self._auto_sim_after_id)
            except Exception:
                pass
            self._auto_sim_after_id = None
        self._cancel_auto_sim_sample_timer()

        self.recording = False
        self.stream.set_simulated_contracting(False)
        self.btn_grip.config(state="normal")
        self.btn_release.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_auto_sim.config(state="normal", text="▶  เริ่มจำลองเก็บข้อมูล")
        self.btn_auto_sim_stop.config(state="disabled")

        new_samples = self.calib_samples[self._auto_sim_start_index:]
        n = len(new_samples)
        if finished:
            self.auto_sim_status_lbl.config(
                text=f"✅ จำลองครบ {self._auto_sim_total_cycles} รอบแล้ว")
            self._log(self.log_box, f"✅ จำลองเก็บข้อมูลอัตโนมัติเสร็จสิ้น — รวม {n} samples")
            self._set_status("จำลองเก็บข้อมูลเสร็จสิ้น")
        else:
            self.auto_sim_status_lbl.config(text="⏹ หยุดจำลองแล้ว")
            self._log(self.log_box, f"⏹ หยุดจำลองเก็บข้อมูลกลางคัน — รวม {n} samples")
            self._set_status("หยุดจำลองเก็บข้อมูล")

        if n > 0 and self._auto_sim_base_uid:
            self._save_auto_sim_samples_to_firebase(new_samples, self._auto_sim_base_uid)

    def _save_auto_sim_samples_to_firebase(self, samples, base_uid):
        """บันทึกข้อมูลจากการ 'จำลองเก็บข้อมูลอัตโนมัติ' ขึ้น Firebase ให้ทันที โดยไม่ต้องรอ
        กดปุ่ม 'บันทึก' — สร้าง/ใช้ user ใหม่ที่ชื่อ '{base_uid}_label' เจตนาแยกออกจาก user
        เดิม (เช่น 'Te' -> 'Te_label') เพื่อให้ชุดข้อมูลที่มี label ยืนยันแน่นอนจากการจำลองนี้
        ไม่ปนกับชุดข้อมูล Real-time ของ user เดิม สามารถเอาทั้งสอง user นี้ไปเปรียบเทียบ/เทรน
        โมเดลแยกกันดูได้ในแท็บ Machine Learning ภายหลัง

        แก้ 2 บั๊กที่เจอจากการใช้งานจริง:
        1. ถ้า base_uid ลงท้ายด้วย '_label' อยู่แล้ว (เช่น เผลอเลือก 'Test_01_label' เป็น
           user ปัจจุบันแล้วกดจำลองซ้ำ) ห้ามเติม '_label' ต่ออีก ไม่งั้นจะเพี้ยนเป็น
           'Test_01_label_label' ไปเรื่อยๆ ทุกรอบที่กดซ้ำ
        2. เดิมเซฟผ่าน log_calibration() ซึ่งเขียนเป็น record เดียวก้อนใหญ่ type='calibration'
           ลง node 'calibrations' — ทำให้ไม่โผล่ใน node 'sessions' เลย (มีแค่ known_users
           ที่เป็นแค่ทะเบียนชื่อ ไม่มีข้อมูลจริง) และ ML tab ก็อ่านไม่ได้ด้วย (ดึงเฉพาะ
           record type='sample' จาก node 'sessions' เท่านั้น — ดู _fetch_firebase_samples)
           แก้โดยเขียนทีละ sample ด้วย log_labeled_sample() แทน ซึ่งลง node 'sessions'
           รูปแบบเดียวกับข้อมูล Real-time ทุกประการ (type='sample') ทำให้ทั้งเห็นใน
           Firebase ใต้ sessions/{label_uid} และโหลดเข้า ML tab ได้ปกติ"""
        label_uid = base_uid if base_uid.endswith("_label") else f"{base_uid}_label"
        self.firebase.start()  # จำลองอัตโนมัติ = เจตนาชัดเจนว่าจะส่งขึ้น Firebase แล้ว
        self.firebase.register_known_user(label_uid)  # ให้โผล่ในรายชื่อ user ทุกที่ทันที
        session_id = f"autolabel_{datetime.now():%Y%m%d_%H%M%S}"
        for i, (rms, gesture, features) in enumerate(samples):
            f = features or {}
            self.firebase.log_labeled_sample(
                label_uid, rms=rms, gesture=gesture, t=round(i * 0.1, 4), session_id=session_id,
                mav=f.get("mav"), variance=f.get("variance"),
                waveform_length=f.get("waveform_length"), zero_crossing=f.get("zero_crossing"),
                slope_sign_change=f.get("slope_sign_change"), iemg=f.get("iemg"),
                person_category=self._current_person_category)
        online_note = "จะซิงก์ขึ้น Firebase อัตโนมัติ" if self.firebase.online else \
                      "รอเน็ตแล้วจะซิงก์ขึ้น Firebase ให้อัตโนมัติ (ตอนนี้ Offline)"
        self._log(self.log_box,
                  f"💾 บันทึก {len(samples)} samples จากการจำลองอัตโนมัติ ลงผู้ใช้ใหม่ "
                  f"[{label_uid}] แล้ว (เข้า node 'sessions' แบบเดียวกับข้อมูล Real-time — "
                  f"แยกจาก [{base_uid}] เพื่อเทียบกัน) — {online_note}")

    def _open_change_user_dialog(self):
        """Dialog เปลี่ยนผู้ใช้งานสำหรับหน้า Dashboard (ใช้กำหนดว่ากำลังบันทึกสัญญาณ
        ให้ user คนไหนอยู่ — self.user_id_var / self._current_user_id) แยกออกจากการ
        เลือก user เพื่อ "ดึงข้อมูลย้อนหลังมาเทรน" ในแท็บ Machine Learning โดยเจตนา
        (ดู self.ml_selected_users ในแท็บนั้น) เพราะ Dashboard เน้นบันทึกข้อมูลเข้า Firebase
        ส่วน Machine Learning เน้นดึงข้อมูลออกมาเทรน คนละทิศทางกัน ไม่ควรผูกกัน

        Layout: ส่วน "เพิ่มผู้ใช้ใหม่" + ปุ่มเลือก + สถานะ pack ด้วย side="bottom" ก่อน
        เสมอ (จองพื้นที่ด้านล่างไว้ก่อน) แล้วค่อย pack listbox แบบ expand=True ทีหลัง
        กันปัญหาเดิมที่ปุ่ม/ช่องเพิ่ม user ใหม่โดนดันหลุดจอไปเวลารายชื่อ user มีเยอะ หรือ
        หน้าต่างเล็กเกินไป — ตอนนี้มองเห็นและกดได้เสมอไม่ว่ารายชื่อจะยาวแค่ไหน"""
        dialog = tk.Toplevel(self)
        dialog.title("เปลี่ยนผู้ใช้งาน")
        dialog.configure(bg=COLORS["surface"])
        dialog.geometry("340x520")
        dialog.minsize(300, 420)
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(dialog, text="👥 เลือกผู้ใช้ที่เคยบันทึกไว้บน Firebase", bg=COLORS["surface"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=14, pady=(14, 6), side="top")

        # ── ส่วนล่าง: pack ด้วย side="bottom" ก่อน (เรียงจากล่างสุดขึ้นบน) ─────────
        bottom_section = tk.Frame(dialog, bg=COLORS["surface"])
        bottom_section.pack(side="bottom", fill="x")
        tk.Frame(bottom_section, bg=COLORS["border"], height=1).pack(fill="x", padx=14, pady=(6, 10))
        tk.Label(bottom_section, text="หรือเพิ่มผู้ใช้ใหม่", bg=COLORS["surface"], fg=COLORS["text"],
                 font=FONT_TH_B).pack(anchor="w", padx=14)
        new_user_var = tk.StringVar(value="")
        new_entry = tk.Entry(bottom_section, textvariable=new_user_var, bg=COLORS["card"],
                              fg=COLORS["text"], insertbackground=COLORS["text"], relief="flat",
                              font=FONT_TH)
        new_entry.pack(fill="x", padx=14, pady=(4, 6))

        def _add_new():
            uid = new_user_var.get().strip()
            if not uid:
                messagebox.showwarning("แจ้งเตือน", "กรุณาใส่ชื่อผู้ใช้ใหม่ก่อน", parent=dialog)
                return
            self.user_id_var.set(uid)
            self._confirm_user_id()
            dialog.destroy()

        new_entry.bind("<Return>", lambda e: _add_new())
        self._btn(bottom_section, "➕  ใช้ผู้ใช้ใหม่นี้", COLORS["success"], _add_new
                  ).pack(fill="x", padx=14, pady=(0, 14), ipady=6)

        status_lbl = tk.Label(dialog, text="กำลังดึงรายชื่อจาก Firebase...", bg=COLORS["surface"],
                               fg=COLORS["text_dim"], font=("TH Sarabun New", 9))
        status_lbl.pack(anchor="w", padx=14, pady=(4, 0), side="bottom")

        def _select_from_list():
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("แจ้งเตือน", "กรุณาเลือกผู้ใช้จากรายชื่อก่อน (หรือพิมพ์เพิ่มผู้ใช้ใหม่ด้านล่าง)",
                                     parent=dialog)
                return
            self.user_id_var.set(listbox.get(sel[0]))
            self._confirm_user_id()
            dialog.destroy()

        self._btn(dialog, "✅  เลือกผู้ใช้นี้", COLORS["accent"], _select_from_list
                  ).pack(fill="x", padx=14, pady=(8, 4), ipady=6, side="bottom")

        # ── ส่วนบน: listbox กินพื้นที่ที่เหลือทั้งหมด (expand=True) ─────────────
        list_frame = tk.Frame(dialog, bg=COLORS["surface"])
        list_frame.pack(fill="both", expand=True, padx=14, pady=(0, 4), side="top")
        # grid แทน pack(side=left/right) — กันปัญหาคอลัมน์ scrollbar โดนบีบจนไม่เหลือ
        # พื้นที่แสดงผล (เจอจริงตอนทดสอบ: pack แบบเดิมบางเครื่องทำให้ scrollbar หายไปเลย)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_columnconfigure(1, weight=0, minsize=48)

        listbox = tk.Listbox(list_frame, bg=COLORS["card"], fg=COLORS["text"],
                              selectbackground=COLORS["accent"], relief="flat",
                              font=FONT_TH, activestyle="none")
        listbox.grid(row=0, column=0, sticky="nsew")
        listbox.bind("<Double-Button-1>", lambda e: _select_from_list())

        # Scrollbar กว้างขึ้น + ปุ่มลูกศร ▲/▼ ใหญ่ — เหมือนกับ _make_touch_scrollable_list
        # (ใช้ไม่ได้ตรงๆ เพราะอันนั้นออกแบบมาสำหรับ Canvas+Checkbutton หลายแถว ส่วนนี่เป็น
        # tk.Listbox แบบเลือกได้ทีละคน โครงสร้าง widget ต่างกัน) แต่แก้ปัญหาเดียวกัน: บน
        # Raspberry Pi ที่เป็น touchscreen ไม่มี mouse wheel เลย scrollbar เดิม (~14px)
        # กดโดนยากมาก พอ user เยอะขึ้นเลื่อนหารายชื่อที่ต้องการไม่เจอ — ใช้ CustomScrollbar
        # (วาดเองด้วย Canvas) แทน tk.Scrollbar เพราะค่า width เดิมไม่ถูกเคารพทุกเครื่อง
        scroll_col = tk.Frame(list_frame, bg=COLORS["surface"], width=48)
        scroll_col.grid(row=0, column=1, sticky="ns")
        scroll_col.grid_propagate(False)
        scroll_col.grid_columnconfigure(0, weight=1)
        scroll_col.grid_rowconfigure(1, weight=1)

        tk.Button(scroll_col, text="▲", font=("TH Sarabun New", 13, "bold"),
                  bg=COLORS["card"], fg=COLORS["accent"], activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  command=lambda: listbox.yview_scroll(-3, "units")
                  ).grid(row=0, column=0, sticky="ew", ipady=8)

        sb = CustomScrollbar(scroll_col, command=listbox.yview, width=42,
                              thumb_color=COLORS["accent"], trough_color=COLORS["border"])
        sb.grid(row=1, column=0, sticky="nsew")
        listbox.configure(yscrollcommand=sb.set)

        tk.Button(scroll_col, text="▼", font=("TH Sarabun New", 13, "bold"),
                  bg=COLORS["card"], fg=COLORS["accent"], activebackground=COLORS["border"],
                  relief="flat", cursor="hand2",
                  command=lambda: listbox.yview_scroll(3, "units")
                  ).grid(row=2, column=0, sticky="ew", ipady=8)

        # ดึงรายชื่อ user จาก Firebase หลังสร้างหน้าต่างเสร็จ (ให้ dialog โผล่มาก่อน
        # ไม่ต้องรอ network call เสร็จก่อนค่อยเห็นหน้าต่าง)
        def _load_users():
            if not (self.firebase.online and self.firebase.db is not None):
                status_lbl.config(text="⚠️ ไม่ได้เชื่อมต่อ Firebase — พิมพ์ชื่อใหม่ด้านล่างแทน")
                return
            try:
                users = self.firebase.list_users()
            except Exception as e:
                status_lbl.config(text=f"❌ ดึงรายชื่อไม่สำเร็จ: {e}")
                return
            listbox.delete(0, "end")
            for u in users:
                listbox.insert("end", u)
            status_lbl.config(text=f"พบ {len(users)} ผู้ใช้" if users else
                               "ยังไม่มีผู้ใช้คนไหนบันทึกข้อมูลไว้เลย — เพิ่มคนใหม่ด้านล่าง")
        dialog.after(50, _load_users)

    def _on_person_category_change(self, event=None):
        """เรียกทันทีที่เลือกประเภทบุคคลจาก Combobox — อัพเดต self._current_person_category
        (key ภาษาอังกฤษ ไม่ใช่ label ไทยที่โชว์บนจอ) ให้มีผลกับ sample ถัดไปทันที ทั้งฝั่ง
        บันทึกขึ้น Firebase (_on_emg_sample) และฝั่งทำนายสด (_drive_servo/แดชบอร์ด)"""
        label = self.person_category_var.get()
        self._current_person_category = _PERSON_CATEGORY_LABEL_TO_KEY.get(
            label, DEFAULT_PERSON_CATEGORY)
        self._log(self.log_box, f"👤 ตั้งประเภทบุคคลเป็น: {label}")

    def _confirm_user_id(self, event=None):
        """ยืนยันรหัสผู้ใช้ที่จะใช้บันทึก/sync สัญญาณ EMG แบบต่อเนื่อง — เรียกเฉพาะตอนกด
        Enter หรือคลิกออกจากช่องกรอก (ไม่ใช่ทุกครั้งที่พิมพ์) กันไม่ให้แต่ละตัวอักษรที่พิมพ์
        ระหว่างทางถูกบันทึกเป็นคนละ user แยกกันขึ้น Firebase

        ปุ่ม 'เริ่มบันทึกแบบ Real-time' จะกดได้ก็ต่อเมื่อมีรหัสผู้ใช้ไม่ว่างเปล่าเท่านั้น
        และถ้าเปลี่ยน/ลบรหัสผู้ใช้ระหว่างที่กำลังบันทึกอยู่ จะหยุดบันทึกของ user เดิมอัตโนมัติ
        เสมอ กันไม่ให้ข้อมูลของสอง user ปนกัน หรือแอบบันทึกทับ user_01 แบบเงียบๆ"""
        uid = self.user_id_var.get().strip()

        if not uid:
            # ลบ/เว้นว่างช่องรหัสผู้ใช้ — บั๊กเดิม: โค้ดนี้เคยข้ามการหยุดบันทึกไปเลย
            # ตอน uid ว่าง (เช็คแค่ "if uid and ...") แล้วดันไปเซ็ต self._current_user_id
            # เป็น 'user_01' แทนแบบเงียบๆ โดยไม่หยุด _realtime_logging_active เลย ผลคือ
            # ถ้ากำลังบันทึกอยู่แล้วลบชื่อ user ทิ้ง มันจะยังบันทึกต่อเนื่องเข้า 'user_01'
            # ต่อไปเรื่อยๆ ทั้งที่ปุ่มโชว์ว่า disabled (ดูเหมือนไม่ได้ทำอะไรอยู่) — แก้โดย
            # หยุดบันทึกทันทีทุกครั้งที่ช่องว่างเปล่า ไม่สนใจว่าเคยบันทึกอยู่ก่อนหรือไม่
            if self._realtime_logging_active:
                self._realtime_logging_active = False
                self._end_current_session_if_any()
                self.btn_toggle_realtime.config(text="▶  เริ่มบันทึกแบบ Real-time", bg=COLORS["success"],
                                                 activebackground=COLORS["success"])
                self._log(self.log_box, "⏹ ลบรหัสผู้ใช้ออก — หยุดบันทึก Real-time ให้อัตโนมัติ")
                self._set_status("หยุดบันทึก Real-time (ไม่มีรหัสผู้ใช้)")
            self.btn_toggle_realtime.config(state="disabled")
            if self.event_recorder.enabled:
                self.event_recorder.enabled = False
                self.event_recorder.reset()
                self.dash_event_toggle_btn.config(text="▶  เปิดใช้งาน", bg=COLORS["success"],
                                                   activebackground=COLORS["success"])
                self.dash_event_status_lbl.config(text="⚪ ปิดอยู่", fg=COLORS["text_dim"])
            self.dash_event_toggle_btn.config(state="disabled")
            return

        if uid != self._current_user_id and self._realtime_logging_active:
            self._realtime_logging_active = False
            self._end_current_session_if_any()
            self.btn_toggle_realtime.config(text="▶  เริ่มบันทึกแบบ Real-time", bg=COLORS["success"],
                                             activebackground=COLORS["success"])
            self._log(self.log_box, f"⏹ เปลี่ยนรหัสผู้ใช้เป็น [{uid}] — หยุดบันทึก Real-time "
                                     f"ของ user เดิมให้อัตโนมัติ (กด 'เริ่มบันทึกแบบ Real-time' "
                                     f"อีกครั้งถ้าจะบันทึกต่อให้ user นี้)")

        if uid != self._current_user_id and self.event_recorder.enabled:
            self.event_recorder.enabled = False
            self.event_recorder.reset()
            self.dash_event_toggle_btn.config(text="▶  เปิดใช้งาน", bg=COLORS["success"],
                                               activebackground=COLORS["success"])
            self.dash_event_status_lbl.config(text="⚪ ปิดอยู่", fg=COLORS["text_dim"])
            self._log(self.log_box, f"⏹ เปลี่ยนรหัสผู้ใช้เป็น [{uid}] — ปิดบันทึก Event "
                                     f"กำมืออัตโนมัติของ user เดิมให้ด้วย")

        if uid != self._current_user_id:
            # ลงทะเบียน user นี้ไว้ใน Firebase (known_users) เผื่อยังไม่เคยลงทะเบียนมาก่อน
            # เขียนแค่ตอน uid เปลี่ยนจริงๆ เท่านั้น (ไม่ใช่ทุกครั้งที่ยืนยันซ้ำ uid เดิม)
            # กันยิง network call บ่อยเกินจำเป็น
            self.firebase.register_known_user(uid)

        self._current_user_id = uid
        self.btn_toggle_realtime.config(state="normal")
        self.dash_event_toggle_btn.config(state="normal")

    def _end_current_session_if_any(self):
        """เรียก end_session() ถ้ามี session ที่กำลังเปิดอยู่ (self._current_session_id
        ไม่ใช่ None) — ใช้ทุกที่ที่การบันทึก Real-time หยุดลง ไม่ว่าจะกดหยุดเอง หรือ
        ระบบหยุดให้อัตโนมัติ (เช่น ตอนเปลี่ยน/ลบ user) กัน session ค้างไม่มี end_time"""
        if self._current_session_id:
            self.firebase.end_session(self._current_user_id, self._current_session_id,
                                       self._current_session_start_iso)
            self._log(self.log_box, f"⏹ จบ session [{self._current_session_id}] แล้ว")
        self._current_session_id = None
        self._current_session_start_iso = None

    def _toggle_realtime_logging(self):
        """เปิด/ปิดการบันทึกสัญญาณ EMG แบบต่อเนื่อง (ทุก sample) ลงไฟล์ในเครื่อง + sync
        ขึ้น Firebase สำหรับ user ปัจจุบัน — ปิดไว้เป็นค่าเริ่มต้นเสมอทุกครั้งที่เปิดแอปใหม่

        แต่ละครั้งที่กดเริ่ม จะสร้าง 'session' ใหม่ 1 อัน (session_id ไม่ซ้ำ ผูกกับเวลา
        ที่กด) พร้อมบันทึก metadata (sample_rate, threshold, start_time) ไว้ 1 record
        แยกต่างหาก แล้วทุก sample ในช่วงนี้จะแปะ session_id นี้ไว้อ้างอิงกลับมา — ตอน
        เอาไป train ML จะรู้ได้ทันทีว่า sample ชุดไหนมาจาก session ไหน ใช้ threshold/
        sample rate เท่าไหร่ตอนนั้น โดยไม่ต้องยัด metadata ซ้ำในทุก sample (ดู log_sample
        ใน firebase_sync.py)"""
        self._realtime_logging_active = not self._realtime_logging_active
        if self._realtime_logging_active:
            self.firebase.start()  # เริ่มเธรด sync ขึ้น Firebase ตอนนี้แหละ (ครั้งแรกที่กดเท่านั้น)
            self._current_session_id = f"sess_{datetime.now():%Y%m%d_%H%M%S}"
            session_rec = self.firebase.start_session(
                self._current_user_id, self._current_session_id,
                sample_rate=self.stream.fs, threshold=self.threshold)
            self._current_session_start_iso = session_rec["start_time"]
            self.btn_toggle_realtime.config(text="⏹  หยุดบันทึก Real-time", bg=COLORS["danger"],
                                             activebackground=COLORS["danger"])
            self._log(self.log_box, f"▶ เริ่ม session [{self._current_session_id}] บันทึกสัญญาณ EMG "
                                     f"แบบ Real-time ของ [{self._current_user_id}] ขึ้น Firebase แล้ว "
                                     f"(sample_rate={self.stream.fs}Hz, threshold={self.threshold:.4f}V)")
            self._set_status(f"กำลังบันทึก Real-time — {self._current_user_id}")
        else:
            self._end_current_session_if_any()
            self.btn_toggle_realtime.config(text="▶  เริ่มบันทึกแบบ Real-time", bg=COLORS["success"],
                                             activebackground=COLORS["success"])
            self._log(self.log_box, "⏹ หยุดบันทึกสัญญาณ EMG แบบ Real-time แล้ว")
            self._set_status("หยุดบันทึก Real-time")

    def _toggle_event_recorder(self):
        """เปิด/ปิดตัวบันทึก Event กำมืออัตโนมัติ (GestureEventRecorder) — แยกอิสระจาก
        _toggle_realtime_logging โดยเจตนา เป็นคนละฟีเจอร์กัน (บันทึกทุก sample vs.
        บันทึกเป็นก้อนๆ ตอนตรวจจับการกำมือได้)"""
        self.event_recorder.enabled = not self.event_recorder.enabled
        if self.event_recorder.enabled:
            self.firebase.start()  # เผื่อยังไม่เคยเริ่มเธรด sync — idempotent เรียกซ้ำได้
            self.dash_event_toggle_btn.config(text="⏹  ปิดใช้งาน", bg=COLORS["danger"],
                                               activebackground=COLORS["danger"])
            self.dash_event_status_lbl.config(text="🟡 กำลังรอสัญญาณกำมือ...", fg=COLORS["warning"])
            self._log(self.dash_activity_log, f"🎬 เปิดบันทึก Event กำมืออัตโนมัติของ "
                                               f"[{self._current_user_id}] แล้ว")
        else:
            self.event_recorder.reset()  # ทิ้ง event ที่กำลังบันทึกค้างอยู่ (ถ้ามี) — ยังไม่ครบเงื่อนไข
            self.dash_event_toggle_btn.config(text="▶  เปิดใช้งาน", bg=COLORS["success"],
                                               activebackground=COLORS["success"])
            self.dash_event_status_lbl.config(text="⚪ ปิดอยู่", fg=COLORS["text_dim"])
            self._log(self.dash_activity_log, "⏹ ปิดบันทึก Event กำมืออัตโนมัติแล้ว")

    def _open_delete_users_dialog(self):
        """Dialog แสดงรายชื่อผู้ใช้ทั้งหมดบน Firebase พร้อมติ๊กถูกเพื่อเลือกลบได้ทีละคน
        หรือหลายคนพร้อมกัน — แทนที่ปุ่มลบแบบเดิมที่ผูกกับช่องกรอกรหัสผู้ใช้ด้านบน (ลบได้
        ทีละคนตามที่พิมพ์/เลือกไว้เท่านั้น) เพราะบางทีอยากเคลียร์ข้อมูลทดสอบของหลาย user
        รวดเดียวตอนทำความสะอาดฐานข้อมูล ไม่ต้องเปลี่ยน user ไปมาแล้วกดลบทีละรอบ

        Layout: ส่วนปุ่ม/สถานะด้านล่าง pack ด้วย side="bottom" ก่อนเสมอ (จองพื้นที่ไว้ก่อน)
        แล้วค่อย pack รายชื่อ user แบบ checkbox (ใน Canvas เลื่อนได้) แบบ expand=True ทีหลัง
        เหมือนกับ dialog เปลี่ยนผู้ใช้งาน กันปัญหาปุ่มโดนดันหลุดจอเวลารายชื่อ user มีเยอะ"""
        dialog = tk.Toplevel(self)
        dialog.title("ลบข้อมูลผู้ใช้")
        dialog.configure(bg=COLORS["surface"])
        dialog.geometry("360x540")
        dialog.minsize(320, 420)
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(dialog, text="🗑 เลือกผู้ใช้ที่ต้องการลบ (ติ๊กได้หลายคน)", bg=COLORS["surface"],
                 fg=COLORS["text"], font=FONT_TH_B).pack(anchor="w", padx=14, pady=(14, 6), side="top")

        # ── ส่วนล่าง: pack ด้วย side="bottom" ก่อน (เรียงจากล่างสุดขึ้นบน) ─────────
        bottom_section = tk.Frame(dialog, bg=COLORS["surface"])
        bottom_section.pack(side="bottom", fill="x")

        status_lbl = tk.Label(bottom_section, text="กำลังดึงรายชื่อจาก Firebase...",
                               bg=COLORS["surface"], fg=COLORS["text_dim"],
                               font=("TH Sarabun New", 9))
        status_lbl.pack(anchor="w", padx=14, pady=(4, 8))

        user_vars: dict[str, tk.BooleanVar] = {}
        user_rows: dict[str, tk.Frame] = {}

        def _selected_users():
            return [uid for uid, var in user_vars.items() if var.get()]

        def _update_delete_btn_label(*_):
            n = len(_selected_users())
            delete_btn.config(text=f"🗑  ลบผู้ใช้ที่เลือก ({n} คน)" if n else "🗑  ลบผู้ใช้ที่เลือก")

        def _delete_selected():
            selected = _selected_users()
            if not selected:
                messagebox.showinfo("แจ้งเตือน", "กรุณาติ๊กเลือกผู้ใช้อย่างน้อย 1 คนก่อน",
                                     parent=dialog)
                return
            names = "\n".join(f"  • {u}" for u in selected)
            if not messagebox.askyesno(
                    "ยืนยันการลบ",
                    f"ต้องการลบข้อมูลทั้งหมดของผู้ใช้ {len(selected)} คนนี้ใช่ไหม?\n\n{names}\n\n"
                    "จะลบทั้งบน Firebase (sessions + calibrations) และไฟล์ในเครื่องนี้\n"
                    "การลบนี้ย้อนกลับไม่ได้", icon="warning", parent=dialog):
                return

            delete_btn.config(state="disabled", text="⏳ กำลังลบ...")
            select_all_btn.config(state="disabled")
            select_none_btn.config(state="disabled")
            dialog.update_idletasks()

            ok_list, fail_list = [], []
            for uid in selected:
                # ถ้ากำลังบันทึก Real-time ของ user นี้อยู่พอดี ต้องหยุดก่อน ไม่งั้นลบไปแล้ว
                # ข้อมูลใหม่จะไหลกลับเข้ามาทันทีในวินาทีถัดไป
                if self._realtime_logging_active and uid == self._current_user_id:
                    self._realtime_logging_active = False
                    self._current_session_id = None
                    self._current_session_start_iso = None
                    self.btn_toggle_realtime.config(text="▶  เริ่มบันทึกแบบ Real-time",
                                                     bg=COLORS["success"],
                                                     activebackground=COLORS["success"])
                    self._log(self.log_box,
                              f"⏹ หยุดบันทึก Real-time ของ [{uid}] ก่อนลบข้อมูลอัตโนมัติ")

                ok, msg = self.firebase.delete_user_data(uid)
                self._log(self.log_box, ("✅ " if ok else "❌ ") + msg)
                (ok_list if ok else fail_list).append(uid)

            # เอา user ที่ลบสำเร็จออกจากรายชื่อในหน้าต่างนี้ทันที ไม่ต้องปิด/เปิด dialog ใหม่
            for uid in ok_list:
                row = user_rows.pop(uid, None)
                if row is not None:
                    row.destroy()
                user_vars.pop(uid, None)

            if fail_list:
                status_lbl.config(text=f"ลบสำเร็จ {len(ok_list)} คน — ล้มเหลว {len(fail_list)} คน")
                messagebox.showerror("ลบไม่สำเร็จบางส่วน",
                                      f"ลบไม่สำเร็จ: {', '.join(fail_list)}", parent=dialog)
            else:
                status_lbl.config(text=f"✅ ลบสำเร็จทั้งหมด {len(ok_list)} คน")
                messagebox.showinfo("สำเร็จ", f"ลบข้อมูลผู้ใช้ {len(ok_list)} คนเรียบร้อยแล้ว",
                                     parent=dialog)

            delete_btn.config(state="normal")
            select_all_btn.config(state="normal")
            select_none_btn.config(state="normal")
            _update_delete_btn_label()

        delete_btn = self._btn(bottom_section, "🗑  ลบผู้ใช้ที่เลือก", COLORS["danger"],
                               _delete_selected)
        delete_btn.pack(fill="x", padx=14, pady=(0, 8), ipady=6)

        select_row = tk.Frame(bottom_section, bg=COLORS["surface"])
        select_row.pack(fill="x", padx=14, pady=(0, 8))

        def _select_all():
            for var in user_vars.values():
                var.set(True)
            _update_delete_btn_label()

        def _select_none():
            for var in user_vars.values():
                var.set(False)
            _update_delete_btn_label()

        select_all_btn = tk.Button(select_row, text="เลือกทั้งหมด", bg=COLORS["surface"],
                                    fg=COLORS["accent"], activebackground=COLORS["surface"],
                                    relief="flat", font=("TH Sarabun New", 10), cursor="hand2",
                                    command=_select_all)
        select_all_btn.pack(side="left")
        select_none_btn = tk.Button(select_row, text="ยกเลิกที่เลือกทั้งหมด", bg=COLORS["surface"],
                                     fg=COLORS["text_dim"], activebackground=COLORS["surface"],
                                     relief="flat", font=("TH Sarabun New", 10), cursor="hand2",
                                     command=_select_none)
        select_none_btn.pack(side="left", padx=(12, 0))

        tk.Frame(bottom_section, bg=COLORS["border"], height=1).pack(fill="x", padx=14, pady=(0, 8))

        # ── ส่วนบน: รายชื่อ user แบบ checkbox กินพื้นที่ที่เหลือทั้งหมด (เลื่อนได้) ────
        list_frame = tk.Frame(dialog, bg=COLORS["surface"])
        list_frame.pack(fill="both", expand=True, padx=14, pady=(0, 4), side="top")

        scroll_area, canvas, inner, cleanup = self._make_touch_scrollable_list(list_frame)
        scroll_area.pack(fill="both", expand=True)

        def _load_users():
            if not (self.firebase.online and self.firebase.db is not None):
                status_lbl.config(text="⚠️ ไม่ได้เชื่อมต่อ Firebase — ไม่สามารถดึงรายชื่อผู้ใช้ได้")
                return
            try:
                users = self.firebase.list_users()
            except Exception as e:
                status_lbl.config(text=f"❌ ดึงรายชื่อไม่สำเร็จ: {e}")
                return
            if not users:
                status_lbl.config(text="ยังไม่มีผู้ใช้คนไหนบันทึกข้อมูลไว้เลย")
                return
            for u in users:
                var = tk.BooleanVar(value=False)
                row = tk.Frame(inner, bg=COLORS["card"])
                row.pack(fill="x", padx=8, pady=2)
                tk.Checkbutton(row, text=u, variable=var, bg=COLORS["card"], fg=COLORS["text"],
                              selectcolor=COLORS["surface"], activebackground=COLORS["card"],
                              activeforeground=COLORS["text"], font=FONT_TH, anchor="w",
                              command=_update_delete_btn_label).pack(fill="x")
                user_vars[u] = var
                user_rows[u] = row
            status_lbl.config(text=f"พบ {len(users)} ผู้ใช้ — ติ๊กเลือกแล้วกด 'ลบผู้ใช้ที่เลือก' ด้านล่าง")

        dialog.after(50, _load_users)

        def _on_dialog_close():
            cleanup()
            dialog.destroy()
        dialog.protocol("WM_DELETE_WINDOW", _on_dialog_close)

    def _save_calibration(self):
        uid = self.user_id_var.get().strip()
        if not uid:
            messagebox.showwarning("แจ้งเตือน", "กรุณาใส่รหัสผู้ใช้ก่อนบันทึก")
            return
        self._confirm_user_id()  # กด "บันทึก" ถือว่ายืนยันรหัสผู้ใช้นี้แล้วเช่นกัน
        self.firebase.start()  # กด "บันทึก" เอง = เจตนาชัดเจนว่าจะส่งขึ้น Firebase แล้ว
        n = len(self.calib_samples)
        if n < 20:
            messagebox.showwarning("แจ้งเตือน", f"Samples ยังน้อยเกินไป ({n} samples)\nกรุณาบันทึกสัญญาณให้มากกว่า 20 ครั้ง")
            return
        model = self.ml_base_var.get()  # ใช้โมเดลจริงที่กำลังใช้งานในแท็บ Machine Learning
        # เขียนลงไฟล์ .jsonl ในเครื่องทันที (offline-first) แล้วเธรดพื้นหลังจะซิงก์
        # ขึ้น Firebase ให้เองเมื่อออนไลน์ — ไม่ต้องรอเน็ตตอนกดบันทึก
        # samples ตอนนี้เก็บ rms, gesture, และ features (5 ค่า) ต่อ sample (แต่ก่อนมีแค่
        # rms ลอยๆ ไม่มี label เลยใช้เทรน ML ไม่ได้ — เพิ่ม gesture เข้าไปเพื่อรองรับแท็บ
        # Machine Learning แล้วต่อมาเพิ่ม feature อีก 4 ตัวเพื่อรองรับโมเดล multi-feature)
        # ไม่เอา key 'rms' จาก features มาทับ v — v คือค่า RMS ต่อ sample ที่ใช้เทียบ
        # threshold ทั่วทั้งแอปอยู่แล้ว ส่วน features['rms'] เป็นค่าจาก sliding window ซึ่ง
        # อาจต่างกันเล็กน้อย (คนละช่วงเวลา/คนละวิธีคำนวณ) ไม่อยากให้ทับกันโดยไม่ตั้งใจ
        labeled = [{"rms": v, "gesture": g,
                    **{k: val for k, val in (f or {}).items() if k != "rms"}}
                   for v, g, f in self.calib_samples[-500:]]
        self.firebase.log_calibration(uid, model, labeled,
                                       person_category=self._current_person_category)
        online_note = "จะซิงก์ขึ้น Firebase อัตโนมัติ" if self.firebase.online else "รอเน็ตแล้วจะซิงก์ขึ้น Firebase ให้อัตโนมัติ (ตอนนี้ Offline)"
        msg = f"💾 บันทึก {n} samples ของ [{uid}] ลงเครื่องแล้ว (โมเดล: {model}) — {online_note}"
        self._log(self.log_box, msg)
        self._set_status("บันทึก Calibration สำเร็จ")
        messagebox.showinfo("สำเร็จ", msg)

    def _clear_calibration(self):
        if messagebox.askyesno("ยืนยัน", "ต้องการล้างข้อมูล Calibration ทั้งหมดใช่หรือไม่?"):
            self.calib_samples.clear()
            self._log(self.log_box, "🗑 ล้างข้อมูล Calibration แล้ว")
            self._set_status("ล้างข้อมูลแล้ว")

    # ─── Dashboard Logic ──────────────────────────────────────────────────────
    def _toggle_grip(self):
        if not self.hardware_mode:
            current = self.stream._sim.contracting
            self.stream.set_simulated_contracting(not current)
            state = "กำมือ ✊" if not current else "คลายมือ 🖐"
            self._log(self.dash_activity_log, f"🔄 สลับ (โหมดจำลอง): {state}")
        else:
            self._log(self.dash_activity_log, "ℹ️ ต่อ Hardware จริงอยู่ — ขยับมือจริงเพื่อทดสอบแทนปุ่มนี้")

    # ─── Main Loop ────────────────────────────────────────────────────────────
    def _start_loop(self):
        self._tick()

    def _tick(self):
        # ~25 fps สำหรับ UI — การอ่านสัญญาณจริง (500Hz) และการบันทึกไฟล์/sync Firebase
        # ทำงานแบบ real-time ในเธรดของ EMGStreamThread แยกจาก loop วาดหน้าจอนี้แล้ว
        #
        # หมายเหตุ: self._current_user_id (ใช้โดย _on_emg_sample สำหรับบันทึกไฟล์/sync
        # ต่อเนื่อง) ไม่ได้อัพเดตจาก self.user_id_var ตรงๆ ในนี้แล้ว — เปลี่ยนเฉพาะตอนกด
        # Enter หรือคลิกออกจากช่องรหัสผู้ใช้ (ดู _confirm_user_id) กันไม่ให้แต่ละตัวอักษร
        # ที่พิมพ์ระหว่างทางถูกบันทึก/sync ขึ้น Firebase เป็นคนละ user แยกกัน

        with self._latest_lock:
            latest = dict(self._latest)
        emg = latest["rms"]
        stream_gesture = latest["gesture"]  # "FIST" / "OPEN" จากเธรดอ่านสัญญาณ

        self.emg_history.append(emg)
        if len(self.emg_history) > 200:
            self.emg_history.pop(0)

        # buffer แยกสำหรับกราฟ Dashboard (ยาวกว่า emg_history เพื่อรองรับหน้าต่างเวลา
        # 30 วิ – 5 นาที ตามที่เลือกใน dropdown "ช่วงเวลา") — deque(maxlen=) evict ให้
        # อัตโนมัติแบบ O(1) ตอน append เกิน maxlen ไม่ต้อง trim มือเองแบบเดิม (ซึ่งช้า)
        window_sec = self._dash_window_options.get(self.dash_window_var.get(), 120)
        max_pts = max(2, window_sec * 25)  # ~25 fps
        if max_pts != self._dash_window_maxlen:
            # ผู้ใช้เพิ่งเปลี่ยน dropdown ช่วงเวลา — สร้าง deque ใหม่ขนาดใหม่ (เกิดไม่บ่อย
            # ไม่กระทบ performance โดยรวม) เก็บข้อมูลเก่าไว้เท่าที่พอดีกับขนาดใหม่
            self._dash_window_maxlen = max_pts
            self.dash_chart_history = collections.deque(self.dash_chart_history, maxlen=max_pts)
        self.dash_chart_history.append(emg)

        self.current_emg.set(emg)

        # Calibration recording (เก็บค่า RMS พร้อม gesture label ที่กำลังบันทึกอยู่
        # ส่วนไฟล์ดิบทุก sample ถูกบันทึกแบบ real-time อยู่แล้วใน _on_emg_sample)
        # ระหว่าง "จำลองเก็บข้อมูลอัตโนมัติ" (self._auto_sim_running) จะไม่ append ตรงนี้ซ้ำ
        # เลยทั้งช่วง ไม่ใช่แค่ตอน self._auto_sim_collecting เป็น True — เพราะตอนนี้มีช่วง
        # warmup (_AUTO_SIM_LABEL_WARMUP_SEC) ที่ recording=True แต่ยังไม่เริ่มเก็บ sample จริง
        # (รอกล้ามเนื้อขยับตาม label ใหม่ก่อน) ถ้าเช็คแค่ _auto_sim_collecting จะมีช่องโหว่ให้
        # _tick ตรงนี้แอบ append sample ตาม UI framerate เข้าไปปนในช่วง warmup ซึ่งเป็นข้อมูล
        # แบบที่เราตั้งใจตัดทิ้งอยู่แล้วพอดี (ดู _run_auto_phase / _start_phase_sampling)
        if self.recording and not self._auto_sim_running:
            self.calib_samples.append((emg, self._current_calib_gesture, latest.get("features")))

        # Gesture classification (ใช้ threshold เดียวกับเธรดอ่านสัญญาณ) — แสดงผลตลอด
        # (ก่อนหน้านี้ต้องกด "เริ่มควบคุม" ก่อนถึงจะโชว์ แต่ตอนนี้ Dashboard เป็นแบบ
        # live monitoring เสมอ ไม่มีปุ่ม start/stop control แยกอีกต่อไป)
        gesture = "กำมือ ✊" if stream_gesture == "FIST" else "คลายมือ 🖐"
        self.gesture_var.set(gesture)

        fb_color = COLORS["success"] if self.firebase.online else COLORS["danger"]
        self.firebase_var.set("⬤  Online" if self.firebase.online else "⬤  Offline")
        if self.firebase.online:
            self._last_fb_online_ts = datetime.now().strftime("%H:%M:%S")

        self._draw_emg(self.calib_canvas, show_recording=True)
        self._update_calib_stats(emg)
        self._update_dashboard_live(emg, stream_gesture, latest.get("features"))
        # หมายเหตุ: ไม่มี "_ml_tick" อีกต่อไป — ML ไม่เทรนจากสัญญาณสดแล้ว
        # (ดู _load_ml_history / _load_ml_file แทน ซึ่งเป็นการเทรนแบบ batch ครั้งเดียว)

        self.after(40, self._tick)  # ~25 fps

    def _toggle_feature_panel(self):
        if self.show_features_var.get():
            self.feature_panel.pack(fill="x", pady=(4, 0))
        else:
            self.feature_panel.pack_forget()

    def _update_feature_panel(self, features):
        """อัพเดตตัวเลข feature ทั้ง 5 ตัวบน Dashboard — no-op ถ้าปิด panel ไว้ (ประหยัด
        การอัพเดต widget ที่มองไม่เห็นอยู่ดี) หรือยังไม่มี feature คำนวณเสร็จเลย (features
        เป็น None ตอน window ยังไม่เต็ม — โชว์ '—' ค้างไว้เหมือนตอนเริ่มโปรแกรม)"""
        if not self.show_features_var.get():
            return
        if features is None:
            return
        self._feature_value_lbls["rms"].config(text=f"{features['rms']:.4f}")
        self._feature_value_lbls["mav"].config(text=f"{features['mav']:.4f}")
        self._feature_value_lbls["variance"].config(text=f"{features['variance']:.6f}")
        self._feature_value_lbls["waveform_length"].config(text=f"{features['waveform_length']:.3f}")
        self._feature_value_lbls["zero_crossing"].config(text=f"{int(features['zero_crossing'])}")
        self._feature_value_lbls["slope_sign_change"].config(text=f"{int(features['slope_sign_change'])}")
        self._feature_value_lbls["iemg"].config(text=f"{features['iemg']:.4f}")

    def _update_dashboard_live(self, emg, stream_gesture, features):
        """อัพเดต widget ทั้งหมดในหน้า Dashboard ทุกเฟรม (~25fps) — แต่ส่วนที่ "หนัก"
        จริงๆ (วาดกราฟ/gauge ใหม่ทั้งอัน + เรียกโมเดลทำนาย) throttle ให้รันแค่ทุกๆ 3 เฟรม
        (~8 ครั้ง/วิ) แทน เพราะเป็นสาเหตุหลักที่ทำให้โปรแกรมหน่วง — ตาก็ยังดูลื่นอยู่ดี
        (มนุษย์แยกความต่างระหว่าง 8fps กับ 25fps ของกราฟ/gauge แทบไม่ออก) แต่ประหยัด CPU
        ไปได้เยอะมาก ส่วนตัวเลข/ข้อความที่ถูกและอัพเดตบ่อยได้ไม่กระทบอะไร (เช่น ค่า EMG,
        ท่าทาง) ยังอัพเดตทุกเฟรมเหมือนเดิมเพื่อความไว

        features: dict 7 ค่า (rms/mav/variance/waveform_length/zero_crossing/
        slope_sign_change/iemg) จาก sliding window ล่าสุด หรือ None ถ้ายังไม่มี window ไหน
        คำนวณเสร็จเลย — ใช้ทั้งแสดงผลใน panel Feature (ถ้าเปิดไว้ ดู self.show_features_var)
        และป้อนเข้าโมเดลทำนาย"""
        self._dash_tick_counter = getattr(self, "_dash_tick_counter", 0) + 1
        heavy_frame = (self._dash_tick_counter % 3 == 0)

        pct = min(100.0, max(0.0, emg / EMG_RANGE_V * 100))
        color = (COLORS["success"] if emg < self.threshold * 0.6
                 else COLORS["warning"] if emg < self.threshold
                 else COLORS["danger"])

        self.dash_emg_val_lbl.config(text=f"{emg:.3f} V", fg=color)
        if heavy_frame:
            self._draw_ring_gauge(self.dash_ring_canvas, pct)
            self._draw_semi_gauge(self.dash_gauge_canvas, emg, EMG_RANGE_V)
            self._draw_dash_chart()
            self._update_feature_panel(features)

        icon = "✊" if stream_gesture == "FIST" else "🖐"
        self.dash_gesture_icon_lbl.config(text=icon)
        self.dash_gesture_text_lbl.config(text=stream_gesture,
                                           fg=COLORS["danger"] if stream_gesture == "FIST"
                                           else COLORS["accent2"])
        if heavy_frame:
            pred, conf = (None, None)
            if features is not None:
                try:
                    pred, conf = self.trainer.predict_with_confidence(
                        features, self._current_person_category)
                except Exception:
                    pass
            if conf is not None:
                self.dash_gesture_conf_lbl.config(text=f"Confidence {conf:.1f}%")
            elif pred is not None:
                self.dash_gesture_conf_lbl.config(text="โมเดลนี้ไม่รองรับ confidence")
            elif features is None:
                # ปกติมากตอนเพิ่งเริ่มสตรีมสัญญาณ — sliding window ยังไม่ครบ (ดีฟอลต์
                # 0.2วิ) ยังไม่มี feature ให้ป้อนเข้าโมเดลได้เลย ไม่ใช่ error
                self.dash_gesture_conf_lbl.config(text="กำลังรอ window สัญญาณแรก...")
            else:
                self.dash_gesture_conf_lbl.config(text="ยังไม่มีโมเดล (ไปแท็บ Machine Learning)")

        self.dash_model_lbl.config(text=self.ml_base_var.get())
        ens = self.ml_ensemble_var.get()
        self.dash_model_ensemble_lbl.config(text=f"Ensemble: {ens}" if ens != "None" else "ไม่ใช้ Ensemble")

        self.dash_fb_status_lbl.config(text="Online" if self.firebase.online else "Offline",
                                        fg=COLORS["success"] if self.firebase.online else COLORS["danger"])
        self.dash_fb_sync_lbl.config(text=f"Sync ล่าสุด: {self._last_fb_online_ts}"
                                      if self.firebase.online else "ออฟไลน์อยู่")

        if heavy_frame:
            snap = self.trainer.get_snapshot()  # ล็อก + copy confusion matrix ทุกครั้ง ไม่ถูกด้วย
            self.dash_samples_lbl.config(text=f"{snap['n_samples']:,}")

            # เดิม error ตอนเทรน (เช่น 'AdaBoost + XGBoost' บางเวอร์ชันไลบรารีเข้ากันไม่ได้)
            # จะ print() ออก console เฉยๆ ผู้ใช้ที่ไม่ได้เปิด terminal คู่ไว้จะไม่รู้เลยว่าทำไม
            # โมเดลนี้ไม่มีผลลัพธ์ขึ้น — เทียบ n_errors ที่เห็นล่าสุดกับของใหม่ ถ้าเพิ่มขึ้น
            # แปลว่ามี error ใหม่เกิดขึ้นจริง เอาไป log ให้เห็นในหน้าจอทั้งแท็บ ML และ Dashboard
            if snap["n_errors"] > self._ml_last_seen_error_count:
                self._ml_last_seen_error_count = snap["n_errors"]
                err_text = (f"❌ เทรน [{self.trainer.display_name}] ไม่สำเร็จ (error ครั้งที่ "
                            f"{snap['n_errors']}): {snap['last_error']}")
                self._log(self.ml_log, err_text)
                self._log(self.dash_activity_log, err_text)
                if hasattr(self, "ml_status_lbl"):
                    self.ml_status_lbl.config(text=err_text, fg=COLORS["danger"])

            if self.event_recorder.enabled:
                if self.event_recorder.is_recording:
                    self.dash_event_status_lbl.config(text="🔴 กำลังบันทึก Event...", fg=COLORS["danger"])
                else:
                    self.dash_event_status_lbl.config(text="🟡 กำลังรอสัญญาณกำมือ...", fg=COLORS["warning"])

        # การ์ดผู้ใช้งาน + สถานะระบบใน sidebar
        self.sidebar_user_lbl.config(text=self._current_user_id if self._current_user_id else "—")
        self.sidebar_person_category_lbl.config(
            text=f"ประเภท: {PERSON_CATEGORY_LABELS.get(self._current_person_category, '—')}")
        if self._realtime_logging_active:
            self.sidebar_user_badge.config(text="Active", bg=COLORS["success"])
        else:
            self.sidebar_user_badge.config(text="ไม่ได้บันทึก", bg=COLORS["text_dim"])

        self.status_rows["emg"].config(
            text="เชื่อมต่อ" if self.hardware_mode else "โหมดจำลอง",
            fg=COLORS["success"] if self.hardware_mode else COLORS["warning"])
        self.status_rows["servo"].config(
            text="เชื่อมต่อ" if self.servo.hardware_mode else "โหมดจำลอง",
            fg=COLORS["success"] if self.servo.hardware_mode else COLORS["warning"])
        self.status_rows["firebase"].config(
            text="Online" if self.firebase.online else "Offline",
            fg=COLORS["success"] if self.firebase.online else COLORS["danger"])
        self.status_rows["system"].config(text="พร้อมใช้งาน", fg=COLORS["success"])

    def _draw_emg(self, canvas: tk.Canvas, show_recording: bool):
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10:
            return

        # เดิม pad=30 ทำให้ป้ายตัวเลขแกน Y (เช่น "0.50") ที่วาดแบบ anchor="e" ที่ x=pad-4=26
        # มีที่ว่างทางซ้ายไม่พอ ตัวอักษร "0" ตัวแรกเลยโดนตัดพ้นขอบซ้ายของ canvas ไปบางส่วน
        # (โดยเฉพาะถ้าฟอนต์ Courier New บนเครื่องที่รันจริงกว้างกว่าที่คาดไว้) — เพิ่ม pad
        # ให้มีระยะขอบซ้าย/ขวาเผื่อป้ายตัวเลขกว้างขึ้น กันไม่ให้ชิดขอบอีก
        pad = 44
        h_range = EMG_RANGE_V

        # Grid lines
        for i in range(5):
            y = pad + (H - 2*pad) * i / 4
            v = h_range * (1 - i/4)
            canvas.create_line(pad, y, W-pad, y,
                               fill=COLORS["border"], dash=(3, 4))
            canvas.create_text(pad-6, y, text=f"{v:.2f}",
                               fill=COLORS["text_dim"],
                               font=("Courier New", 8), anchor="e")

        # เส้น Threshold บน (activate/กำมือ) + ล่าง (release/พัก) + โซน noise ตรงกลาง
        ty_hi = pad + (H - 2*pad) * (1 - self.threshold / h_range)
        ty_lo = pad + (H - 2*pad) * (1 - self.release_threshold / h_range)
        canvas.create_rectangle(pad, ty_hi, W-pad, ty_lo,
                                fill=COLORS["card"], outline="", stipple="gray25")
        canvas.create_line(pad, ty_hi, W-pad, ty_hi,
                           fill=COLORS["threshold"], dash=(6, 3), width=2)
        canvas.create_text(W-pad+2, ty_hi, text=f"บน {self.threshold:.3f}V",
                           fill=COLORS["threshold"],
                           font=("Courier New", 8), anchor="w")
        canvas.create_line(pad, ty_lo, W-pad, ty_lo,
                           fill=COLORS["warning"], dash=(6, 3), width=2)
        canvas.create_text(W-pad+2, ty_lo, text=f"ล่าง {self.release_threshold:.3f}V",
                           fill=COLORS["warning"],
                           font=("Courier New", 8), anchor="w")

        # EMG waveform
        n = len(self.emg_history)
        if n < 2:
            return
        pts = []
        for i, v in enumerate(self.emg_history):
            x = pad + (W - 2*pad) * i / (n - 1)
            y = pad + (H - 2*pad) * (1 - v / h_range)
            pts.append((x, y))

        for i in range(len(pts)-1):
            canvas.create_line(pts[i][0], pts[i][1],
                               pts[i+1][0], pts[i+1][1],
                               fill=COLORS["emg_line"], width=2)

        # Recording indicator
        if show_recording and self.recording:
            canvas.create_oval(W-14, 8, W-4, 18, fill=COLORS["danger"], outline="")
            canvas.create_text(W-18, 13, text="REC",
                               fill=COLORS["danger"],
                               font=("Courier New", 9, "bold"), anchor="e")

        # Current value dot
        if pts:
            lx, ly = pts[-1]
            canvas.create_oval(lx-4, ly-4, lx+4, ly+4,
                               fill=COLORS["emg_line"], outline=COLORS["bg"])

    def _update_calib_stats(self, emg):
        self.calib_emg_lbl.config(text=f"{emg:.3f}")
        self.calib_count_lbl.config(text=str(len(self.calib_samples)))

    # ─── Machine Learning Logic ───────────────────────────────────────────────
    # หมายเหตุ: ไม่มีการเทรนจากสัญญาณสด (real-time) อีกต่อไป — ตามที่สรุปกันว่าค่า EMG
    # วิ่งเข้ามาตลอดเวลาทำให้เทรนถี่เกินไปและหนักเครื่อง เทรนได้เฉพาะแบบ batch จากข้อมูล
    # ย้อนหลังเท่านั้น (ดู _load_ml_history และ _load_ml_file ด้านล่าง)

    def _update_ml_model_note(self):
        base, ens = self.ml_base_var.get(), self.ml_ensemble_var.get()
        notes = ["✅ ใช้ของจริงล้วนๆ ไม่มีการประมาณ (scikit-learn"
                 + (" / XGBoost" if base == "XGBoost" else "")
                 + (" / CatBoost" if base == "CatBoost" else "") + ")"]
        if self.trainer.is_partial:
            notes.append("🟢 เทรนแบบ Online จริง (partial_fit ทุก sample)")
        else:
            notes.append(f"🔵 Full-refit ทุกๆ {self.trainer.retrain_every_n} sample "
                          f"(เทรนอยู่เบื้องหลัง ไม่บล็อกหน้าจอ)")
        self.ml_model_note_lbl.config(text="\n".join(notes))

    def _on_ml_model_change(self, event=None):
        base, ens = self.ml_base_var.get(), self.ml_ensemble_var.get()
        # k-NN ใช้กับ AdaBoost ไม่ได้จริงๆ — KNeighborsClassifier ของ scikit-learn ไม่รองรับ
        # sample_weight ที่ AdaBoost ต้องใช้เทรนภายใน (ทดสอบแล้วเทรนไม่สำเร็จ 100% ของเวลา)
        # กันไว้ตรงนี้เลยแทนที่จะปล่อยให้ผู้ใช้เลือกได้แล้วเทรนล้มเหลวแบบเงียบๆ
        if base == "k-NN" and ens == "AdaBoost":
            self.ml_ensemble_var.set("None")
            ens = "None"
            self._log(self.ml_log, "⚠️ k-NN ใช้กับ AdaBoost ไม่ได้ (ข้อจำกัดของ scikit-learn) "
                                    "— เปลี่ยน Ensemble กลับเป็น None ให้อัตโนมัติ")
        old_trainer = self.trainer
        self.trainer = OnlineTrainer(base, ens)
        self._ml_last_seen_error_count = 0  # trainer ใหม่ — รีเซ็ตตัวนับ error ที่เคย log ไปแล้ว
        old_trainer.stop()  # ปิดเธรดพื้นหลังของโมเดลเก่าทิ้ง กันเธรดค้าง
        self._update_ml_model_note()
        self._log(self.ml_log, f"🔁 เปลี่ยนเป็น {self.trainer.display_name} — โมเดลว่างเปล่า "
                                f"กดโหลดข้อมูลย้อนหลังหรืออัพโหลดไฟล์เพื่อเทรนใหม่")
        self._refresh_ml_metrics()

    def _reset_ml_model(self):
        old_trainer = self.trainer
        self.trainer = OnlineTrainer(self.ml_base_var.get(), self.ml_ensemble_var.get())
        self._ml_last_seen_error_count = 0  # trainer ใหม่ — รีเซ็ตตัวนับ error ที่เคย log ไปแล้ว
        old_trainer.stop()
        self._log(self.ml_log, "🗑 รีเซ็ตโมเดลแล้ว")
        self._refresh_ml_metrics()

    # ─── เทรนทุกโมเดลรวดเดียว (Base Model ทุกตัว x Ensemble Method ทุกแบบ) ────────────
    def _train_all_models(self):
        """เทรนทุกชุดค่าผสมของ Base Model x Ensemble Method โดยอัตโนมัติทีละคู่ (รวม Base
        Model เดี่ยวๆ ที่ไม่มี Ensemble ด้วย เพราะ ENSEMBLE_METHODS มี 'None' รวมอยู่แล้ว)
        กับ dataset ล่าสุดที่โหลดไว้ (self._ml_last_loaded_samples — โหลดผ่าน 'โหลดข้อมูล
        ย้อนหลัง & เทรน' หรืออัพโหลดไฟล์ JSON มาก่อนหน้านี้แล้ว) แทนที่จะต้องมาเปลี่ยน
        dropdown Base Model/Ensemble Method แล้วกดโหลด/เทรนเองทีละคู่ (ประหยัดเวลามาก ถ้ามี
        Base Model 9 ตัว x Ensemble 3 แบบ = 27 คู่)

        ประมวลผลทีละคู่ต่อเนื่องกันแบบไม่บล็อกหน้าจอ (ผ่าน after) — แต่ละคู่ยังใช้กลไก
        เดียวกับการเทรนปกติทุกอย่าง (OnlineTrainer ใหม่ + add_sample ทั้ง dataset +
        force_retrain_now + รอเทรนรอบสุดท้ายเสร็จ) ผลลัพธ์แต่ละคู่จึงถูกบันทึกลง
        self.model_results อัตโนมัติผ่าน _refresh_ml_metrics เหมือนเทรนทีละตัวปกติ — เทรน
        ครบแล้วไปดูเทียบกันได้ทันทีที่หน้า 'สรุปผลโมเดล'"""
        if getattr(self, "_ml_train_all_running", False):
            messagebox.showinfo("กำลังเทรนอยู่", "กำลังเทรนทุกโมเดลอยู่ — กรุณารอให้เสร็จก่อน")
            return

        samples = self._ml_last_loaded_samples
        if not samples:
            messagebox.showwarning(
                "ยังไม่มีข้อมูล",
                "กรุณากด 'โหลดข้อมูลย้อนหลัง & เทรน' หรืออัพโหลดไฟล์ JSON ก่อน "
                "ถึงจะมี dataset ให้เทรนทุกโมเดลพร้อมกันได้")
            return

        combos = [(base, ens) for base in BASE_MODELS for ens in ENSEMBLE_METHODS]
        if not messagebox.askyesno(
                "เทรนทุกโมเดล",
                f"จะเทรน Base Model ทั้งหมด {len(BASE_MODELS)} ตัว x Ensemble Method "
                f"{len(ENSEMBLE_METHODS)} แบบ (รวม Base เดี่ยวๆ ด้วย) = {len(combos)} โมเดล "
                f"กับ dataset {len(samples)} samples ที่โหลดไว้ล่าสุด\n\n"
                f"อาจใช้เวลาสักพัก (โมเดลหนักอย่าง SVM/ANN/XGBoost/CatBoost จะช้ากว่าโมเดล"
                f"อื่น) — ต้องการเริ่มเลยไหม?"):
            return

        self._ml_train_all_queue = combos
        self._ml_train_all_total = len(combos)
        self._ml_train_all_done = 0
        self._ml_train_all_running = True
        self.btn_ml_train_all.config(state="disabled", text="⏳  กำลังเทรนทุกโมเดล...")
        self._log(self.ml_log, f"🚀 เริ่มเทรนทุกโมเดล — {len(BASE_MODELS)} Base Model x "
                                f"{len(ENSEMBLE_METHODS)} Ensemble Method = {len(combos)} ชุด "
                                f"(ใช้ dataset {len(samples)} samples ที่โหลดไว้ล่าสุด)")
        self._train_all_step()

    def _train_all_step(self):
        """ประมวลผลคิว self._ml_train_all_queue ทีละคู่ (base, ensemble) — เรียกตัวเองซ้ำ
        ผ่าน after หลังแต่ละคู่เทรนเสร็จ จนคิวหมด"""
        if not self._ml_train_all_queue:
            self._ml_train_all_running = False
            self.btn_ml_train_all.config(state="normal", text="🚀  เทรนทุกโมเดล (Base + Ensemble)")
            self._log(self.ml_log, f"✅ เทรนครบทุกโมเดลแล้ว ({self._ml_train_all_done}/"
                                    f"{self._ml_train_all_total}) — ดูผลเปรียบเทียบได้ที่หน้า "
                                    f"'สรุปผลโมเดล'")
            messagebox.showinfo("เทรนครบแล้ว",
                                 f"เทรนครบทุกโมเดลแล้ว {self._ml_train_all_done} ชุด\n"
                                 f"ไปดูผลเปรียบเทียบที่หน้า 'สรุปผลโมเดล' ได้เลย")
            return

        base, ens = self._ml_train_all_queue.pop(0)
        self._ml_train_all_done += 1

        old_trainer = self.trainer
        self.trainer = OnlineTrainer(base, ens)
        self._ml_last_seen_error_count = 0  # trainer ใหม่ — รีเซ็ตตัวนับ error ที่เคย log ไปแล้ว
        old_trainer.stop()  # ปิดเธรดพื้นหลังของโมเดลก่อนหน้าทิ้ง กันเธรดค้าง

        # ให้ dropdown บนจอ + caption "เปลี่ยนแล้วรีเซ็ตโมเดล" สะท้อนโมเดลที่กำลังเทรนอยู่จริง
        self.ml_base_var.set(base)
        self.ml_ensemble_var.set(ens)
        self._update_ml_model_note()
        label = self.trainer.display_name
        self.ml_status_lbl.config(
            text=f"🚀 กำลังเทรนทุกโมเดล — [{label}] "
                 f"({self._ml_train_all_done}/{self._ml_train_all_total})")
        self._log(self.ml_log, f"🔁 [{self._ml_train_all_done}/{self._ml_train_all_total}] "
                                f"กำลังเทรน: {label}")

        samples = self._ml_last_loaded_samples
        for features, gesture, person_category in samples:
            self.trainer.add_sample(features, gesture, person_category)
        # ⚠️ บั๊กเดิม: capture n_retrains_before ตรงนี้แล้วรอแค่ "มีรอบเทรนใหม่เกิดขึ้น" อย่าง
        # เดียวไม่พอ — background thread ของ trainer อาจเทรนกลางคันไปแล้วด้วยข้อมูลแค่บางส่วน
        # (ทุกครั้งที่สะสมครบ retrain_every_n samples ระหว่าง loop ป้อนข้อมูลด้านบน) รอบที่
        # เพิ่งเสร็จพอดีตอนเราเช็คอาจเป็นรอบที่ใช้ข้อมูล "ไม่ครบ" ที่เพิ่งเทรนค้างอยู่ก่อนหน้า
        # ไม่ใช่รอบใหม่ที่ใช้ข้อมูลครบทั้งหมดจริงๆ — ทำให้ metric ที่อ่านได้ขึ้นกับจังหวะ
        # thread ในแต่ละครั้งที่รัน (นี่คือสาเหตุจริงที่ผลเทรนไม่นิ่งแม้ข้อมูล/seed เดิมทุก
        # อย่าง) แก้โดยจด "จำนวน sample ที่ควรเทรนครบ" ไว้ก่อน แล้วรอจนกว่าจะเจอรอบที่ history
        # ล่าสุดบันทึกว่าเทรนด้วย sample ครบตามนี้จริงๆ (ดู _wait_train_all_retrain)
        expected_min_n_train = self.trainer.get_snapshot()["split_summary"]["n_train"]
        n_retrains_before = self.trainer.get_snapshot()["n_retrains"]
        self.trainer.force_retrain_now()
        self._wait_train_all_retrain(n_retrains_before, expected_min_n_train, time.time() + 20.0)

    def _wait_train_all_retrain(self, n_retrains_before, expected_min_n_train, deadline):
        """เหมือน _wait_final_retrain แต่รีเฟรชผลลัพธ์แล้วต่อคู่ถัดไปในคิวเทรนทุกโมเดล
        (deadline นานกว่าปกติ 20 วิ เผื่อโมเดลหนักอย่าง SVM/ANN/XGBoost/CatBoost)

        เงื่อนไขที่ต้องครบทั้ง 2 ข้อก่อนถือว่า 'เทรนรอบสุดท้ายเสร็จจริง' (กัน race กับรอบที่
        เทรนค้างด้วยข้อมูลไม่ครบ ดู comment ใน _train_all_step):
          1. n_retrains เพิ่มขึ้นจากตอนก่อนสั่ง force_retrain_now
          2. รอบล่าสุดใน history เทรนด้วย sample จำนวน >= expected_min_n_train (ครบจริง)
        ถ้าเลย deadline แล้วยังไม่ครบเงื่อนไข จะ log เตือนไว้ว่าอาจได้ผลจากรอบที่ข้อมูลไม่ครบ
        (กันสถานการณ์สุดโต่งที่ retrain ช้าผิดปกติจนไม่จบใน deadline ไม่ให้ค้างทั้งคิวไปเลย)"""
        snap = self.trainer.get_snapshot()
        history = snap["history"]
        last_n_train = history[-1]["n_train"] if history else 0
        complete = snap["n_retrains"] > n_retrains_before and last_n_train >= expected_min_n_train
        if complete:
            self._refresh_ml_metrics()  # บันทึกผลลง self.model_results ให้โมเดลนี้
            self.after(30, self._train_all_step)  # ต่อคู่ถัดไปในคิว
            return
        if time.time() > deadline:
            self._log(self.ml_log, f"⚠️ [{self.trainer.display_name}] เทรนไม่ครบ dataset ทัน "
                                    f"ภายในเวลาที่กำหนด (เทรนด้วย {last_n_train}/"
                                    f"{expected_min_n_train} samples) — ผลลัพธ์อาจไม่นิ่ง 100% "
                                    f"ลองกด 'เทรนทุกโมเดล' ซ้ำถ้าอยากได้ผลแม่นขึ้น")
            self._refresh_ml_metrics()
            self.after(30, self._train_all_step)
            return
        self.after(100, lambda: self._wait_train_all_retrain(
            n_retrains_before, expected_min_n_train, deadline))

    # Feature นอกจาก rms ที่ record ต้องมีครบถึงจะใช้กับโมเดล multi-feature ใหม่ได้ — ดึงจาก
    # emg_features.FEATURE_NAMES ตรงๆ (ตัด 'rms' ออกเพราะเช็คแยกอยู่แล้ว) แทนที่จะ hardcode
    # ชื่อ feature ซ้ำไว้ที่นี่อีกที่ กันลืมอัพเดตจุดนี้เวลาเพิ่ม/ลด feature ในอนาคต
    _REQUIRED_EXTRA_FEATURES = tuple(name for name in FEATURE_NAMES if name != "rms")

    def _record_to_sample(self, record: dict):
        """แปลง record 1 ตัว (dict จาก Firebase/ไฟล์ export) เป็น
        (features_dict, gesture, person_category)

        คืนค่า (None, None, None, True) ถ้าเป็น record 'เก่า' ที่มีแค่ rms/gesture (ไม่มี
        feature อีก 6 ตัวครบ) — ธง is_legacy=True บอกผู้เรียกว่าควรนับแยกไว้แจ้งเตือนผู้ใช้
        (ข้อกำหนดเรื่อง Backward Compatibility: dataset เก่าที่มีแค่ RMS ใช้กับโมเดลใหม่ไม่ได้
        ต้องแจ้งเตือน ไม่ใช่เติม 0 ปลอมๆ เข้าไปเทรนปนกับข้อมูลจริง เพราะจะทำให้โมเดลเรียนรู้ผิด
        ว่า 'ไม่มีข้อมูล' กับ 'ค่าจริงเป็น 0' คือเรื่องเดียวกัน)

        คืนค่า (None, None, None, False) ถ้า record ไม่สมบูรณ์แบบอื่น (ไม่มี rms/gesture เลย
        ไม่ใช่กรณี legacy — เช่น record type อื่นที่หลุดเข้ามาปนโดยไม่ตั้งใจ)

        person_category: ต่างจาก 6 feature ข้างบนตรงที่ "ไม่บังคับ" — record เก่าก่อนมี
        ฟีเจอร์นี้ (หรือ record จากภายนอกที่ไม่มี field นี้เลย) จะได้ค่า fallback เป็น
        DEFAULT_PERSON_CATEGORY แทนที่จะถูกนับเป็น legacy/ข้ามไป เพราะ person_category เป็น
        "บริบทเพิ่มเติม" ไม่ใช่ feature จากสัญญาณ EMG ที่ถ้าขาดแล้วความหมายจะเพี้ยนแบบ
        rms/mav/... ที่ขาดไม่ได้เลย"""
        if not isinstance(record, dict) or "rms" not in record or "gesture" not in record:
            return None, None, None, False
        if not all(record.get(k) is not None for k in self._REQUIRED_EXTRA_FEATURES):
            return None, None, None, True  # legacy: มี rms/gesture แต่ขาด feature อื่นบางตัว
        features = {name: record[name] for name in FEATURE_NAMES}
        person_category = record.get("person_category") or DEFAULT_PERSON_CATEGORY
        if person_category not in PERSON_CATEGORIES:
            person_category = DEFAULT_PERSON_CATEGORY
        return features, record["gesture"], person_category, False

    def _collect_rms_gesture_records(self, node, found=None):
        """เดินเข้าไปในโครงสร้าง JSON แบบไหนก็ได้ (dict/list ซ้อนกันกี่ชั้นก็ได้) หา dict
        ที่มีทั้ง 'rms' และ 'gesture' อยู่ในนั้น เก็บ record ทั้งก้อน (dict) ไว้ — ไม่ใช่แค่
        (rms, gesture) เหมือนเดิม เพื่อให้ผู้เรียกตรวจสอบทีหลังได้ว่า record นั้นมี feature
        ครบ 5 ตัว (multi-feature ใหม่) หรือมีแค่ rms ตัวเดียว (dataset เก่า) — ดู _record_to_sample

        ใช้กับไฟล์ export ตรงจาก Firebase Console ("⋮ → Export JSON") ที่โครงสร้างเป็น
        {push_key: record, ...} หรือถ้า export จาก root จะซ้อนลึกกว่านั้นอีกชั้น
        {"sessions": {"user_id": {push_key: record, ...}}} — เดินหาได้ไม่ว่าจะซ้อนกี่ชั้น"""
        if found is None:
            found = []
        if isinstance(node, dict):
            if "rms" in node and "gesture" in node:
                found.append(node)
            else:
                for v in node.values():
                    self._collect_rms_gesture_records(v, found)
        elif isinstance(node, list):
            for item in node:
                self._collect_rms_gesture_records(item, found)
        return found

    def _parse_ml_json_file(self, path):
        """อ่านไฟล์ JSON แบบยืดหยุ่น รองรับ 4 รูปแบบ:
          1) JSON array ปกติ:        [{"rms": 0.03, "gesture": "OPEN", "mav":.., ...}, ...]
          2) JSON object จาก Firebase Console Export: {"-pushKey": {"rms":.., "gesture":..}, ...}
             (ซ้อนลึกกี่ชั้นก็หาเจอ เช่น export จาก root ที่มีหลาย user)
          3) JSONL แบบ flat:          {"rms": 0.03, "gesture": "OPEN", ...}\\n (บรรทัดละ 1 record)
          4) JSONL แบบ Calibration เดิม: {"samples": [{"rms":.., "gesture":..}, ...], ...}
        คืนค่า (samples: list[(features_dict, gesture, person_category)], skipped: int,
        legacy_skipped: int)
        — legacy_skipped คือจำนวน record ที่มีแค่ rms/gesture ไม่มี feature อีก 4 ตัว (dataset
        เก่าก่อนอัพเดตระบบนี้) ใช้กับโมเดล multi-feature ใหม่ไม่ได้ นับแยกจาก skipped ทั่วไป
        (record เสียรูปแบบ/ไม่มี rms หรือ gesture เลย) เพื่อแจ้งเตือนผู้ใช้ได้ตรงประเด็นกว่า
        person_category ที่ได้กลับมาไม่นับเป็น legacy ถ้าขาด (fallback เป็น DEFAULT_PERSON_
        CATEGORY แทน — ดู _record_to_sample)"""
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()

        samples, skipped, legacy_skipped = [], 0, 0

        if raw.startswith("[") or raw.startswith("{"):
            # รูปแบบที่ 1/2: ไฟล์ทั้งก้อนเป็น JSON เดียว (array หรือ object) — ใช้ตัวเดิน
            # โครงสร้างแบบ recursive หา record ที่มี rms+gesture ไม่ว่าจะซ้อนอยู่ชั้นไหน
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"ไฟล์ JSON เสียรูปแบบ: {e}")
            for record in self._collect_rms_gesture_records(data):
                features, gesture, person_category, is_legacy = self._record_to_sample(record)
                if features is not None:
                    samples.append((features, gesture, person_category))
                elif is_legacy:
                    legacy_skipped += 1
                else:
                    skipped += 1
        else:
            # รูปแบบที่ 3/4: JSONL บรรทัดละ 1 record
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                items = record["samples"] if (isinstance(record, dict) and "samples" in record) \
                    else [record]
                for item in items:
                    features, gesture, person_category, is_legacy = self._record_to_sample(item)
                    if features is not None:
                        samples.append((features, gesture, person_category))
                    elif is_legacy:
                        legacy_skipped += 1
                    else:
                        skipped += 1

        return samples, skipped, legacy_skipped

    def _on_ml_user_mode_change(self):
        specific = self.ml_user_mode_var.get() == "specific"
        self.btn_ml_select_users.config(state="normal" if specific else "disabled")

    def _fetch_firebase_samples(self, user_id: str, cutoff_iso: str):
        """ดึง (features_dict, gesture, person_category) ของ user คนเดียวจาก Firebase
        ย้อนหลังตาม cutoff_iso
        คืนค่า (samples, skipped, legacy_skipped) — โยน exception ต่อถ้า query ล้มเหลวด้วย
        สาเหตุอื่นที่ไม่ใช่ index (ให้ผู้เรียกจัดการ)

        legacy_skipped: จำนวน record ที่มีแค่ rms/gesture (บันทึกไว้ก่อนอัพเดตระบบนี้ให้รองรับ
        multi-feature) ไม่มี feature อื่นครบทั้ง 6 ตัว — ใช้กับโมเดล
        multi-feature ใหม่ไม่ได้ นับแยกไว้ให้ผู้เรียกแจ้งเตือนผู้ใช้ (ข้อกำหนดเรื่อง
        Backward Compatibility) แทนที่จะเงียบๆ ข้ามไปแบบไม่บอกอะไรเลย

        Firebase Realtime Database ต้องมี .indexOn ที่ field 'timestamp' ถึงจะใช้
        order_by_child('timestamp') แบบเร็ว/ประหยัด bandwidth ได้ — ถ้ายังไม่ได้ตั้งค่า
        ใน Firebase Console → Rules จะได้ error 'Index not defined' กลับมา แทนที่จะ
        บังคับให้ผู้ใช้ไปตั้งค่า Console ก่อนถึงจะใช้ฟีเจอร์นี้ได้ ให้ fallback มาดึงข้อมูล
        ทั้งหมดของ user คนนั้นแล้วกรองช่วงเวลาเอาเองฝั่ง client แทน (ช้ากว่า/หนักกว่าถ้า
        user มีข้อมูลสะสมเยอะมาก แต่ยังใช้งานได้ทันทีโดยไม่ต้องรอตั้งค่า Firebase ก่อน)"""
        try:
            result = (self.firebase.db.child("sessions").child(user_id)
                      .order_by_child("timestamp").start_at(cutoff_iso).get())
            indexed = True
        except Exception as e:
            if "index not defined" not in str(e).lower():
                raise  # error อื่นที่ไม่เกี่ยวกับ index — โยนต่อให้ผู้เรียกจัดการตามเดิม
            self._log(self.ml_log, f"⚠️ ยังไม่ได้ตั้ง index 'timestamp' ให้ '{user_id}' บน "
                                    f"Firebase — ดึงข้อมูลทั้งหมดมากรองเองแทน (ช้ากว่าปกติ)")
            result = self.firebase.db.child("sessions").child(user_id).get()
            indexed = False

        samples, skipped, legacy_skipped = [], 0, 0
        for item in (result.each() or []):
            record = item.val()
            if not isinstance(record, dict):
                skipped += 1
                continue
            if record.get("type") != "sample":
                continue
            # ถ้าไม่ได้ผ่าน indexed query มา (server ยังไม่ได้กรองช่วงเวลาให้) ต้องกรอง
            # timestamp เองตรงนี้ — ถ้า indexed แล้วก็กรองซ้ำได้ไม่เสียหาย (ผ่านหมดอยู่แล้ว)
            if not indexed and record.get("timestamp", "") < cutoff_iso:
                continue
            features, gesture, person_category, is_legacy = self._record_to_sample(record)
            if features is not None:
                samples.append((features, gesture, person_category))
            elif is_legacy:
                legacy_skipped += 1
            else:
                skipped += 1
        return samples, skipped, legacy_skipped

    def _load_ml_history(self):
        """โหลดข้อมูลย้อนหลัง N เดือน (ตามที่เลือกใน self.ml_range_var) จาก Firebase
        Realtime Database โดยตรง (node sessions/<user_id>) แทนไฟล์ในเครื่อง — ข้อดีคือ
        ดึงได้แม้ข้อมูลนั้นถูก sync ขึ้นมาจากเครื่องอื่น (เช่น Raspberry Pi หลายตัว sync
        รวมกันไว้บน Firebase) ไม่ใช่แค่ข้อมูลของเครื่องที่กำลังรัน GUI นี้อยู่ตอนนี้เท่านั้น
        แลกกับต้องรอโหลดผ่านเน็ต (ช้ากว่าไฟล์ในเครื่องที่เดิมใช้อยู่)

        เลือก user ได้ 3 แบบ (ดู self.ml_user_mode_var — แยกจาก self.user_id_var ของ
        Dashboard โดยตั้งใจ): 'specific' ดึงของ user ที่ติ๊กเลือกไว้ (เลือกได้หลายคนพร้อมกัน
        ผ่าน _open_ml_user_select_dialog — ดู self.ml_selected_users), 'all' ดึงของ
        ทุก user ที่เคยมีข้อมูลบน Firebase มารวมกันเป็น dataset เดียว (ไม่สนว่าจะมี sample
        ที่ label แล้วจริงหรือเปล่า) หรือ 'all_labeled' ("ทุก user (Label)") ที่เหมือน 'all'
        ทุกอย่างแต่กรองออกหลังดึงข้อมูลแต่ละคนมาแล้ว — เก็บเฉพาะ user ที่มี sample ซึ่ง label
        (gesture) แล้วจริงอย่างน้อย 1 ตัวในช่วงย้อนหลังที่เลือก คนที่ไม่มีข้อมูล label เลย
        (0 samples) จะถูกข้ามไปเงียบๆ ไม่นับรวมเข้า dataset/normalize"""
        months = int(self.ml_range_var.get())

        if not (self.firebase.online and self.firebase.db is not None):
            messagebox.showwarning(
                "ไม่ได้เชื่อมต่อ Firebase",
                "ต้องเชื่อมต่อ Firebase ก่อนถึงจะโหลดข้อมูลย้อนหลังได้ (ดูสถานะที่หัวข้อ "
                "Firebase) — ลองเช็คอินเทอร์เน็ต หรือ FIREBASE_CONFIG ในไฟล์ firebase_sync.py")
            return

        user_mode = self.ml_user_mode_var.get()
        filter_labeled_only = user_mode == "all_labeled"
        if user_mode in ("all", "all_labeled"):
            user_ids = self.firebase.list_users()
            if not user_ids:
                messagebox.showwarning("ไม่พบ user", "ยังไม่มี user คนไหนบันทึกข้อมูลไว้บน Firebase เลย")
                return
            # label ตอนนี้ยังเป็นแค่ข้อความ "กำลังตรวจสอบ" — ถ้าเป็น all_labeled จะแทนที่ด้วย
            # จำนวนจริงหลังกรองเสร็จ (ดูหลัง for loop ด้านล่าง) เพราะยังไม่รู้ล่วงหน้าว่ากี่คน
            # ที่มี label จริงจนกว่าจะดึงข้อมูลมาเช็คทีละคนก่อน
            label = (f"ทุก user ที่มี Label (กำลังตรวจสอบ {len(user_ids)} คน)"
                     if filter_labeled_only else f"ทุก user ({len(user_ids)} คน)")
        else:
            user_ids = list(self.ml_selected_users)
            if not user_ids:
                messagebox.showwarning("แจ้งเตือน",
                                        "กรุณากดปุ่ม 'เลือกผู้ใช้...' แล้วติ๊กเลือกอย่างน้อย 1 คนก่อน")
                return
            label = (user_ids[0] if len(user_ids) == 1
                     else f"{len(user_ids)} user ที่เลือก ({', '.join(user_ids)})")

        cutoff_iso = (datetime.now() - timedelta(days=30 * months)).isoformat()
        self._log(self.ml_log, f"📅 กำลังดึงข้อมูลย้อนหลัง {months} เดือน ของ {label} จาก Firebase...")
        self._set_status(f"กำลังดึงข้อมูลจาก Firebase — {label}")
        self.update_idletasks()  # โชว์ log/status ทันทีก่อนเริ่ม network call ที่อาจใช้เวลา

        do_normalize = self.ml_normalize_var.get() and len(user_ids) > 1
        if self.ml_normalize_var.get() and len(user_ids) == 1:
            self._log(self.ml_log, "ℹ️ เลือกแค่ 1 user — ข้ามการ normalize (มีผลเฉพาะตอนรวม "
                                    "หลาย user เท่านั้น) ใช้ค่า RMS ดิบตามเดิม")

        samples, skipped, legacy_skipped = [], 0, 0
        labeled_user_ids, unlabeled_user_ids = [], []  # ใช้เฉพาะตอน filter_labeled_only
        for i, uid in enumerate(user_ids):
            try:
                u_samples, u_skipped, u_legacy = self._fetch_firebase_samples(uid, cutoff_iso)
            except Exception as e:
                messagebox.showerror(
                    "ดึงข้อมูลไม่สำเร็จ",
                    f"ดึงข้อมูลย้อนหลังของ '{uid}' จาก Firebase ไม่สำเร็จ:\n{e}\n\n"
                    f"ถ้า error พูดถึง 'index not defined' ให้ไปเพิ่ม index ที่ field "
                    f"'timestamp' ใต้ sessions/{uid} ใน Firebase Console → Realtime "
                    f"Database → Rules")
                self._log(self.ml_log, f"❌ ดึงข้อมูลของ '{uid}' ไม่สำเร็จ: {e}")
                return
            # โหมด "ทุก user (Label)": user คนนี้ยังไม่มี sample ที่ label (gesture) แล้วเลย
            # สักตัวในช่วงย้อนหลังที่เลือก — ข้ามไปทั้งคน ไม่นับรวมเข้า dataset/normalize/
            # skipped-count เพราะถือว่า "ไม่มีข้อมูลให้ดึง" ไม่ใช่ "ดึงมาแล้วเสีย"
            if filter_labeled_only and not u_samples:
                unlabeled_user_ids.append(uid)
                if len(user_ids) > 1:
                    self._set_status(f"กำลังตรวจสอบ Label จาก Firebase — {i+1}/{len(user_ids)} user")
                    self.update_idletasks()
                continue
            if filter_labeled_only:
                labeled_user_ids.append(uid)
            if do_normalize and u_samples:
                u_samples, stats = self._normalize_user_features(u_samples)
                stats_txt = ", ".join(f"{k}: mean={v[0]:.5g} std={v[1]:.5g}" for k, v in stats.items())
                self._log(self.ml_log, f"   ↳ [{uid}] {len(u_samples)} samples — normalize ({stats_txt})")
            samples.extend(u_samples)
            skipped += u_skipped
            legacy_skipped += u_legacy
            if len(user_ids) > 1:
                self._set_status(f"กำลังดึงข้อมูลจาก Firebase — {i+1}/{len(user_ids)} user")
                self.update_idletasks()

        if filter_labeled_only:
            # ตอนนี้รู้แล้วว่ากี่คนที่มี label จริง — แทนที่ label (ข้อความ "กำลังตรวจสอบ")
            # ด้วยจำนวนจริง และ log รายชื่อคนที่ถูกข้ามให้เห็นชัดว่าทำไม dataset ถึงไม่รวมเขา
            label = f"ทุก user ที่มี Label ({len(labeled_user_ids)}/{len(user_ids)} คน)"
            if unlabeled_user_ids:
                self._log(self.ml_log,
                          f"ℹ️ ข้าม {len(unlabeled_user_ids)} user ที่ยังไม่มีข้อมูล label "
                          f"ในช่วง {months} เดือนย้อนหลัง: {', '.join(unlabeled_user_ids)}")

        if legacy_skipped:
            # ข้อกำหนดเรื่อง Backward Compatibility: dataset เก่าที่มีแค่ RMS (บันทึกไว้ก่อน
            # อัพเดตระบบให้รองรับ multi-feature) ใช้กับโมเดลใหม่ไม่ได้ — ต้องแจ้งเตือนผู้ใช้
            # ชัดเจน ไม่ใช่เงียบๆ ข้ามไปแล้วให้ dataset ดูเหมือนมีข้อมูลน้อยกว่าที่ควรโดยไม่รู้สาเหตุ
            messagebox.showwarning(
                "พบข้อมูลรูปแบบเก่า",
                f"พบ {legacy_skipped} samples ที่บันทึกไว้ก่อนระบบอัพเดตเป็น multi-feature "
                f"(มีแค่ RMS ไม่มี MAV/Variance/Waveform Length/Zero Crossing/Slope Sign "
                f"Change/IEMG ครบ) — ข้อมูลกลุ่มนี้"
                f"ใช้กับโมเดลใหม่ไม่ได้ จึงถูกข้ามไปโดยอัตโนมัติ ไม่ได้เอามาเทรนด้วย\n\n"
                f"ถ้าต้องการใช้ข้อมูลกลุ่มนี้ ต้องเก็บสัญญาณใหม่ผ่านระบบเวอร์ชันนี้ (ที่คำนวณ "
                f"feature ครบ 7 ตัวให้อัตโนมัติอยู่แล้ว)")
            self._log(self.ml_log, f"⚠️ ข้าม {legacy_skipped} samples รูปแบบเก่า (มีแค่ RMS) "
                                    f"— ใช้กับโมเดล multi-feature ใหม่ไม่ได้")

        if not samples:
            messagebox.showwarning(
                "ไม่มีข้อมูลในช่วงที่เลือก",
                f"ไม่พบข้อมูลของ {label} บน Firebase ย้อนหลัง {months} เดือนเลย\n"
                "ลองเลือกช่วงที่ยาวขึ้น หรือกด 'เริ่มบันทึกแบบ Real-time' เพื่อสะสมข้อมูล "
                "ขึ้น Firebase ก่อน")
            return

        old_trainer = self.trainer
        self.trainer = OnlineTrainer(self.ml_base_var.get(), self.ml_ensemble_var.get())
        self._ml_last_seen_error_count = 0  # trainer ใหม่ — รีเซ็ตตัวนับ error ที่เคย log ไปแล้ว
        old_trainer.stop()
        self._ml_replay_queue = samples
        self._ml_replay_total = len(samples)
        self._ml_last_loaded_samples = list(samples)
        note = f" (ข้าม {skipped} record ที่อ่านไม่ได้/ไม่มี label)" if skipped else ""
        norm_note = " (normalize feature ต่อคนแล้ว)" if do_normalize else ""
        self._log(self.ml_log, f"📅 โหลด {len(samples)} samples ย้อนหลัง {months} เดือน "
                                f"ของ {label} จาก Firebase{note}{norm_note} — เริ่มเทรนแบบ Offline")
        if do_normalize:
            self._log(self.ml_log,
                      "⚠️ โมเดลนี้เทรนด้วย feature ที่ normalize แล้ว ไม่ใช่สเกลดิบเหมือนที่ Dashboard "
                      "ใช้ทำนายสด — เหมาะสำหรับดูภาพรวม/เปรียบเทียบโมเดลข้ามหลายคน ถ้าจะเอาไปใช้"
                      "ทำนายสดกับ user คนใดคนหนึ่งจริงจัง แนะนำเทรนแบบเลือก user รายคนเดียว "
                      "(ไม่ normalize) แทน")
        self._batch_load_step()

    def _normalize_user_features(self, samples):
        """Normalize feature ทั้ง 5 ตัวของ user 1 คน ด้วย z-score (ลบ mean หารด้วย std ของ
        user คนนั้นเอง แยกทีละ feature) ก่อนเอาไปรวมกับ user คนอื่น — เพราะแรงกล้ามเนื้อ/
        ตำแหน่ง electrode ของแต่ละคนไม่เท่ากัน ไม่ใช่แค่ RMS เท่านั้นที่กระทบ แต่ MAV/Variance/
        Waveform Length ก็แปรผันตามแรงกล้ามเนื้อเหมือนกัน (Zero Crossing กระทบน้อยกว่าเพราะ
        เป็นเรื่องความถี่ ไม่ใช่แอมพลิจูด แต่ normalize ให้เหมือนกันหมดเพื่อความสม่ำเสมอ)
        z-score ทำให้ทุก feature ของทุกคนอยู่ในหน่วยเดียวกันคือ 'ห่างจากค่าเฉลี่ยของตัวเองกี่
        std' แทนที่จะเป็นหน่วยดิบที่ต่างสเกลกันทั้งข้ามคนและข้าม feature

        samples: list ของ (features_dict, gesture, person_category) — normalize เฉพาะ 7
        feature จากสัญญาณ EMG เท่านั้น person_category เป็น "บริบท" ไม่ใช่ค่าดิบที่ต้อง
        z-score (one-hot อยู่แล้ว ดู emg_features.person_category_to_vector) จึงส่งผ่านไป
        เฉยๆ ไม่แตะ
        คืนค่า (normalized_samples, stats) — stats เป็น dict {feature_name: (mean, std)}
        เอาไว้ log ให้ผู้ใช้เห็น ถ้า std ใกล้ 0 (ข้อมูลแทบไม่มี variation เลย ผิดปกติ) จะ
        fallback ใช้ std=1 กัน divide-by-zero"""
        stats = {}
        for name in FEATURE_NAMES:
            vals = [f[name] for f, _, _ in samples]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals)
            if std < 1e-9:
                std = 1.0
            stats[name] = (mean, std)

        normalized = []
        for f, gesture, person_category in samples:
            norm_f = {name: (f[name] - stats[name][0]) / stats[name][1] for name in FEATURE_NAMES}
            normalized.append((norm_f, gesture, person_category))
        return normalized, stats

    def _export_ml_dataset_csv(self):
        """ส่งออก dataset ที่โหลดล่าสุด (จาก 'โหลดข้อมูลย้อนหลัง & เทรน' หรือ 'อัพโหลดไฟล์
        JSON เอง') เป็นไฟล์ .csv — คอลัมน์: gesture,rms,mav,variance,waveform_length,
        zero_crossing,slope_sign_change,iemg,person_category,timestamp,session_id,t
        (timestamp/session_id/t ไม่ได้เก็บอยู่ใน self._ml_last_loaded_samples เพราะ
        pipeline ปัจจุบันดึงมาแค่ features+gesture+person_category ที่จำเป็นต่อการเทรน
        เท่านั้น จึงเว้นว่างไว้ 3 คอลัมน์นี้เสมอ — คอลัมน์ยังคงอยู่ครบตามสเปกเพื่อให้ import
        กลับเข้าเครื่องมืออื่นที่คาดหวัง schema นี้ได้ แม้บางคอลัมน์จะว่าง)"""
        samples = self._ml_last_loaded_samples
        if not samples:
            messagebox.showinfo("ไม่มีข้อมูล", "ยังไม่มี dataset ที่โหลดไว้ — กด 'โหลดข้อมูล"
                                                 "ย้อนหลัง & เทรน' หรืออัพโหลดไฟล์ JSON ก่อน")
            return
        path = filedialog.asksaveasfilename(
            title="Export Dataset เป็น CSV", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], initialfile="emg_dataset.csv")
        if not path:
            return
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["gesture", "rms", "mav", "variance", "waveform_length",
                                  "zero_crossing", "slope_sign_change", "iemg",
                                  "person_category", "timestamp", "session_id", "t"])
                for features, gesture, person_category in samples:
                    writer.writerow([
                        gesture, features["rms"], features["mav"], features["variance"],
                        features["waveform_length"], features["zero_crossing"],
                        features["slope_sign_change"], features["iemg"],
                        person_category, "", "", "",
                    ])
            self._log(self.ml_log, f"⬇ ส่งออก dataset {len(samples)} samples → {path}")
            messagebox.showinfo("สำเร็จ", f"ส่งออกไฟล์ไปที่:\n{path}")
        except Exception as e:
            messagebox.showerror("ส่งออกไม่สำเร็จ", str(e))

    def _load_ml_file(self):
        path = filedialog.askopenfilename(
            title="เลือกไฟล์ JSON สำหรับเทรนแบบ Offline",
            filetypes=[("JSON files", "*.json *.jsonl"), ("All files", "*.*")]
        )
        if not path:
            return  # ผู้ใช้กดยกเลิก

        try:
            flat_samples, skipped, legacy_skipped = self._parse_ml_json_file(path)
        except Exception as e:
            messagebox.showerror("อ่านไฟล์ไม่สำเร็จ", f"เกิดข้อผิดพลาด:\n{e}")
            return

        if legacy_skipped:
            messagebox.showwarning(
                "พบข้อมูลรูปแบบเก่า",
                f"พบ {legacy_skipped} samples ในไฟล์นี้ที่มีแค่ RMS (ไม่มี MAV/Variance/"
                f"Waveform Length/Zero Crossing) — เป็นข้อมูลจากก่อนระบบอัพเดตเป็น "
                f"multi-feature ใช้กับโมเดลใหม่ไม่ได้ จึงถูกข้ามไปโดยอัตโนมัติ")
            self._log(self.ml_log, f"⚠️ ข้าม {legacy_skipped} samples รูปแบบเก่า (มีแค่ RMS) ในไฟล์นี้")

        if not flat_samples:
            messagebox.showwarning(
                "ไม่มีข้อมูลที่ใช้ได้",
                "ไม่พบ sample ที่มีครบทั้ง rms/mav/variance/waveform_length/zero_crossing/"
                "slope_sign_change/iemg/gesture ในไฟล์นี้เลย\nตรวจสอบว่าไฟล์เป็น JSON array "
                "ของ record ที่มี field ครบ หรือ JSONL บรรทัดละ record"
            )
            return

        old_trainer = self.trainer
        self.trainer = OnlineTrainer(self.ml_base_var.get(), self.ml_ensemble_var.get())
        self._ml_last_seen_error_count = 0  # trainer ใหม่ — รีเซ็ตตัวนับ error ที่เคย log ไปแล้ว
        old_trainer.stop()
        self._ml_replay_queue = flat_samples
        self._ml_replay_total = len(flat_samples)
        self._ml_last_loaded_samples = list(flat_samples)
        note = f" (ข้าม {skipped} record ที่อ่านไม่ได้/ไม่มี label)" if skipped else ""
        self._log(self.ml_log, f"📂 โหลด {len(flat_samples)} samples จาก {os.path.basename(path)}{note} "
                                f"— เทรนแบบ Offline (เร็ว ไม่จำลอง real-time)")
        self._batch_load_step()

    def _batch_load_step(self):
        """ป้อนข้อมูลจากไฟล์เข้า trainer เป็น batch ใหญ่ต่อเนื่องกันแบบเร็วที่สุด (ไม่หน่วงเทียม
        แบบตอน 'real-time' เพราะนี่คือข้อมูล offline อยู่แล้ว ไม่มีประโยชน์ที่จะจำลองว่ามาช้าๆ)
        ใช้ after(0, ...) แค่เพื่อคืนการควบคุมให้ Tk event loop เป็นระยะ กันหน้าจอค้างตอนไฟล์ใหญ่
        มากๆ เท่านั้น ไม่ได้ตั้งใจหน่วงเวลา — ตัวเทรนจริงเบื้องหลังจะถี่แค่ไหนก็ยังถูกคุมด้วย
        min_retrain_interval ใน OnlineTrainer อยู่ดี ไม่มีทางกิน CPU เกินเหมือนเดิม"""
        if not self._ml_replay_queue:
            # โหลดครบแล้ว — สั่งเทรนรอบสุดท้ายทันที (ข้าม throttle) ด้วยข้อมูล "ทั้งหมด"
            # ที่เพิ่งโหลดมา กันปัญหาที่ตัวเลขค้างอยู่ที่ค่าเก่าตอน throttle จำกัดความถี่ไว้
            # ระหว่างที่กำลังยัดข้อมูลเข้าเร็วๆ (ปัญหาที่เจอมาก่อนหน้านี้)
            self.ml_status_lbl.config(text=f"โหลดครบ {self.trainer.n_samples} samples — กำลังเทรนรอบสุดท้าย...")
            # ⚠️ เดิม capture n_retrains_before อย่างเดียวแล้วรอแค่ "มีรอบเทรนใหม่เกิดขึ้น" —
            # ไม่พอ เพราะ background thread อาจเทรนกลางคันด้วยข้อมูลบางส่วนไปแล้วระหว่าง loop
            # ป้อนข้อมูลด้านบน (ทุกครั้งที่สะสมครบ retrain_every_n) รอบที่เพิ่งเสร็จตอนเราเช็ค
            # อาจเป็นรอบเก่าที่ข้อมูลไม่ครบ ไม่ใช่รอบใหม่ที่ครบจริง ทำให้ metric ที่อ่านได้ขึ้น
            # กับจังหวะ thread ในแต่ละครั้งที่รัน (สาเหตุจริงที่ผลเทรนไม่นิ่งแม้ข้อมูล/seed
            # เดิมทุกอย่าง) — จดจำนวน sample ที่ควรเทรนครบไว้ก่อน แล้วรอจนกว่าจะเจอรอบที่
            # history ล่าสุดยืนยันว่าเทรนด้วยจำนวนครบตามนี้จริง (ดู _wait_final_retrain)
            expected_min_n_train = self.trainer.get_snapshot()["split_summary"]["n_train"]
            n_retrains_before = self.trainer.get_snapshot()["n_retrains"]
            self.trainer.force_retrain_now()
            self._wait_final_retrain(n_retrains_before, expected_min_n_train, time.time() + 15.0)
            return
        batch, self._ml_replay_queue = self._ml_replay_queue[:500], self._ml_replay_queue[500:]
        for features, gesture, person_category in batch:
            self.trainer.add_sample(features, gesture, person_category)
        done = self._ml_replay_total - len(self._ml_replay_queue)
        self.ml_status_lbl.config(text=f"กำลังโหลด... {done}/{self._ml_replay_total} samples")
        self.after(0, self._batch_load_step)

    def _wait_final_retrain(self, n_retrains_before, expected_min_n_train, deadline):
        """รอ (แบบไม่บล็อก GUI — โพลผ่าน after) จนกว่ารอบเทรนสุดท้ายจากไฟล์ offline จะเสร็จ
        แล้วค่อยรีเฟรชตัวเลขบนจอ ให้สะท้อนข้อมูลทั้งหมดที่โหลดมาจริงๆ

        เงื่อนไขที่ต้องครบทั้ง 2 ข้อก่อนถือว่า 'เทรนรอบสุดท้ายเสร็จจริง' (กัน race กับรอบที่
        เทรนค้างอยู่ก่อนหน้าด้วยข้อมูลไม่ครบ — ดู comment ใน _batch_load_step):
          1. n_retrains เพิ่มขึ้นจากตอนก่อนสั่ง force_retrain_now
          2. รอบล่าสุดใน history เทรนด้วย sample จำนวน >= expected_min_n_train (ครบจริง)"""
        snap = self.trainer.get_snapshot()
        history = snap["history"]
        last_n_train = history[-1]["n_train"] if history else 0
        complete = snap["n_retrains"] > n_retrains_before and last_n_train >= expected_min_n_train
        if complete or time.time() > deadline:
            if not complete:
                self._log(self.ml_log, f"⚠️ เทรนไม่ครบ dataset ทันภายในเวลาที่กำหนด "
                                        f"(เทรนด้วย {last_n_train}/{expected_min_n_train} "
                                        f"samples) — ผลลัพธ์อาจไม่นิ่ง 100% ลองกดโหลด/เทรนซ้ำ")
            self._log(self.ml_log, f"✅ เทรนรอบสุดท้ายเสร็จแล้ว (samples={snap['n_samples']}, "
                                    f"retrains={snap['n_retrains']})")
            self._refresh_ml_metrics()
            return
        self.after(100, lambda: self._wait_final_retrain(
            n_retrains_before, expected_min_n_train, deadline))

    def _refresh_ml_metrics(self):
        snap = self.trainer.get_snapshot()
        self.ml_status_lbl.config(
            text=f"Samples: {snap['n_samples']}   |   เทรนไปแล้ว: {snap['n_retrains']} ครั้ง")
        self._render_metrics_table(snap)
        self._render_confusion(snap["confusion"])
        self._render_dash_metrics(snap)
        self._render_dash_confusion(snap["confusion"])
        self._render_ml_split_summary(snap)
        self._render_ml_loss_graph(snap)
        self._render_ml_performance_graph(snap)
        self._render_ml_predict_graph(snap)
        self._render_ml_boundary_graph(snap)

        if snap["accuracy"] is not None:
            # บันทึกผลลัพธ์ล่าสุดของโมเดลนี้ไว้เทียบกับโมเดลอื่นๆ ที่เคยเทรนในหน้า
            # "สรุปผลโมเดล" — ใช้ display_name เป็น key เขียนทับของเดิมถ้าเทรนโมเดล
            # เดียวกันซ้ำ (เก็บแค่ผลล่าสุดของแต่ละ metric ไม่สะสมประวัติซ้ำๆ ให้รก ยกเว้น
            # 'history' ที่เก็บ accuracy ย้อนหลังไว้ 20 ค่าล่าสุด ใช้วาด sparkline แนวโน้ม)
            prev_history = self.model_results.get(self.trainer.display_name, {}).get("history", [])
            history = (prev_history + [snap["accuracy"]])[-20:]
            self.model_results[self.trainer.display_name] = {
                "accuracy": snap["accuracy"], "precision": snap["precision"],
                "recall": snap["recall"], "f1": snap["f1"],
                "auc": snap["auc"], "kappa": snap["kappa"],
                "n_samples": snap["n_samples"],
                "trained_at": datetime.now().strftime("%H:%M:%S"),
                "history": history,
            }
            self._compare_last_updated = datetime.now()
            if hasattr(self, "compare_table_frame"):
                self._refresh_model_comparison()

    def _render_dash_metrics(self, snap):
        """อัพเดตการ์ด metric เล็กๆ ในหน้า Dashboard (ใช้ snapshot เดียวกับแท็บ ML)"""
        if snap["accuracy"] is None:
            for lbl in self.dash_metric_lbls.values():
                lbl.config(text="—")
            self.dash_metrics_summary_lbl.config(text="ยังไม่มีข้อมูลพอประเมิน")
            return
        values = {
            "Accuracy":  f"{snap['accuracy']*100:.2f}%",
            "Precision": f"{snap['precision']*100:.2f}%",
            "Recall":    f"{snap['recall']*100:.2f}%",
            "F1-score":  f"{snap['f1']:.4f}",
            "AUC":       f"{snap['auc']:.4f}" if snap["auc"] is not None else "—",
            "Kappa":     f"{snap['kappa']:.4f}",
        }
        for name, text in values.items():
            self.dash_metric_lbls[name].config(text=text)
        self.dash_metrics_summary_lbl.config(
            text=f"Samples: {snap['n_samples']:,}   |   เทรนไปแล้ว: {snap['n_retrains']} ครั้ง")

    def _render_metrics_table(self, snap):
        if snap["accuracy"] is None:
            self.ml_metrics_lbl.config(
                text="ยังไม่มีข้อมูลพอประเมิน (ต้องมีทั้ง FIST และ OPEN ใน test set)")
            return
        rows = [
            f"Accuracy   {snap['accuracy']*100:6.2f}%   (ทายถูกกี่ %)",
            f"Precision  {snap['precision']*100:6.2f}%   (ทายว่า FIST แล้วถูกจริงกี่ %)",
            f"Recall     {snap['recall']*100:6.2f}%   (หาเคส FIST เจอครบแค่ไหน)",
            f"F1-score   {snap['f1']:.4f}     (คะแนนรวม Precision + Recall)",
            f"AUC        {snap['auc']:.4f}     (ใกล้ 1 = แยกคลาสได้ดีมาก)" if snap["auc"] is not None
            else "AUC        —",
            f"Kappa      {snap['kappa']:.4f}     (ความน่าเชื่อถือเทียบกับเดาสุ่ม)",
        ]
        self.ml_metrics_lbl.config(text="\n".join(rows))

    def _render_confusion(self, conf):
        if conf is None:
            self.ml_confusion_lbl.config(
                text="ยังไม่มีข้อมูลพอประเมิน (ต้องมีทั้ง FIST และ OPEN ใน test set)")
            self._render_confusion_heatmap(None)
            return
        labels = list(CLASSES)
        header = " " * 14 + "".join(f"ทำนาย:{l:<8}" for l in labels)
        rows = [header]
        for i, actual in enumerate(labels):
            row = f"จริง:{actual:<9}" + "".join(f"{conf[i][j]:>14d}" for j in range(len(labels)))
            rows.append(row)
        self.ml_confusion_lbl.config(text="\n".join(rows))
        self._render_confusion_heatmap(conf)

    def _render_confusion_heatmap(self, conf):
        """วาด Confusion Matrix แบบ heatmap (matplotlib) คู่กับตัวอักษรใน _render_confusion
        — ใช้ตอบโจทย์ 'บทที่ 4 ต้องมีรูปผลการทดลอง' เพราะเป็นรูปที่บันทึกเป็น .png ได้ตรงๆ
        ผ่าน _save_figure_png() ต่างจาก ml_confusion_lbl ที่เป็นแค่ตัวอักษรล้วน"""
        ax = self._ml_ax_cm
        ax.clear()
        ax.set_facecolor(COLORS["card"])
        labels = list(CLASSES)
        if conf is None:
            ax.text(0.5, 0.5, "Not enough data yet\n(need a Test Set evaluation first)",
                    ha="center", va="center", color=COLORS["text_dim"], fontsize=8,
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._ml_canvas_cm.draw_idle()
            return
        im = ax.imshow(conf, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, color=COLORS["text_dim"], fontsize=8)
        ax.set_yticklabels(labels, color=COLORS["text_dim"], fontsize=8)
        ax.set_xlabel("Predicted", color=COLORS["text_dim"], fontsize=8)
        ax.set_ylabel("Actual", color=COLORS["text_dim"], fontsize=8)
        vmax = conf.max() if conf.max() > 0 else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = int(conf[i][j])
                # สีตัวเลขต้องอิงความสว่างของ "เซลล์ heatmap" (cmap Blues) ไม่ใช่สีตัวอักษร
                # ของธีมแอป (COLORS["text"]) — เดิมใช้ COLORS["text"] ซึ่งเป็นสีขาวอมฟ้าอ่อน
                # (#e8ecf7) ที่ออกแบบมาให้อ่านง่ายบนพื้นหลังกรมท่าเข้มของแอป แต่ Blues cmap
                # ค่าน้อยจะได้เซลล์สีขาว/อ่อนเกือบขาว ทำให้ตัวเลขที่เป็น #e8ecf7 กลืนไปกับพื้น
                # เซลล์จนมองแทบไม่เห็น แก้โดยสลับเป็นขาว/ดำล้วนตามความเข้มของเซลล์แทน
                color = "#ffffff" if val > vmax * 0.5 else "#000000"
                ax.text(j, i, str(val), ha="center", va="center", color=color,
                        fontsize=11, fontweight="bold")
        self._ml_fig_cm.tight_layout()
        self._ml_canvas_cm.draw_idle()

    def on_close(self):
        self.recording = False
        self._auto_sim_running = False
        if self._auto_sim_after_id is not None:
            try:
                self.after_cancel(self._auto_sim_after_id)
            except Exception:
                pass
        if self._auto_sim_sample_after_id is not None:
            try:
                self.after_cancel(self._auto_sim_sample_after_id)
            except Exception:
                pass
        self.stream.stop()
        self.firebase.stop()
        self.trainer.stop()
        self.destroy()

# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from login_window import LoginWindow

    # แสดงหน้าต่าง Login ก่อนเสมอ — ต้องล็อกอินผ่าน Firebase Authentication และมีสิทธิ์
    # Admin (เช็คจาก node "admins" ใน Realtime Database) ถึงจะเข้าโปรแกรมหลักได้
    # ดู auth_manager.py และ setup_admin.py สำหรับตั้งค่าบัญชี Admin คนแรก
    login = LoginWindow()
    login.mainloop()

    if login.logged_in_user is None:
        raise SystemExit(0)  # ปิดหน้าต่าง Login เอง หรือล็อกอินไม่สำเร็จ → จบโปรแกรม

    app = ProstheticApp()
    app.current_user = login.logged_in_user  # เผื่ออยากเอาไปโชว์/ใช้ต่อในแอปหลัก
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()