"""
firebase_sync.py
บันทึกข้อมูล EMG ลงไฟล์ Text (.jsonl) ในเครื่อง (Raspberry Pi) ก่อนเสมอ
แล้วค่อยซิงก์ขึ้น Firebase Realtime Database เป็นงานเบื้องหลัง (background thread)

ทำไมต้องทำแบบนี้ (Offline-first):
  - Raspberry Pi ที่ติดอยู่กับมือเทียม อาจไม่มีอินเทอร์เน็ตตลอดเวลา
  - ถ้าเขียนขึ้น Firebase ตรงๆ ทุก sample แล้วเน็ตหลุด ข้อมูลจะหายทันที
  - วิธีนี้: เขียนไฟล์ .jsonl ในเครื่องก่อนเสมอ (ไม่มีวันข้อมูลหาย) แล้วเธรดพื้นหลัง
    จะคอยส่ง record ที่ยัง "synced": false ขึ้น Firebase เมื่อออนไลน์ ทีละ batch
    ถ้าเน็ตหลุดระหว่างซิงก์ ก็แค่รอแล้วลองใหม่ โดยไม่กระทบการเก็บข้อมูล real-time เลย

หมายเหตุเรื่องอัตราการส่งขึ้น Firebase:
  สัญญาณ EMG ดิบมี sample rate ~500Hz ซึ่งเร็วเกินไปที่จะส่งขึ้น Cloud Database
  ทุก sample (ทั้งแพงและช้า) ไฟล์ .jsonl ในเครื่องจะเก็บ "ทุก sample" ไว้ครบ
  (เอาไว้ใช้เทรน ML/ตรวจสอบย้อนหลัง) แต่ค่าที่จะซิงก์ขึ้น Firebase จะถูก throttle
  ผ่านพารามิเตอร์ cloud_sample_every (ดีฟอลต์ทุก 25 samples ~ 20Hz) เพื่อไม่ให้ Firebase
  โดนยิง request รัวเกินไป — ปรับได้ตามต้องการ
"""

import json
import os
import threading
import time
from datetime import datetime

try:
    import pyrebase
    PYREBASE_AVAILABLE = True
except ImportError:
    PYREBASE_AVAILABLE = False
    print("[WARNING] pyrebase4 ไม่ได้ติดตั้ง → บันทึกได้เฉพาะไฟล์ในเครื่อง (Offline only)")
    print("          ติดตั้งด้วย: pip install pyrebase4 --break-system-packages")

# ─── กำหนดค่า Firebase ───────────────────────────────────────────────────────
# ค่าจริงของโปรเจกต์ (apiKey/databaseURL/ฯลฯ) ไม่ได้เก็บไว้ในไฟล์นี้โดยตรง — ย้ายไปไว้ใน
# firebase_config.py แยกต่างหาก (ไฟล์นี้ถูกใส่ไว้ใน .gitignore แล้ว จะไม่ถูก push ขึ้น Git
# repo สาธารณะโดยไม่ตั้งใจ) ดูวิธีตั้งค่าใน README.md หัวข้อ "การติดตั้ง" — ถ้ายังไม่เคยสร้าง
# ไฟล์นี้ ให้คัดลอกจาก firebase_config.example.py แล้วใส่ค่าจริงจาก Firebase Console เอง
try:
    from firebase_config import FIREBASE_CONFIG
except ImportError:
    FIREBASE_CONFIG = {
        "apiKey": "YOUR_API_KEY", "authDomain": "", "databaseURL": "",
        "projectId": "", "storageBucket": "", "messagingSenderId": "", "appId": "",
    }
    print("[WARNING] ไม่พบไฟล์ firebase_config.py — คัดลอกจาก firebase_config.example.py "
          "แล้วใส่ค่าจริงจาก Firebase Console ก่อนใช้งานจริง (ดู README.md หัวข้อการติดตั้ง)")

LOCAL_LOG_DIR = "emg_logs"
SESSIONS_DIR = os.path.join(LOCAL_LOG_DIR, "sessions")        # emg_logs/sessions/<user_id>.jsonl
CALIB_DIR = os.path.join(LOCAL_LOG_DIR, "calibrations")        # emg_logs/calibrations/<user_id>.jsonl
DATASETS_DIR = os.path.join(LOCAL_LOG_DIR, "datasets")          # emg_logs/datasets/<user_id>.jsonl
SESSION_META_DIR = os.path.join(LOCAL_LOG_DIR, "session_meta")  # emg_logs/session_meta/<user_id>.jsonl
CURSOR_FILE = os.path.join(LOCAL_LOG_DIR, "sync_cursors.json")  # จำ byte offset ที่ sync ไปแล้วต่อไฟล์

SYNC_RETRY_SEC = 5      # เวลาลองต่อ Firebase ใหม่เมื่อ offline
SYNC_IDLE_SEC = 1.0     # เวลาพักระหว่างรอบ sync เมื่อ online
SYNC_BATCH_SIZE = 200   # ส่งขึ้น Firebase ครั้งละไม่เกินกี่ record ต่อไฟล์ต่อรอบ


def _safe_user_id(user_id: str) -> str:
    """กันชื่อไฟล์เพี้ยน ถ้า user_id มีอักขระที่ใช้เป็นชื่อไฟล์ไม่ได้ (space, /, \\, ฯลฯ)"""
    user_id = str(user_id).strip()
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id)
    return safe or "unknown_user"


def _session_path(user_id: str) -> str:
    """ไฟล์ log ของ user คนนั้นๆ โดยเฉพาะ เช่น emg_logs/sessions/user_0.jsonl"""
    return os.path.join(SESSIONS_DIR, f"{_safe_user_id(user_id)}.jsonl")


def _calib_path(user_id: str) -> str:
    return os.path.join(CALIB_DIR, f"{_safe_user_id(user_id)}.jsonl")


def _dataset_path(user_id: str) -> str:
    """ไฟล์เก็บ 'เหตุการณ์กำมือ' (gesture event) ของ user คนนั้นๆ — เขียนโดย
    GestureEventRecorder ผ่าน log_dataset_event() เท่านั้น (1 บรรทัด = 1 เหตุการณ์
    กำมือที่บันทึกครบแล้ว ไม่ใช่ 1 sample แบบไฟล์ sessions)"""
    return os.path.join(DATASETS_DIR, f"{_safe_user_id(user_id)}.jsonl")


def _session_meta_path(user_id: str) -> str:
    """ไฟล์เก็บ metadata ของแต่ละ session การบันทึกแบบ Real-time (เริ่ม/จบตอนไหน,
    sample_rate/threshold ตอนนั้น) — แยกจากไฟล์ sessions หลัก (ซึ่งเก็บ sample ดิบทุกตัว)
    เขียนโดย start_session()/end_session() เท่านั้น ไม่ใช่ log_sample()"""
    return os.path.join(SESSION_META_DIR, f"{_safe_user_id(user_id)}.jsonl")


def session_log_path(user_id: str) -> str:
    """Public wrapper รอบ _session_path — ให้โมดูลอื่น (เช่น prosthetic_gui.py) เอาไปใช้
    หาไฟล์ log สัญญาณ EMG ของ user คนนั้นๆ เพื่ออ่านข้อมูลย้อนหลังมาเทรน ML แบบ Offline
    โดยไม่ต้อง import ฟังก์ชันที่ขึ้นต้นด้วย _ ตรงๆ"""
    return _session_path(user_id)


def _add_extra_features(record: dict, mav, variance, waveform_length, zero_crossing,
                          slope_sign_change=None, iemg=None, person_category=None):
    """เติม feature เพิ่มเติม (นอกจาก rms) เข้า record ให้เฉพาะตัวที่มีค่าจริง (ไม่ใช่ None)
    เท่านั้น — ใช้ร่วมกันทั้ง log_sample() และ log_labeled_sample() กันโค้ดซ้ำ

    เจตนาที่ไม่ใส่ key ที่เป็น None ลงไปเลย (แทนที่จะใส่เป็น null): เพื่อให้แยกได้ชัดเจนระหว่าง
    (1) record เก่าก่อนอัพเดตระบบนี้ ที่ไม่มี key พวกนี้อยู่เลย กับ (2) record ใหม่ที่ตั้งใจ
    ส่งมาแค่ rms (ไม่ผ่าน multi-feature extraction) — ทั้งสองแบบไม่มี key พวกนี้เหมือนกัน ทำให้
    ml_online.py / prosthetic_gui.py เช็คแค่ 'key มีอยู่ไหม' เพื่อรู้ว่า record นี้ใช้กับโมเดล
    multi-feature ได้หรือไม่ (ดู requirement เรื่อง Backward Compatibility) — ครบทั้ง 6 ตัว
    (mav/variance/waveform_length/zero_crossing/slope_sign_change/iemg) ถือว่าใช้ได้ ขาด
    ตัวใดตัวหนึ่งถือว่าเป็น record รูปแบบเก่า (ดู prosthetic_gui._record_to_sample)

    person_category: ประเภทบุคคล (ปกติ/เด็ก/คนแก่/ผู้ป่วยกล้ามเนื้อ/ผู้พิการ — ดู
    emg_features.PERSON_CATEGORIES) เก็บแยก key ต่างหาก ('person_category') ไม่ปนกับ 6 ตัว
    ข้างบน โดยเจตนา — เป็น optional เหมือนกัน (ไม่ระบุ = ไม่เขียน key นี้ลง record เลย) เพื่อ
    ไม่ให้กระทบการเช็ค 'record รูปแบบเก่า/ใหม่' ของ multi-feature เดิม ฝั่งที่อ่านย้อนกลับ (เช่น
    prosthetic_gui._record_to_sample) จะ fallback เป็น emg_features.DEFAULT_PERSON_CATEGORY
    เองถ้า record ไหนไม่มี key นี้ (เช่น record เก่าก่อนมีฟีเจอร์นี้)"""
    if mav is not None:
        record["mav"] = round(mav, 6)
    if variance is not None:
        record["variance"] = round(variance, 8)
    if waveform_length is not None:
        record["waveform_length"] = round(waveform_length, 6)
    if zero_crossing is not None:
        record["zero_crossing"] = int(zero_crossing)
    if slope_sign_change is not None:
        record["slope_sign_change"] = int(slope_sign_change)
    if iemg is not None:
        record["iemg"] = round(iemg, 6)
    if person_category is not None:
        record["person_category"] = str(person_category)


def list_local_users() -> list:
    """คืนรายชื่อ user_id ทั้งหมดที่มีไฟล์ log อยู่ในเครื่อง (เผื่ออยากดูว่ามีใครเก็บข้อมูลไว้บ้าง)"""
    users = set()
    for d in (SESSIONS_DIR, CALIB_DIR, DATASETS_DIR):
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.endswith(".jsonl"):
                    users.add(fname[:-len(".jsonl")])
    return sorted(users)


def _load_cursors() -> dict:
    """โหลดตำแหน่ง (byte offset) ที่ sync ไปถึงแล้วของแต่ละไฟล์ — ใช้แทนการอ่านทั้งไฟล์
    log ทุกรอบ (ไฟล์ log สัญญาณ EMG จริงมีเป็นแสนเป็นล้านบรรทัดถ้าใช้งานไปนานๆ อ่าน/เขียน
    ทั้งไฟล์ทุกวินาทีจะทำให้แอปค้าง — ดู FirebaseSyncManager._sync_file)"""
    if os.path.exists(CURSOR_FILE):
        try:
            with open(CURSOR_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cursors(cursors: dict):
    """เขียนไฟล์ cursor แบบ atomic (เขียนไฟล์ tmp แล้ว rename ทับ) กันไฟล์เสียถ้าโปรแกรม
    ถูกปิดกลางคันระหว่างเขียนพอดี — ไฟล์นี้เล็กมาก (แค่ path -> offset) เขียนเร็ว ไม่หนักเครื่อง"""
    os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
    tmp_path = CURSOR_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cursors, f)
    os.replace(tmp_path, CURSOR_FILE)


class FirebaseSyncManager:
    def __init__(self, cloud_sample_every: int = 25, auto_start: bool = True):
        """
        cloud_sample_every: ซิงก์ค่า EMG sample ขึ้น Firebase ทุกๆ กี่ sample
                             (ไฟล์ในเครื่องยังคงบันทึกทุก sample เสมอ ไม่ถูก throttle)

        โครงสร้างไฟล์ในเครื่อง (แยกเป็นคนละไฟล์ต่อ user เพื่อเอาไปเทรน ML แยกคนได้ง่าย):
            emg_logs/sessions/<user_id>.jsonl       — ข้อมูล EMG real-time ทุก sample
            emg_logs/calibrations/<user_id>.jsonl   — ข้อมูล Calibration จากแท็บ Calibration ใน GUI
            emg_logs/datasets/<user_id>.jsonl       — เหตุการณ์กำมือที่บันทึกครบเป็นก้อนๆ
                                                        (จาก GestureEventRecorder)
        """
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        os.makedirs(CALIB_DIR, exist_ok=True)
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(SESSION_META_DIR, exist_ok=True)
        self.cloud_sample_every = max(1, cloud_sample_every)
        self._sample_counter = 0

        self.db = None
        self.online = False
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._cursors = _load_cursors()  # {path: byte_offset ที่ sync ไปแล้ว} — กันอ่าน/เขียนทั้งไฟล์ทุกรอบ

        if PYREBASE_AVAILABLE:
            self._connect()

        self._sync_thread_started = False
        if auto_start:
            self.start()

    def start(self):
        """เริ่มเธรด sync พื้นหลัง (idempotent — เรียกซ้ำได้ ไม่เริ่มเธรดซ้อนอีกเธรด)
        แยกออกมาจาก __init__ เพื่อให้ผู้เรียก (เช่น GUI) เลือกเองได้ว่าจะเริ่ม sync ขึ้น
        Firebase ตอนไหน — เช่น รอจนกว่าผู้ใช้จะกด 'เริ่มบันทึกแบบ Real-time' เองครั้งแรก
        ก่อน แทนที่จะ sync backlog เก่าขึ้น Firebase อัตโนมัติทันทีที่เปิดแอป โดยที่ผู้ใช้
        ยังไม่ทันกดอะไรเลย (ข้อมูลใน local ไฟล์ยังปลอดภัยเสมอไม่ว่าจะ sync หรือยัง —
        แค่ยังไม่ขึ้น Firebase จนกว่าจะเริ่ม)"""
        if self._sync_thread_started:
            return
        self._sync_thread_started = True
        threading.Thread(target=self._sync_loop, daemon=True).start()

    def _connect(self):
        try:
            app = pyrebase.initialize_app(FIREBASE_CONFIG)
            self.db = app.database()
            self.online = True
            print("[Firebase] เชื่อมต่อสำเร็จ")
        except Exception as e:
            self.online = False
            print(f"[Firebase] เชื่อมต่อไม่ได้ ({e}) → ทำงานในโหมด Offline")

    # ---------- Public API: เขียนไฟล์ local เสมอ ไม่บล็อกรอ Firebase ----------

    def log_sample(self, user_id: str, voltage: float, rms: float, gesture: str, t: float,
                    session_id: str = None, mav: float = None, variance: float = None,
                    waveform_length: float = None, zero_crossing: float = None,
                    slope_sign_change: float = None, iemg: float = None,
                    person_category: str = None):
        """
        บันทึกค่า EMG 1 sample ลงไฟล์ .jsonl ในเครื่องแบบ real-time เสมอ
        (การซิงก์ขึ้น Firebase จะถูก throttle ตาม cloud_sample_every โดยอัตโนมัติ
        ผ่าน flag 'cloud_sync' ที่ติดไปกับ record — เธรด sync จะข้าม record ที่ cloud_sync=False)

        session_id: ผูก sample นี้เข้ากับ session การบันทึกแบบ Real-time ที่กำลังเปิดอยู่
                    (ถ้ามี — ดู start_session()) เพื่อให้รู้ได้ว่า sample ชุดไหนมาจาก
                    session ไหนตอนเอาไป train ML โดยไม่ต้องยัด sample_rate/threshold
                    ซ้ำในทุก sample เอง (metadata พวกนั้นอยู่ใน session_meta แทน)

        mav/variance/waveform_length/zero_crossing: Feature เพิ่มเติมนอกเหนือจาก RMS
                    (ดู emg_features.extract_features) — เป็น optional โดยเจตนา (ดีฟอลต์
                    None) เพื่อไม่ให้โค้ดเดิมที่เรียก log_sample() แบบเดิม (มีแค่ rms) พัง
                    ค่าที่เป็น None จะไม่ถูกเขียนลง record เลย (ไม่ใช่เขียนเป็น null) เพื่อให้
                    แยกออกได้ชัดเจนระหว่าง record เก่า (ไม่มี key พวกนี้เลย) กับ record ใหม่
                    ที่มีค่าจริงครบ — ดู _fetch_firebase_samples ในฝั่ง prosthetic_gui.py ที่ใช้
                    ความแตกต่างนี้เช็ค backward-compatibility ของ dataset เก่า/ใหม่

        person_category: ประเภทบุคคลของเจ้าของสัญญาณ sample นี้ (ปกติ/เด็ก/คนแก่/ผู้ป่วย
                    กล้ามเนื้อ/ผู้พิการ — ดู emg_features.PERSON_CATEGORIES) เก็บขึ้น Firebase
                    ไปด้วยเพื่อใช้เป็น feature เพิ่มเติมตอนเทรนโมเดลกำ-คลายมือ (ดู
                    ml_online.OnlineTrainer.add_sample) — optional เหมือนกัน (None = ไม่เขียน
                    key นี้ลง record)
        """
        self._sample_counter += 1
        should_sync_cloud = (self._sample_counter % self.cloud_sample_every == 0)

        record = {
            "type": "sample",
            "user_id": user_id,
            "voltage": round(voltage, 6),
            "rms": round(rms, 6),
            "gesture": gesture,
            "t": round(t, 4),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "cloud_sync": should_sync_cloud,
            "synced": not should_sync_cloud,  # record ที่ไม่ได้ตั้งใจส่งขึ้น cloud ถือว่า "synced" แล้ว
        }
        _add_extra_features(record, mav, variance, waveform_length, zero_crossing,
                            slope_sign_change, iemg, person_category)
        self._append_local(_session_path(user_id), record)

    def log_calibration(self, user_id: str, model: str, samples: list, person_category: str = None):
        """บันทึกชุด Calibration — ส่งขึ้น Firebase เสมอ (ไม่ throttle เพราะเกิดไม่บ่อย)

        person_category: ประเภทบุคคลของ user คนนี้ตอนคาลิเบรต (optional — ดู
        emg_features.PERSON_CATEGORIES) เก็บไว้เป็นข้อมูลอ้างอิงคู่กับชุด Calibration"""
        record = {
            "type": "calibration",
            "user_id": user_id,
            "model": model,
            "samples": samples,
            "person_category": person_category,
            "timestamp": datetime.now().isoformat(),
            "cloud_sync": True,
            "synced": False,
        }
        self._append_local(_calib_path(user_id), record)
        return record

    def log_labeled_sample(self, user_id: str, rms: float, gesture: str,
                            t: float = None, session_id: str = None, voltage: float = None,
                            mav: float = None, variance: float = None,
                            waveform_length: float = None, zero_crossing: float = None,
                            slope_sign_change: float = None, iemg: float = None,
                            person_category: str = None):
        """บันทึก 1 sample ที่มี label ยืนยันแน่นอนแล้ว (เช่นจากฟีเจอร์ 'จำลองเก็บข้อมูล
        อัตโนมัติ' ในแท็บ Calibration ที่นับถอยหลังสลับคลายมือ/กำมือให้เอง) ลงไฟล์
        sessions/<user_id>.jsonl ด้วย record type='sample' โครงสร้างเดียวกันกับที่
        log_sample() เขียนทุกประการ (ไม่ใช่ type='calibration' แบบ log_calibration ที่เก็บ
        เป็นก้อน samples ซ้อนอยู่ใน record เดียว) — เพราะ _fetch_firebase_samples() ใน
        prosthetic_gui.py (ใช้โหลดข้อมูลย้อนหลังมาเทรน ML) อ่านเฉพาะ record type='sample'
        ที่มี field 'rms'/'gesture' อยู่ระดับบนสุดเท่านั้น ถ้าเก็บผ่าน log_calibration
        แทน ข้อมูลจะไปกองอยู่ node 'calibrations' ในรูปแบบที่ ML tab อ่านไม่ได้เลย

        mav/variance/waveform_length/zero_crossing/person_category: เหมือน log_sample() —
        optional เพื่อ backward-compat ดู docstring ของ log_sample() ประกอบ

        ไม่ throttle การซิงก์ขึ้น Firebase (cloud_sync=True เสมอ ไม่ผ่าน cloud_sample_every
        เหมือน log_sample) เพราะชุดข้อมูลนี้เก็บจำนวนจำกัดโดยเจตนา (เช่น 50 samples/ช่วง)
        ไม่ใช่สัญญาณต่อเนื่องที่ ~500Hz จึงไม่ต้องกลัวยิง Firebase ถี่เกินไป"""
        record = {
            "type": "sample",
            "user_id": user_id,
            "voltage": round(voltage, 6) if voltage is not None else None,
            "rms": round(rms, 6),
            "gesture": gesture,
            "t": round(t, 4) if t is not None else None,
            "session_id": session_id,
            "source": "auto_label",  # แยกให้รู้ว่ามาจากการจำลอง/label ไม่ใช่สัญญาณสดจริง
            "timestamp": datetime.now().isoformat(),
            "cloud_sync": True,
            "synced": False,
        }
        _add_extra_features(record, mav, variance, waveform_length, zero_crossing,
                            slope_sign_change, iemg, person_category)
        self._append_local(_session_path(user_id), record)
        return record

    def log_dataset_event(self, user_id: str, event: dict):
        """บันทึก 'เหตุการณ์กำมือ' 1 ครั้ง (จาก GestureEventRecorder — รวมสัญญาณตั้งแต่
        pre-roll ก่อนกำมือ ผ่านช่วงกำมือ+คลายมือสลับกัน จนคลายมือค้างครบเวลาที่ตั้งไว้)
        ลงไฟล์ในเครื่องก่อนเสมอ (offline-first แบบเดียวกับ log_sample/log_calibration)
        แล้วให้เธรด sync พื้นหลังส่งขึ้น Firebase node 'datasets' ทีหลัง

        ไม่ throttle การ sync (cloud_sync=True เสมอ) เพราะ event เกิดเป็นครั้งๆ ไม่บ่อย
        เท่า sample ดิบที่ ~500Hz จึงไม่ต้องกลัวยิง Firebase ถี่เกินไป"""
        record = {
            "type": "gesture_event",
            "user_id": user_id,
            "start_time": event.get("start_time"),
            "end_time": event.get("end_time"),
            "n_samples": event.get("n_samples", 0),
            "samples": event.get("samples", []),
            "timestamp": datetime.now().isoformat(),
            "cloud_sync": True,
            "synced": False,
        }
        self._append_local(_dataset_path(user_id), record)
        return record

    def start_session(self, user_id: str, session_id: str, sample_rate=None, threshold=None):
        """บันทึก metadata ตอนเริ่ม session การบันทึกสัญญาณ EMG แบบ Real-time 1 รอบ
        (เรียกจาก GUI ตอนกด 'เริ่มบันทึกแบบ Real-time') — เก็บ start_time ไว้ใช้คู่กับ
        end_session() ทีหลัง พร้อมผูก sample_rate/threshold ตอนนั้นไว้ในระเบียนเดียว
        แทนที่จะยัดซ้ำในทุก sample (แต่ละ sample แปะแค่ session_id อ้างอิงกลับมาแทน
        ดู log_sample ด้านบน)

        คืนค่า dict ที่มี key 'start_time' (ISO string) — ผู้เรียกใช้เก็บไว้ส่งต่อให้
        end_session() ทีหลังตอนจบ session"""
        start_time = datetime.now().isoformat()
        record = {
            "type": "session_start",
            "user_id": user_id,
            "session_id": session_id,
            "start_time": start_time,
            "sample_rate": sample_rate,
            "threshold": threshold,
            "timestamp": start_time,
            "cloud_sync": True,
            "synced": False,
        }
        self._append_local(_session_meta_path(user_id), record)
        return record

    def end_session(self, user_id: str, session_id: str, start_time: str = None):
        """บันทึก metadata ตอนจบ session — ไม่ว่าจะกด 'หยุดบันทึก Real-time' เอง หรือ
        ระบบหยุดให้อัตโนมัติ (เช่นตอนเปลี่ยน/ลบ user) เขียนเป็นระเบียนใหม่แยกต่างหาก
        (ไม่ได้ย้อนไปแก้ระเบียน session_start เดิม เพราะไฟล์ log เป็นแบบ append-only
        เสมอตามหลักการเดียวกับไฟล์ log อื่นๆ ในโมดูลนี้) อ้างอิงกลับไปยัง session_id เดิม"""
        end_time = datetime.now().isoformat()
        record = {
            "type": "session_end",
            "user_id": user_id,
            "session_id": session_id,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": end_time,
            "cloud_sync": True,
            "synced": False,
        }
        self._append_local(_session_meta_path(user_id), record)
        return record

    def _append_local(self, path, record):
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- Background sync ----------

    def _sync_loop(self):
        while not self._stop_flag.is_set():
            if not PYREBASE_AVAILABLE:
                time.sleep(SYNC_RETRY_SEC)
                continue

            if not self.online:
                self._connect()
                if not self.online:
                    time.sleep(SYNC_RETRY_SEC)
                    continue

            try:
                self._sync_dir(SESSIONS_DIR, "sessions")
                self._sync_dir(CALIB_DIR, "calibrations")
                self._sync_dir(DATASETS_DIR, "datasets")
                self._sync_dir(SESSION_META_DIR, "session_meta")
            except Exception as e:
                print(f"[Firebase] sync error ({e}) → กลับไปโหมด Offline ชั่วคราว")
                self.online = False

            time.sleep(SYNC_IDLE_SEC)

    def _sync_dir(self, directory, firebase_node):
        """ไล่ sync ทุกไฟล์ .jsonl ในโฟลเดอร์ (แต่ละไฟล์ = ข้อมูลของ user 1 คน)"""
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if fname.endswith(".jsonl"):
                self._sync_file(os.path.join(directory, fname), firebase_node)

    def _sync_file(self, path, firebase_node):
        """Sync แบบ incremental — อ่านเฉพาะ 'บรรทัดใหม่' ที่ยังไม่เคย sync (นับจาก byte
        offset ที่จำไว้ใน self._cursors) แทนการอ่าน/เขียนทั้งไฟล์ทุกรอบเหมือนเดิม

        ของเดิมใช้ f.readlines() + f.writelines() ทั้งไฟล์ทุก ๆ 1 วินาที ซึ่งไฟล์ log
        สัญญาณ EMG (เขียนทุก sample ที่ ~500Hz โดย log_sample) จะโตแบบไม่มีเพดาน — เล่นไป
        แค่ 10 นาทีก็ ~300,000 บรรทัดแล้ว ยิ่งไฟล์ใหญ่ยิ่งอ่าน/เขียนช้าลงเรื่อยๆ แถมตอนเขียน
        ทับทั้งไฟล์ยังถือ self._lock ตัวเดียวกับที่ log_sample() ใช้ ทำให้เธรดอ่านสัญญาณ EMG
        (และทั้งแอป) ค้างรอจนกว่าจะเขียนไฟล์ยักษ์เสร็จ — นี่คือสาเหตุหลักที่แอปค้างบ่อยๆ

        วิธีใหม่: seek ไปที่ offset เดิม อ่านเฉพาะบรรทัดใหม่ทีละบรรทัด ส่งขึ้น Firebase
        เฉพาะ record ที่ cloud_sync=True แล้วขยับ offset ไปเรื่อยๆ ไม่ต้องเขียนไฟล์ log
        เดิมกลับเลย (ไฟล์ log ถูกเขียนแบบ append-only โดย _append_local เท่านั้น)"""
        try:
            size = os.path.getsize(path)
        except OSError:
            return  # ไฟล์ยังไม่ถูกสร้าง หรือถูกลบไปแล้ว

        offset = self._cursors.get(path, 0)
        if offset > size:
            # ไฟล์เล็กกว่า offset เดิม (ถูกลบ/สร้างใหม่) — เริ่มอ่านใหม่จากต้นไฟล์
            offset = 0
        if offset >= size:
            return  # ไม่มีข้อมูลใหม่ตั้งแต่รอบที่แล้ว

        pushed = 0
        cur_offset = offset
        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
            while pushed < SYNC_BATCH_SIZE:
                line = f.readline()
                if not line:
                    break  # อ่านถึงท้ายไฟล์ (เท่าที่มีตอนนี้)
                if not line.endswith("\n"):
                    # บรรทัดสุดท้ายยังเขียนไม่เสร็จ (ชนกับ log_sample ที่กำลัง append อยู่
                    # พอดี) — หยุดตรงนี้ก่อน ไม่ขยับ offset เกินบรรทัดที่ยังไม่สมบูรณ์
                    break
                line_end = f.tell()
                text = line.strip()
                if text:
                    try:
                        record = json.loads(text)
                    except json.JSONDecodeError:
                        record = None
                    if record and record.get("cloud_sync"):
                        payload = {k: v for k, v in record.items()
                                   if k not in ("synced", "cloud_sync")}
                        # ถ้า push() error (เช่นเน็ตหลุดกลางคัน) exception จะลอยขึ้นไปให้
                        # _sync_loop จับ — cur_offset ยังไม่ถูกขยับผ่านบรรทัดนี้ (อัพเดต
                        # อยู่บรรทัดถัดไป) รอบหน้าเลยจะ push record นี้ซ้ำใหม่โดยอัตโนมัติ
                        self.db.child(firebase_node).child(record["user_id"]).push(payload)
                        pushed += 1
                cur_offset = line_end

        if cur_offset != offset:
            self._cursors[path] = cur_offset
            _save_cursors(self._cursors)

    def stop(self):
        self._stop_flag.set()

    def list_users(self) -> list:
        """คืนรายชื่อ user_id ทั้งหมดที่รู้จัก — รวม 2 แหล่ง:
        1. node 'known_users' (ทะเบียนเบาๆ ที่แอปเขียนเองทุกครั้งที่มีการยืนยัน user ใหม่
           ผ่าน register_known_user — อ่านเร็วมาก ไม่ต้องโหลดข้อมูล sample จริง)
        2. สแกน node 'sessions' ตรงๆ เป็น fallback/merge เผื่อมี user เก่าที่บันทึกไว้
           ก่อนจะมีทะเบียน known_users (ยังไม่เคยถูกลงทะเบียน)

        ก่อนหน้านี้เคยลองใช้ shallow query (shallow=true) เพื่อประหยัด bandwidth แต่ไม่
        มั่นใจว่า pyrebase เวอร์ชันที่ติดตั้งจริงรองรับ method นี้หรือไม่ (ทดสอบไม่ได้เพราะ
        ไม่มี pyrebase ในเครื่องมือพัฒนา) — พบว่าเป็นสาเหตุที่ทำให้รายชื่อ user ไม่ขึ้นเลย
        แม้ Firebase จะมีข้อมูลอยู่จริง จึงเปลี่ยนมาใช้ .get()/.val() ธรรมดาแทน ซึ่งเป็น
        pattern เดียวกับที่ใช้อยู่แล้วทั่วทั้งโค้ด (เช่น ADMIN_LIST_PATH ใน auth_manager.py)
        พิสูจน์แล้วว่าใช้งานได้จริงแน่นอน"""
        if not (self.online and self.db is not None):
            return []
        users = set()

        try:
            result = self.db.child("known_users").get()
            val = result.val()
            if isinstance(val, dict):
                users.update(val.keys())
        except Exception as e:
            print(f"[Firebase] list_users() อ่าน known_users ไม่สำเร็จ: {e}")

        try:
            result = self.db.child("sessions").get()
            val = result.val()
            if isinstance(val, dict):
                users.update(val.keys())
        except Exception as e:
            print(f"[Firebase] list_users() สแกน sessions ไม่สำเร็จ: {e}")

        return sorted(users)

    def register_known_user(self, user_id: str):
        """เพิ่ม user_id เข้าทะเบียน 'known_users' (เขียนค่าเล็กๆ ครั้งเดียว idempotent —
        เขียนซ้ำก็ไม่เสียหาย) เพื่อให้ list_users() ครั้งต่อๆ ไปหา user คนนี้เจอเร็วขึ้น
        โดยไม่ต้องสแกน node 'sessions' ทั้งก้อน (ซึ่งอาจมีข้อมูลเป็นแสน record ต่อคน)
        เรียกจาก GUI ทุกครั้งที่มีการยืนยันรหัสผู้ใช้ (ดู _confirm_user_id)"""
        if not (self.online and self.db is not None):
            return
        try:
            self.db.child("known_users").child(user_id).set(True)
        except Exception as e:
            print(f"[Firebase] register_known_user('{user_id}') ไม่สำเร็จ: {e}")

    def delete_user_data(self, user_id: str) -> tuple[bool, str]:
        """ลบข้อมูลของ user คนนี้แบบเด็ดขาด ทั้งบน Firebase และในเครื่อง

        ทำไมลบผ่าน Firebase Console ไม่ได้ (สำหรับ user ที่มีข้อมูลเยอะ):
        Console เป็นหน้าเว็บที่ต้อง render ทุก child key ในเบราว์เซอร์ — ถ้า node มีข้อมูล
        เป็นหมื่นเป็นแสน record (เช่นจากตอนที่แอปเคย sync รัวๆ ก่อนแก้บั๊ก) เบราว์เซอร์จะ
        ค้าง Firebase เลยล็อกเป็น 'Read-only / non-realtime mode' อัตโนมัติกันเบราว์เซอร์
        พัง ทำให้กดลบไม่ได้เลย (เห็นข้อความ 'Select a key with fewer records')

        ฟังก์ชันนี้ใช้ pyrebase ยิง DELETE ตรงไปที่ REST API ของ Firebase แทน — เซิร์ฟเวอร์
        เป็นคนลบเอง ไม่ต้อง render อะไรในเบราว์เซอร์เลย จึงลบ node ใหญ่แค่ไหนก็ได้

        คืนค่า (success: bool, message: str)
        """
        errors = []
        firebase_deleted = False
        if PYREBASE_AVAILABLE and self.db is not None:
            try:
                self.db.child("sessions").child(user_id).remove()
                firebase_deleted = True
            except Exception as e:
                errors.append(f"sessions: {e}")
            try:
                self.db.child("calibrations").child(user_id).remove()
            except Exception as e:
                errors.append(f"calibrations: {e}")
            try:
                self.db.child("datasets").child(user_id).remove()
            except Exception as e:
                errors.append(f"datasets: {e}")
            try:
                self.db.child("known_users").child(user_id).remove()
            except Exception as e:
                errors.append(f"known_users: {e}")
        else:
            errors.append("ไม่ได้เชื่อมต่อ Firebase — ลบได้แค่ไฟล์ในเครื่อง")

        # ลบไฟล์ในเครื่อง + เอา cursor ของไฟล์นั้นออก กัน sync ย้อนกลับไปเจอไฟล์เก่าแล้วส่ง
        # ข้อมูลของ user ที่เพิ่งลบไปกลับขึ้น Firebase ซ้ำอีกรอบ (เป็นสาเหตุที่เคยลบใน
        # Console แล้วข้อมูลกลับมาใหม่)
        for path in (_session_path(user_id), _calib_path(user_id), _dataset_path(user_id)):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                errors.append(f"local file {path}: {e}")
            self._cursors.pop(path, None)
        _save_cursors(self._cursors)

        if errors:
            return False, "ลบไม่สำเร็จบางส่วน: " + "; ".join(errors)
        return True, f"ลบข้อมูลของ '{user_id}' ออกจาก Firebase และเครื่องนี้เรียบร้อยแล้ว"