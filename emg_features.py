"""
emg_features.py
ฟังก์ชันกลางสำหรับคำนวณ EMG Feature จากสัญญาณดิบ (ที่ผ่าน filter แล้ว) ภายใน 1 Sliding
Window — ให้ทุกส่วนของระบบ (emg_stream.py, calibrate.py, ml_online.py, prosthetic_gui.py)
เรียกใช้ฟังก์ชันเดียวกันเสมอ (ข้อกำหนดเรื่อง Code Refactoring: extract_features() ตัวกลาง)
เพื่อไม่ให้สูตรคำนวณเพี้ยนกันไปคนละที่คนละแบบ — แก้ปัญหาเดิมที่ระบบใช้แค่ RMS ตัวเดียว

Feature ที่คำนวณ (7 ตัว ต่อ 1 window):
    - RMS               Root Mean Square           = sqrt(mean(x^2))
    - MAV               Mean Absolute Value        = mean(|x|)
    - Variance          ความแปรปรวนของสัญญาณ         = mean((x - mean(x))^2)
    - Waveform Length   ความยาวคลื่นสะสม              = sum(|x[i] - x[i-1]|)
    - Zero Crossing     จำนวนครั้งที่สัญญาณตัดผ่าน 0    = นับ sign change (มี threshold กัน noise)
    - Slope Sign Change จำนวนครั้งที่ความชันกลับทิศ     = นับจุดยอด/จุดหุบ (มี threshold กัน noise)
    - IEMG              Integrated EMG              = sum(|x|)  (เหมือน MAV แต่ไม่หารเฉลี่ย)

ทุก feature คำนวณจาก "สัญญาณที่ผ่าน filter แล้ว" (bandpass+notch output คือ y_notch ใน
emg_filters.EMGFilter.process_sample — ไม่ใช่ rectified/RMS-envelope) เพราะต้องการสัญญาณที่
แกว่งรอบ 0 จริงๆ โดยเฉพาะ Zero Crossing/Slope Sign Change/Waveform Length ที่ความหมายจะ
เพี้ยนไปเลยถ้าใช้สัญญาณ rectified ที่เป็นบวกตลอด (ตัดผ่าน 0 ไม่มีทางเกิดขึ้น)

FEATURE_NAMES เรียงลำดับตายตัว — ใช้เป็น "สัญญา" (contract) เดียวกันทั้งระบบ ตั้งแต่ตอน
บันทึกลง Firebase จนถึงตอนสร้าง feature vector X ป้อนเข้าโมเดล ML ห้ามสลับลำดับเด็ดขาด
ไม่งั้นโมเดลจะเรียนรู้ผิด column (เช่น เอา zero_crossing ไปอยู่ตำแหน่งของ rms)
"""

import collections
import numpy as np

FEATURE_NAMES = ["rms", "mav", "variance", "waveform_length", "zero_crossing",
                  "slope_sign_change", "iemg"]

# โวลต์ — กันสัญญาณรบกวนเล็กๆ ใกล้ 0 ทำให้นับ zero-crossing เกินจริง ถ้าไม่มี threshold นี้
# สัญญาณ noise ที่แกว่งเบาๆ รอบ 0 (เช่นตอนกล้ามเนื้อพักสนิท) จะถูกนับเป็น "ตัดผ่าน 0" รัวๆ
# ทั้งที่ไม่ได้เกิดจากการหดตัวของกล้ามเนื้อจริง ทำให้ ZC สูงผิดปกติตอน rest
ZC_THRESHOLD = 0.005

# โวลต์ — threshold แบบเดียวกันแต่ใช้กับ Slope Sign Change (เปรียบเทียบขนาดของ "ผลต่างความชัน"
# ไม่ใช่ตัวสัญญาณเอง) กัน noise เล็กๆ ทำให้ความชันสลับทิศรัวๆ โดยกล้ามเนื้อไม่ได้ขยับจริง
SSC_THRESHOLD = 0.005


def extract_features(raw_window) -> dict:
    """คำนวณ Feature ทั้ง 7 ตัวจาก 1 sliding window ของสัญญาณ EMG ที่ผ่าน filter แล้ว
    (list/array/deque ของ float ความยาวเท่ากับ window_size)

    คืนค่า dict: {"rms":..., "mav":..., "variance":..., "waveform_length":...,
                  "zero_crossing":..., "slope_sign_change":..., "iemg":...}

    ถ้า window ว่างเปล่าคืนค่า 0 ทั้งหมด (edge case ตอนเพิ่งเริ่มสตรีมสัญญาณ) ถ้ามีแค่ 1
    sample จะคำนวณ rms/mav/variance/iemg ได้ปกติ แต่ waveform_length/zero_crossing เป็น 0
    (ต้องมีอย่างน้อย 2 samples ถึงจะมี "ผลต่างระหว่าง sample" ให้คำนวณ) ส่วน
    slope_sign_change ต้องมีอย่างน้อย 3 samples (ต้องมี "ผลต่างของผลต่าง" 2 ช่วงมาเทียบกัน)"""
    x = np.asarray(raw_window, dtype=float)
    n = len(x)
    if n == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    rms = float(np.sqrt(np.mean(x ** 2)))
    mav = float(np.mean(np.abs(x)))
    variance = float(np.var(x))
    iemg = float(np.sum(np.abs(x)))  # เหมือน MAV แต่ไม่หารด้วย n (ผลรวมสะสม ไม่ใช่ค่าเฉลี่ย)

    if n < 2:
        waveform_length = 0.0
        zero_crossing = 0
    else:
        diffs = np.diff(x)
        waveform_length = float(np.sum(np.abs(diffs)))
        # นับ zero crossing เฉพาะตอนสัญญาณสลับเครื่องหมาย (+/-) ระหว่าง sample ติดกัน
        # และทั้งสอง sample ต้องมีขนาดเกิน ZC_THRESHOLD ด้วย (กัน noise เล็กๆ ใกล้ 0)
        signs_ok = (np.abs(x[:-1]) > ZC_THRESHOLD) & (np.abs(x[1:]) > ZC_THRESHOLD)
        sign_change = (x[:-1] * x[1:]) < 0
        zero_crossing = int(np.sum(signs_ok & sign_change))

    if n < 3:
        slope_sign_change = 0
    else:
        # Slope Sign Change: นับจุดยอด/จุดหุบ — เทียบผลต่าง 2 ช่วงติดกัน (x[i]-x[i-1]) กับ
        # (x[i+1]-x[i]) ถ้าเครื่องหมายต่างกัน (สัญญาณกำลังขึ้นแล้วเปลี่ยนเป็นลง หรือกลับกัน)
        # นับเป็น 1 จุด — ต้องมีขนาดของทั้งสองผลต่างเกิน SSC_THRESHOLD ด้วย กัน noise
        d1 = x[1:-1] - x[:-2]   # ผลต่างช่วงก่อนหน้า (x[i]-x[i-1])
        d2 = x[2:] - x[1:-1]    # ผลต่างช่วงถัดไป   (x[i+1]-x[i])
        diffs_ok = (np.abs(d1) > SSC_THRESHOLD) & (np.abs(d2) > SSC_THRESHOLD)
        slope_change = (d1 * d2) < 0
        slope_sign_change = int(np.sum(diffs_ok & slope_change))

    return {
        "rms": rms,
        "mav": mav,
        "variance": variance,
        "waveform_length": waveform_length,
        "zero_crossing": zero_crossing,
        "slope_sign_change": slope_sign_change,
        "iemg": iemg,
    }


def features_to_vector(features: dict) -> list:
    """แปลง feature dict -> list เรียงลำดับตาม FEATURE_NAMES เสมอ (ใช้ตอนป้อนเข้าโมเดล ML
    เป็น X = [[rms, mav, variance, waveform_length, zero_crossing]]) รองรับ dict ที่ไม่มี
    key ครบทุกตัว (เช่น dataset เก่าที่ผ่าน backward-compat conversion) โดยเติม 0.0 แทน"""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


# ─── ประเภทบุคคล (Person Category) ────────────────────────────────────────────
# เพิ่มเข้ามาเพื่อให้โมเดลกำ-คลายมือ "รู้บริบทคนไข้" ด้วย ไม่ใช่แค่ตัดสินจากรูปร่างสัญญาณ
# EMG อย่างเดียว — คนแต่ละกลุ่มมีลักษณะสัญญาณกล้ามเนื้อต่างกัน (เช่น เด็ก/คนแก่มักมีแอมพลิจูด
# ต่ำกว่า, ผู้ป่วยกล้ามเนื้อ/ผู้พิการอาจมีรูปคลื่นผิดปกติกว่าคนทั่วไป) การให้โมเดลรู้ว่ากำลัง
# อ่านสัญญาณของคนกลุ่มไหนอยู่ (ผ่าน one-hot ต่อท้ายเวกเตอร์ feature เดิม) ช่วยให้โมเดลเดียว
# แยกแยะ FIST/OPEN ได้แม่นยำขึ้นข้ามกลุ่มคนไข้ที่หลากหลาย โดยไม่ต้องแยกเทรนคนละโมเดล
#
# ลำดับตายตัวเหมือน FEATURE_NAMES — ห้ามสลับ ไม่งั้น one-hot จะเข้าคอลัมน์ผิดตำแหน่ง
PERSON_CATEGORIES = ["normal", "child", "elderly", "muscle_patient", "disabled"]

# ป้ายภาษาไทยสำหรับแสดงผลใน GUI/รายงาน — key ต้องตรงกับ PERSON_CATEGORIES ทุกตัว
PERSON_CATEGORY_LABELS = {
    "normal":         "บุคคลปกติ",
    "child":          "เด็ก",
    "elderly":        "คนแก่",
    "muscle_patient": "ผู้ป่วยกล้ามเนื้อ",
    "disabled":       "ผู้พิการ",
}

DEFAULT_PERSON_CATEGORY = "normal"  # ใช้เมื่อไม่ได้ระบุ (เช่น record เก่าก่อนมีฟีเจอร์นี้)

# คอลัมน์ one-hot ต่อท้าย FEATURE_NAMES — นี่คือ "สัญญา" (contract) ของเวกเตอร์เต็มที่ป้อน
# เข้าโมเดลกำ-คลายมือจริงๆ (ดู full_feature_vector) เก็บไว้ให้ทุกจุดที่ต้องรู้จำนวน/ชื่อ
# คอลัมน์ (เช่นตอน export CSV) อ้างอิงจากที่เดียวกันเสมอ
PERSON_CATEGORY_FEATURE_NAMES = [f"person_{c}" for c in PERSON_CATEGORIES]
FULL_FEATURE_NAMES = FEATURE_NAMES + PERSON_CATEGORY_FEATURE_NAMES


def normalize_person_category(category) -> str:
    """แปลงค่า person_category ที่รับมา (อาจเป็น None/ค่าว่าง/พิมพ์ผิด) ให้เป็นหนึ่งใน
    PERSON_CATEGORIES เสมอ — fallback เป็น DEFAULT_PERSON_CATEGORY ถ้าไม่รู้จัก (เช่น
    record เก่าที่ไม่มี field นี้เลย หรือค่าที่หลุดมาจาก dataset ภายนอก)"""
    return category if category in PERSON_CATEGORIES else DEFAULT_PERSON_CATEGORY


def person_category_to_vector(category) -> list:
    """แปลงประเภทบุคคล -> one-hot list ความยาว len(PERSON_CATEGORIES) เรียงลำดับตาม
    PERSON_CATEGORIES เสมอ (ใช้ต่อท้าย features_to_vector ก่อนป้อนเข้าโมเดล)"""
    category = normalize_person_category(category)
    return [1.0 if category == c else 0.0 for c in PERSON_CATEGORIES]


def full_feature_vector(features: dict, person_category=None) -> list:
    """เวกเตอร์เต็มที่ป้อนเข้าโมเดลกำ-คลายมือจริงๆ = [7 ค่า feature ของสัญญาณ EMG] +
    [one-hot ประเภทบุคคล 5 ค่า] เรียงคอลัมน์ตาม FULL_FEATURE_NAMES เสมอ — ใช้ทั้งตอนเทรน
    (ml_online.OnlineTrainer) และตอนทำนายสด (predict/predict_with_confidence) ต้องเรียก
    ฟังก์ชันนี้จุดเดียวกันเสมอทั้งคู่ ไม่งั้นคอลัมน์จะไม่ตรงกัน"""
    return features_to_vector(features) + person_category_to_vector(person_category)


class SlidingWindowFeatureExtractor:
    """บัฟเฟอร์ sliding window แบบ efficient สำหรับใช้งาน real-time (เช่นบน Raspberry Pi)

    ป้อนสัญญาณเข้าทีละ sample ผ่าน add_sample() — คืนค่า feature dict เฉพาะตอนที่ window
    เพิ่งขยับครบ step_size sample ใหม่ (และ buffer เต็ม window_size แล้ว) เท่านั้น ไม่ใช่ทุก
    sample ที่ป้อนเข้ามา เพื่อลดภาระคำนวณซ้ำ (ข้อกำหนดเรื่อง Performance — ถ้าเรียก
    extract_features() ทุก sample ที่ 500Hz จะหนักเกินไปสำหรับ Raspberry Pi โดยเฉพาะ
    waveform_length/zero_crossing ที่ต้อง scan ทั้ง window)

    window_size / step_size หน่วยเป็นจำนวน sample (ไม่ใช่วินาที) — แปลงจากวินาทีเองตอน
    สร้าง instance ผ่าน classmethod from_seconds() ซึ่งเป็นทางที่ควรใช้ปกติ (ไม่ต้องรู้ fs
    เองตอนเรียก แค่บอกวินาทีที่ต้องการ)"""

    def __init__(self, window_size: int, step_size: int):
        self.window_size = max(1, int(window_size))
        self.step_size = max(1, int(step_size))
        self._buffer = collections.deque(maxlen=self.window_size)
        self._since_last_step = self.step_size  # ให้ window แรกคำนวณได้ทันทีที่ buffer เต็ม

    @classmethod
    def from_seconds(cls, fs: float, window_sec: float, step_sec: float):
        """สร้าง extractor จาก sample rate (Hz) + window/step เป็นวินาที — วิธีที่แนะนำให้ใช้
        ปกติ เช่น from_seconds(fs=500, window_sec=0.2, step_sec=0.1) = window 100 samples
        ขยับทีละ 50 samples (คำนวณ feature ใหม่ทุก 0.1 วิ จากข้อมูลย้อนหลัง 0.2 วิ)"""
        return cls(window_size=max(1, round(fs * window_sec)),
                    step_size=max(1, round(fs * step_sec)))

    def add_sample(self, x: float):
        """ป้อน sample ใหม่เข้า buffer คืนค่า feature dict ถ้า window เพิ่งขยับครบ step_size
        sample ใหม่ (และ buffer เต็ม window_size แล้ว) ไม่งั้นคืนค่า None — ผู้เรียกควรเก็บ
        ค่าล่าสุดที่ได้ไว้ใช้ต่อระหว่างที่ยังไม่มี window ใหม่ (ดู emg_stream.py)"""
        self._buffer.append(x)
        self._since_last_step += 1
        if len(self._buffer) < self.window_size:
            return None
        if self._since_last_step < self.step_size:
            return None
        self._since_last_step = 0
        return extract_features(self._buffer)

    def reset(self):
        self._buffer.clear()
        self._since_last_step = self.step_size
