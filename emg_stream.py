"""
emg_stream.py
เธรดอ่านสัญญาณ EMG แบบ Real-time ใช้ร่วมกันทั้ง main.py และ prosthetic_gui.py
เพื่อไม่ให้ตรรกะการอ่านค่า/กรองสัญญาณ ซ้ำซ้อนกันคนละที่คนละแบบ

โหมดการทำงาน:
  - HARDWARE MODE: ถ้ามีไลบรารี adafruit-blinka/ads1x15 (เช่นตอนรันบน Raspberry Pi 5
    ที่ต่อ ADS1115 + SEN0240 จริง) จะอ่านค่าจริงผ่าน emg_reader.EMGReader แล้วกรองผ่าน
    emg_filters.EMGFilter (Bandpass + Notch + Moving RMS)
  - SIMULATED MODE: ถ้ารันบนเครื่อง dev (เช่น PC ที่ไม่มี ADS1115 ต่ออยู่) จะ fallback
    เป็นสัญญาณจำลองอัตโนมัติ เพื่อให้พัฒนา/ทดสอบ GUI ต่อได้โดยไม่ต้องมีฮาร์ดแวร์

ทุก sample จะถูกส่งผ่าน callback on_sample(voltage, rms, gesture, t) ทันทีที่อ่านได้
(Real-time — ไม่มีการ buffer ทั้งก้อนก่อนค่อยประมวลผล) — signature ของ on_sample คงเดิม
ทุกประการ (ไม่ลบความสามารถเดิม) ผู้ที่เคยใช้งานอยู่แล้ว (main.py, prosthetic_gui.py เดิม)
ไม่ต้องแก้อะไรถ้าไม่สนใจ feature ใหม่

เพิ่มใหม่: Multi-feature extraction (RMS, MAV, Variance, Waveform Length, Zero Crossing)
คำนวณจาก Sliding Window ของสัญญาณที่ผ่าน filter แล้ว (ดู emg_features.py) — เก็บไว้ที่
self.latest_features (thread-safe ผ่าน get_latest_features()) อัพเดตทุกครั้งที่ window
ขยับครบ 1 step (ดีฟอลต์ window=0.2วิ / step=0.1วิ ปรับได้ผ่าน feature_window_sec/
feature_step_sec ตอนสร้าง instance หรือใส่ไว้ใน calibration.json) แยกออกจาก on_sample
โดยตั้งใจ — ไม่เพิ่มพารามิเตอร์ใหม่เข้า on_sample เพื่อไม่ให้กระทบโค้ดเดิมที่เรียกใช้อยู่

รองรับ pause()/resume() — ใช้ตอนอยากหยุดอ่านสัญญาณชั่วคราวเพื่อประหยัด CPU
(เช่น ตอนกำลังเทรน ML แบบ Offline จากไฟล์ ไม่จำเป็นต้องอ่านสัญญาณสดๆ ควบคู่ไปด้วย)
"""

import json
import math
import os
import random
import threading
import time

from emg_features import SlidingWindowFeatureExtractor, FEATURE_NAMES

try:
    from emg_reader import EMGReader
    from emg_filters import EMGFilter
    HARDWARE_AVAILABLE = True
except Exception:
    # ครอบคลุมทั้ง ImportError (ไม่มีไลบรารี) และ NotImplementedError
    # (เช่น busio.I2C ถูกเรียกบนเครื่องที่ไม่มี GPIO จริง)
    HARDWARE_AVAILABLE = False

CALIB_FILE = "calibration.json"


def load_calibration():
    """โหลดค่า config จาก calibration.json ถ้ามี (รัน calibrate.py มาก่อนแล้ว)"""
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


class _SimulatedSource:
    """สัญญาณจำลอง หน่วยใกล้เคียงโวลต์จริง (0–~0.4V) เพื่อให้ threshold เทียบกับของจริงได้

    read_one() คืนค่า (voltage, centered) — voltage คือค่าดิบเดิมทุกประการ (ใช้กับ
    rms/threshold/servo เหมือนเดิม ไม่กระทบพฤติกรรมเดิมเลย) ส่วน centered คือสัญญาณเดียวกัน
    แต่ลบ baseline DC ออก (ให้แกว่งรอบ 0) ใช้เฉพาะป้อนเข้า sliding-window feature extractor
    เท่านั้น เพราะ voltage ดิบเป็นบวกตลอด (ไม่มีทาง zero-crossing จริง) ถ้าป้อน voltage ตรงๆ
    เข้า feature window จะได้ zero_crossing=0 เสมอ ไม่มีความหมายอะไรเลยตอน dev/จำลอง"""

    def __init__(self):
        self.t = 0.0
        self.contracting = False

    def set_contracting(self, state: bool):
        self.contracting = state

    def read_one(self, dt):
        self.t += dt
        noise = random.gauss(0, 0.01)
        baseline = 0.30 if self.contracting else 0.03
        oscillation = (0.05 * math.sin(self.t * 3) if self.contracting
                       else 0.01 * math.sin(self.t * 0.5))
        voltage = max(0.0, baseline + oscillation + noise)
        centered = oscillation + noise  # เอาแค่ส่วนที่แกว่งรอบ 0 (ตัด baseline DC ออก)
        return voltage, centered


class EMGStreamThread(threading.Thread):
    """
    เธรดพื้นหลังอ่านค่า EMG ต่อเนื่องที่ sample rate fs (Hz)
    เรียก on_sample(voltage, rms, gesture, t) ทุกครั้งที่ได้ sample ใหม่

    หมายเหตุเรื่อง thread-safety: on_sample จะถูกเรียกจากเธรดนี้ ไม่ใช่ main thread
    ถ้าใช้กับ Tkinter ต้องเก็บค่าไว้ในตัวแปรที่มี lock แล้วให้ Tk อ่านผ่าน .after()
    (ห้ามสั่งอัปเดต widget ตรงๆ ใน on_sample)
    """

    def __init__(self, on_sample, fs=None, channel=None, bandpass=None,
                 notch_freq=None, threshold=None, feature_window_sec=None,
                 feature_step_sec=None):
        super().__init__(daemon=True)
        self.on_sample = on_sample
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()  # set = หยุดอ่านชั่วคราว (ไม่ทำลายเธรด)

        cfg = load_calibration()
        self.fs = fs or cfg.get("fs", 500)
        self.channel = channel if channel is not None else cfg.get("channel", 0)
        self.bandpass = tuple(bandpass or cfg.get("bandpass", (20, 249)))
        self.notch_freq = notch_freq or cfg.get("notch_freq", 50)
        self.threshold = threshold if threshold is not None else cfg.get("threshold", 0.05)

        # Multi-feature sliding window (RMS/MAV/Variance/Waveform Length/Zero Crossing) —
        # ดีฟอลต์ window=0.2วิ (100 samples ที่ 500Hz) / step=0.1วิ (คำนวณ feature ใหม่ทุก
        # 0.1วิ = 10Hz) อ่านจาก calibration.json ได้ถ้ามี (key: feature_window_sec/
        # feature_step_sec) ไม่งั้นใช้ดีฟอลต์นี้ — window สั้นพอที่จะยัง real-time responsive
        # บน Raspberry Pi แต่ยาวพอที่ waveform_length/zero_crossing จะมีความหมาย
        self.feature_window_sec = feature_window_sec or cfg.get("feature_window_sec", 0.2)
        self.feature_step_sec = feature_step_sec or cfg.get("feature_step_sec", 0.1)
        self._feature_extractor = SlidingWindowFeatureExtractor.from_seconds(
            fs=self.fs, window_sec=self.feature_window_sec, step_sec=self.feature_step_sec)
        self._features_lock = threading.Lock()
        self.latest_features = None  # dict 5 ค่า หรือ None ถ้ายังไม่มี window ไหนคำนวณเสร็จเลย

        self.hardware_mode = HARDWARE_AVAILABLE
        if self.hardware_mode:
            try:
                self.reader = EMGReader(channel=self.channel, data_rate=860, gain=1)
                self.emg_filter = EMGFilter(fs=self.fs, bandpass=self.bandpass,
                                            notch_freq=self.notch_freq)
            except Exception as e:
                print(f"[EMGStreamThread] เปิด Hardware ไม่สำเร็จ ({e}) → ใช้โหมดจำลองแทน")
                self.hardware_mode = False

        if not self.hardware_mode:
            self._sim = _SimulatedSource()
            print("[EMGStreamThread] ไม่พบ Hardware (ADS1115) → ใช้สัญญาณจำลอง (โหมด dev)")

    def set_threshold(self, value: float):
        self.threshold = float(value)

    def get_latest_features(self):
        """คืนค่า dict feature ล่าสุด (rms/mav/variance/waveform_length/zero_crossing) ที่
        คำนวณเสร็จจาก sliding window ล่าสุด — คืนค่า None ถ้ายังไม่มี window ไหนคำนวณเสร็จ
        เลย (เช่นเพิ่งเริ่มสตรีมสัญญาณ ข้อมูลยังไม่ครบ window_size) thread-safe เรียกจาก
        thread ไหนก็ได้ (เช่น main thread ของ Tkinter ผ่าน .after() ตามปกติ)"""
        with self._features_lock:
            return dict(self.latest_features) if self.latest_features is not None else None

    def set_simulated_contracting(self, state: bool):
        """ใช้เฉพาะตอนไม่มี Hardware จริง (เช่น ปุ่มสลับมือตอน dev บน PC)"""
        if not self.hardware_mode:
            self._sim.set_contracting(state)

    def reset_features(self):
        """เคลียร์ sliding window ของ feature extractor ทิ้ง — เรียกตอนสลับ user/เริ่ม
        session ใหม่ กันไม่ให้ window แรกของคนใหม่มีสัญญาณของคนเก่าปนอยู่"""
        with self._features_lock:
            self._feature_extractor.reset()
            self.latest_features = None

    def pause(self):
        """หยุดอ่านสัญญาณชั่วคราว (ประหยัด CPU) — ไม่ทำลายเธรด เรียก resume() เพื่อกลับมาอ่านต่อได้"""
        self._pause_flag.set()

    def resume(self):
        """กลับมาอ่านสัญญาณต่อหลังจาก pause()"""
        self._pause_flag.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause_flag.is_set()

    def stop(self):
        self._stop_flag.set()

    def run(self):
        period = 1.0 / self.fs
        start = time.perf_counter()
        next_t = start
        while not self._stop_flag.is_set():
            if self._pause_flag.is_set():
                # โหมด pause: sleep เบาๆ รอจนกว่าจะ resume() ไม่กินซีพียูโดยไม่จำเป็น
                time.sleep(0.05)
                next_t = time.perf_counter()  # กัน timestamp กระโดดพรวดตอน resume กลับมา
                continue

            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.001, next_t - now))
                continue

            if self.hardware_mode:
                voltage = self.reader.read()
                filtered, rectified = self.emg_filter.process_sample(voltage)
                rms = self.emg_filter.update_rms(rectified)
                # ป้อนสัญญาณที่ผ่าน filter แล้ว (แกว่งรอบ 0 จริง) เข้า feature window —
                # ไม่ใช้ rectified เพราะเป็นบวกตลอด จะทำให้ waveform_length/zero_crossing
                # ผิดความหมาย (ดู emg_features.py)
                feature_input = filtered
            else:
                voltage, centered = self._sim.read_one(period)
                rms = voltage  # โหมดจำลอง: ไม่มี filter จริง ใช้ค่าดิบแทน envelope (เดิม)
                feature_input = centered

            new_features = self._feature_extractor.add_sample(feature_input)
            if new_features is not None:
                with self._features_lock:
                    self.latest_features = new_features

            gesture = "FIST" if rms > self.threshold else "OPEN"
            # ครอบ try/except รอบ on_sample() ไว้โดยเจตนา — on_sample เป็น callback ของ
            # ฝั่ง GUI (เช่น prosthetic_gui.py) ที่อาจมีบั๊กได้ (เช่นเรียก method ผิด
            # signature ตอน sync ขึ้น Firebase) ถ้าไม่ครอบไว้ exception จากฝั่ง callback
            # จะหลุดขึ้นมาฆ่าเธรดอ่านสัญญาณทั้งเธรดแบบเงียบๆ (ไม่มี traceback โผล่ให้เห็น
            # ชัดเจนในบาง environment) ทำให้ค่า EMG ที่โชว์บนจอค้างนิ่งตลอดไปโดยไม่รู้สาเหตุ
            # — พฤติกรรมที่ถูกต้องคือ: log error แล้วอ่านสัญญาณต่อไปเรื่อยๆ ไม่ให้ฝั่ง GUI
            # ทำเธรดฮาร์ดแวร์พังได้
            try:
                self.on_sample(voltage, rms, gesture, now - start)
            except Exception as e:
                print(f"[EMGStreamThread] on_sample callback error ({e}) — "
                      f"เธรดยังอ่านสัญญาณต่อไปตามปกติ (ไม่ค้าง) แต่ sample นี้ไม่ถูกส่งต่อ")
            next_t += period
