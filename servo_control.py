"""
servo_control.py
ควบคุม Servo Motor ผ่าน PCA9685 (16-Channel PWM Driver) แบบ Real-time
ตามค่า RMS ของสัญญาณ EMG (ผ่าน update()) หรือตามผลทำนายของโมเดล ML multi-feature
(ผ่าน update_gesture() — เพิ่มใหม่ ดูด้านล่าง) — เลือกได้จาก GUI (self.servo_source_var
ใน prosthetic_gui.py) ดีฟอลต์ยังเป็น RMS Threshold เดิมเสมอ

Logic ตามที่ต้องการ (โหมด RMS Threshold — ดีฟอลต์):
  RMS >= threshold (default 0.020V)  → หมุนไปที่ angle_active (default 90°)
  RMS <  threshold                    → หมุนกลับไปที่ angle_rest   (default 0°)

โหมด ML Model (update_gesture): ตัดสินจาก gesture label ที่โมเดลทำนายมาตรงๆ แทนการเทียบ
threshold เอง (gesture == "FIST" → active, อื่นๆ → rest) ผ่าน debounce logic เดียวกัน

จะสั่งขยับ servo เฉพาะตอน "เปลี่ยนสถานะ" เท่านั้น (ไม่ยิงคำสั่งซ้ำทุก sample ที่ 500Hz)
เพื่อไม่ให้ยิง I2C ถี่เกินจำเป็น และมี debounce กันสัญญาณ/คำทำนายแกว่งใกล้จุดตัดสินใจ
ทำให้ servo สั่นไปมา (debounce ใช้ร่วมกันทั้ง 2 โหมด ดู _update_target_state)

รองรับ fallback โหมดจำลอง ถ้ารันบนเครื่องที่ไม่มี PCA9685 ต่ออยู่ (เช่น dev บน Windows)
เหมือนแนวทางเดียวกับ emg_stream.py — จะ print บอกแทนการสั่งมอเตอร์จริง

ติดตั้งก่อนใช้งานจริงบน Pi:
    pip install adafruit-circuitpython-servokit --break-system-packages
"""

import threading

try:
    from adafruit_servokit import ServoKit
    HARDWARE_AVAILABLE = True
except Exception:
    # ครอบคลุมทั้ง ImportError (ไม่มีไลบรารี) และ error อื่นๆ ตอนเปิด I2C บนเครื่องที่ไม่มี Hardware
    HARDWARE_AVAILABLE = False


class EMGServoController:
    def __init__(self, channel: int = 0, threshold: float = 0.020,
                 angle_active: float = 90, angle_rest: float = 0,
                 pca_channels: int = 16, debounce_samples: int = 10,
                 hysteresis_margin: float = 0.015):
        """
        channel:          channel บน PCA9685 ที่ต่อ servo (0-15) — ดีฟอลต์ 0 ตามที่ใช้งานอยู่
        threshold:        ค่า RMS (โวลต์) ที่จะสั่งขยับไป angle_active — ดีฟอลต์ 0.020V ตามที่วัดได้
        angle_active:      องศาตอนเกร็งกล้ามเนื้อ (RMS >= threshold)
        angle_rest:        องศาตอนพัก (RMS < release_threshold)
        debounce_samples:  ต้องมีค่าคงที่ต่อเนื่องกี่ sample ก่อนสั่งขยับจริง
                            (กัน servo สั่นตอนสัญญาณ RMS แกว่งใกล้ threshold พอดี)
                            ที่ fs=500Hz ค่า 10 = ต้องคงที่ต่อเนื่อง ~20ms ก่อนขยับจริง
        hysteresis_margin: ช่องว่าง (โวลต์ ค่าสัมบูรณ์ ไม่ใช่สัดส่วนของ threshold) ระหว่างเส้น
                            "กำมือ" กับเส้น "คลายมือ" — ดีฟอลต์ 0.015V หมายถึงต้องกำมือที่
                            RMS >= threshold แต่ต้อง RMS ลดลงต่ำกว่า (threshold - 0.015V) ถึง
                            จะยอมคลายมือ ถ้า RMS อยู่ระหว่างสองเส้นนี้จะค้างสถานะเดิมไว้ —
                            ใช้ค่าสัมบูรณ์แทนสัดส่วนของ threshold เพราะถ้า threshold ตั้งไว้
                            ต่ำ (ใกล้ noise floor) ช่องว่างแบบสัดส่วนจะเล็กจนไม่มีความหมาย
                            หมายเหตุ: ถ้า threshold เองยังต่ำกว่า/ใกล้ noise floor ของสัญญาณ
                            ตอนพัก hysteresis เพียงอย่างเดียวแก้ปัญหาไม่ได้เต็มที่ — ต้องปรับ
                            threshold ให้สูงห่างจาก noise floor จริงๆ ก่อน (ดู calibration.json
                            หรือ slider ในแอป)
        """
        self.channel = channel
        self.threshold = threshold
        self.hysteresis_margin = hysteresis_margin
        self.release_threshold = max(0.0, threshold - hysteresis_margin)
        self.angle_active = angle_active
        self.angle_rest = angle_rest
        self.debounce_samples = max(1, debounce_samples)

        self._lock = threading.Lock()
        self._current_state = "rest"
        self._pending_state = "rest"
        self._pending_count = 0

        self.hardware_mode = HARDWARE_AVAILABLE
        if self.hardware_mode:
            try:
                self.kit = ServoKit(channels=pca_channels)
                self.kit.servo[self.channel].angle = self.angle_rest
            except Exception as e:
                print(f"[EMGServoController] เปิด PCA9685 ไม่สำเร็จ ({e}) → ใช้โหมดจำลองแทน")
                self.hardware_mode = False

        if not self.hardware_mode:
            print(f"[EMGServoController] โหมดจำลอง (dev) — servo channel {self.channel} "
                  f"เริ่มที่ {self.angle_rest}° (threshold={self.threshold}V)")

    def update(self, rms: float):
        """
        เรียกทุก sample จาก EMGStreamThread callback (on_sample) — งานเบามาก ไม่บล็อก
        ยิงคำสั่งไป PCA9685 เฉพาะตอนสถานะเปลี่ยนจริง (ผ่าน debounce แล้ว) เท่านั้น

        ตัดสินสถานะเป้าหมายจาก RMS เทียบ threshold แบบ hysteresis (2 เส้น) แทนเส้นเดียว —
        ถ้า RMS อยู่ระหว่าง release_threshold กับ threshold ให้ค้างสถานะปัจจุบันไว้ก่อน
        ไม่ตัดสินใหม่ กันสัญญาณที่กำลังสลายตัวช้าๆ (ตอนคลายมือ) แกว่งผ่านเส้นเดียวซ้ำไปมา"""
        if rms >= self.threshold:
            target_state = "active"
        elif rms < self.release_threshold:
            target_state = "rest"
        else:
            target_state = self._current_state  # อยู่ในช่วง hysteresis — ค้างสถานะเดิม
        self._update_target_state(target_state)

    def update_gesture(self, gesture: str):
        """เหมือน update() ทุกประการ (ผ่าน debounce logic เดียวกัน กัน servo สั่นแบบเดียวกัน)
        แต่รับ gesture label ('FIST'/'OPEN') ตรงๆ แทนที่จะรับ RMS มาเทียบ threshold เอง —
        ใช้ตอนแหล่งสั่งงาน servo เป็นผลทำนายจากโมเดล ML (multi-feature) แทนที่จะเป็น RMS
        เทียบ threshold ตัวเดียวแบบเดิม (ดู prosthetic_gui.py: self.servo_source_var)

        เจตนาที่ให้ทั้ง 2 เมธอดนี้ใช้ debounce state ('_pending_state'/'_pending_count')
        ร่วมกันตัวเดียว: ถ้าแอปสลับไปมาระหว่างแหล่งสั่งงาน (เช่น ML ความมั่นใจต่ำ fallback
        ไปใช้ threshold เฉพาะบาง sample) servo จะยังนิ่งสม่ำเสมอ ไม่สั่นเพิ่มจากการสลับแหล่ง"""
        target_state = "active" if gesture == "FIST" else "rest"
        self._update_target_state(target_state)

    def _update_target_state(self, target_state: str):
        """Debounce logic ที่ใช้ร่วมกันทั้ง update() และ update_gesture() — ต้องมีค่าคงที่
        ต่อเนื่องกัน debounce_samples ครั้งก่อนถึงจะสั่งขยับ servo จริง (กันสัญญาณ/คำทำนาย
        แกว่งใกล้จุดตัดสินใจพอดี ทำให้ servo สั่นไปมา)"""
        with self._lock:
            if target_state == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state = target_state
                self._pending_count = 1

            if self._pending_count >= self.debounce_samples and target_state != self._current_state:
                self._current_state = target_state
                angle = self.angle_active if target_state == "active" else self.angle_rest
                self._set_angle(angle)

    def _set_angle(self, angle: float):
        if self.hardware_mode:
            self.kit.servo[self.channel].angle = angle
        print(f"[Servo] channel {self.channel} → {angle}°  (RMS state: {self._current_state})")

    def set_threshold(self, value: float):
        with self._lock:
            self.threshold = float(value)
            self.release_threshold = max(0.0, self.threshold - self.hysteresis_margin)

    def set_thresholds(self, threshold: float, release_threshold: float):
        """ตั้งเส้นบน (threshold/activate) กับเส้นล่าง (release_threshold) ตรงๆ พร้อมกัน —
        ใช้ตอนแอปคำนวณ 2 เส้นจากข้อมูล calibration จริงของ user แต่ละคน (ดู
        prosthetic_gui._suggest_threshold_from_calibration) แทนที่จะให้เส้นล่างคำนวณจาก
        threshold - hysteresis_margin คงที่เสมอไปแบบ set_threshold() เพียงอย่างเดียว"""
        with self._lock:
            self.threshold = float(threshold)
            self.release_threshold = max(0.0, float(release_threshold))
            self.hysteresis_margin = self.threshold - self.release_threshold
