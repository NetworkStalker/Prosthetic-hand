"""
event_recorder.py
บันทึก "เหตุการณ์กำมือ" (gesture event) แบบเป็นก้อนๆ อัตโนมัติ — ต่างจาก
firebase_sync.log_sample ที่บันทึกทุก sample ต่อเนื่องตลอดเวลา ตัวนี้จะรวบรวม
สัญญาณของการ "กำมือ 1 ครั้ง" ทั้งหมดไว้เป็นก้อนเดียว แล้วบันทึกทีเดียวตอนจบ

Logic ตามที่ต้องการ:
  1. เก็บ pre-roll buffer ของสัญญาณย้อนหลัง N วินาที (ดีฟอลต์ 5) ไว้ตลอดเวลา แม้ยังไม่
     เริ่มบันทึก event ใดๆ
  2. พอเจอ "กำมือ" (FIST) ครั้งแรก (จากสถานะ "คลายมือ") ให้เริ่ม event ใหม่ทันที โดยเอา
     pre-roll buffer ทั้งก้อน (สัญญาณ N วินาทีก่อนหน้า) มาเป็นจุดเริ่มต้นของ event ด้วย
     (ตามที่ขอ: "บันทึกตั้งแต่ 5 วิก่อนสัญญาณเปลี่ยน")
  3. บันทึกทุก sample ต่อเนื่องไปเรื่อยๆ ตราบใดที่ event ยังไม่จบ
  4. นับเวลา "คลายมือ" (OPEN) ต่อเนื่อง — ถ้าคลายมือครบ N วินาทีติดต่อกัน (ดีฟอลต์ 5)
     ถือว่า event จบแล้ว บันทึกเป็น "1 ครั้ง"
  5. แต่ถ้ากำมืออีกครั้งก่อนตัวนับคลายมือจะครบ N วินาที ให้ถือว่ายังเป็น event เดียวกัน
     อยู่ — รีเซ็ตตัวนับคลายมือ แล้วบันทึกต่อไปเรื่อยๆ (ไม่ตัด event ใหม่)

Thread-safety: ออกแบบให้เรียก add_sample() จากเธรดอ่านสัญญาณ EMG (EMGStreamThread)
โดยตรงได้เลย ไม่ต้องมี lock เพิ่มจากฝั่งผู้เรียก — ป้องกัน race condition ภายในตัวเองแล้ว
"""

import collections
import threading

from emg_features import FEATURE_NAMES


class GestureEventRecorder:
    def __init__(self, pre_roll_sec: float = 5.0, release_hold_sec: float = 5.0,
                 on_event_saved=None, enabled: bool = False):
        """
        pre_roll_sec: เก็บสัญญาณย้อนหลังกี่วินาทีก่อนกำมือ ใส่เข้า event ด้วย (ดีฟอลต์ 5)
        release_hold_sec: ต้องคลายมือต่อเนื่องกี่วินาทีถึงจะถือว่า event จบ (ดีฟอลต์ 5)
        on_event_saved: callback(event: dict) เรียกตอน event จบ (รันจากเธรดเดียวกับที่
                        เรียก add_sample() เข้ามา — ถ้าจะแตะ Tk widget ต้อง marshal เข้า
                        main thread เอง เช่นผ่าน tk_root.after(0, ...))
        enabled: เปิดใช้งานทันทีหรือไม่ (ดีฟอลต์ False — ต้องเปิดเองจาก GUI เสมอ ตาม
                หลักการเดียวกับฟีเจอร์บันทึก real-time อื่นๆ ในแอปนี้ที่ไม่ auto-write
                ขึ้น Firebase โดยที่ผู้ใช้ยังไม่ได้กดอะไรเลย)
        """
        self.pre_roll_sec = pre_roll_sec
        self.release_hold_sec = release_hold_sec
        self.on_event_saved = on_event_saved
        self.enabled = enabled
        self.n_events_recorded = 0

        self._lock = threading.Lock()
        # pre-roll buffer: (t, voltage, rms, gesture, wall_time) ย้อนหลัง pre_roll_sec
        # วินาทีเสมอ (ตัดตาม timestamp จริง ไม่ใช่จำนวน sample ตายตัว เพราะ sample rate
        # เปลี่ยนไปตามโหมด hardware/จำลอง)
        self._preroll = collections.deque()

        self._recording = False
        self._event_samples: list = []
        self._event_start_wall = None
        self._release_started_at = None  # timestamp (t) ที่เริ่มคลายมือต่อเนื่องล่าสุด

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def reset(self):
        """ยกเลิก event ที่กำลังบันทึกอยู่ (ถ้ามี) แล้วเคลียร์ pre-roll buffer — ใช้ตอน
        ผู้ใช้ปิดฟีเจอร์นี้กลางคันหรือเปลี่ยน user"""
        with self._lock:
            self._preroll.clear()
            self._recording = False
            self._event_samples = []
            self._event_start_wall = None
            self._release_started_at = None

    def add_sample(self, t: float, voltage: float, rms: float, gesture: str, wall_time,
                    features: dict = None):
        """เรียกทุก sample จาก EMGStreamThread callback
        t: timestamp วินาที (monotonic เทียบกับตอนเริ่ม stream)
        wall_time: datetime.now() ของ sample นี้ (ใช้บันทึกเวลาเริ่ม/จบ event จริง)
        features: dict 5 ค่า (rms/mav/variance/waveform_length/zero_crossing) จาก sliding
        window ล่าสุด (emg_features.extract_features) — optional เพื่อ backward-compat กับ
        โค้ดเดิมที่เรียก add_sample() แบบไม่มี feature พวกนี้ อาจเป็น None ได้ (window ยังไม่
        เต็ม) จะถูกใส่ไว้ใน event['samples'] ให้ครบถ้ามี ไม่งั้นเว้นว่างไว้เหมือนเดิม"""
        if not self.enabled:
            return

        with self._lock:
            self._preroll.append((t, voltage, rms, gesture, wall_time, features))
            cutoff = t - self.pre_roll_sec
            while self._preroll and self._preroll[0][0] < cutoff:
                self._preroll.popleft()

            if not self._recording:
                if gesture != "FIST":
                    return
                # เริ่ม event ใหม่ — เอา pre-roll buffer ทั้งก้อน (N วิ ก่อนหน้า) มาเป็น
                # จุดเริ่มต้นของ event เลย ตามที่ขอ
                self._recording = True
                self._event_samples = list(self._preroll)
                self._event_start_wall = self._preroll[0][4] if self._preroll else wall_time
                self._release_started_at = None
                return

            self._event_samples.append((t, voltage, rms, gesture, wall_time, features))

            if gesture == "OPEN":
                if self._release_started_at is None:
                    self._release_started_at = t
                elif t - self._release_started_at >= self.release_hold_sec:
                    self._finalize_event_locked(wall_time)
            else:  # FIST — ยังไม่ปล่อยมือจริง รีเซ็ตตัวนับคลายมือ (ยังไม่จบ event)
                self._release_started_at = None

    def _finalize_event_locked(self, end_wall_time):
        """เรียกจากภายใน add_sample() เท่านั้น (ถือ self._lock อยู่แล้ว)"""
        samples = self._event_samples
        start_wall = self._event_start_wall

        self._recording = False
        self._event_samples = []
        self._release_started_at = None
        self._event_start_wall = None
        self.n_events_recorded += 1

        event = {
            "start_time": start_wall.isoformat() if start_wall else None,
            "end_time": end_wall_time.isoformat(),
            "n_samples": len(samples),
            "samples": [
                {"t": round(t, 4), "voltage": round(v, 6), "rms": round(r, 6), "gesture": g,
                 # ใส่ feature อื่นๆ ทั้งหมดจาก FEATURE_NAMES แบบ generic (ไม่ hardcode ชื่อ
                 # key ตรงนี้) กัน bug แบบที่เคยเกิด: ตอนเพิ่ม slope_sign_change/iemg เข้า
                 # emg_features.py แล้วลืมแก้จุดนี้ตาม ทำให้ 2 feature ใหม่หายไปจาก event
                 # เงียบๆ โดยไม่มี error เตือนเลย — ใช้ FEATURE_NAMES เป็นแหล่งความจริงเดียว
                 **({k: (round(feats[k], 8) if isinstance(feats[k], float) else int(feats[k]))
                     for k in FEATURE_NAMES if k != "rms"} if feats else {})}
                for t, v, r, g, _, feats in samples
            ],
        }
        if self.on_event_saved:
            self.on_event_saved(event)
