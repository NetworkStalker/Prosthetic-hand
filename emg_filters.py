"""
emg_filters.py
สร้าง filter สำหรับสัญญาณ EMG:
- Bandpass (ตัดความถี่ต่ำ/สูงที่ไม่ใช่ EMG จริง)
- Notch (ตัดสัญญาณรบกวนไฟบ้าน 50/60 Hz)

ใช้ scipy.signal ในการออกแบบ IIR filter (Butterworth) แล้วรันแบบ
real-time ทีละ sample ด้วย lfilter + zi (filter state) เพื่อไม่ต้อง
buffer ข้อมูลทั้งก้อนก่อนแล้วค่อย filter (จะหน่วงเวลา)
"""

import numpy as np
from scipy import signal


class EMGFilter:
    def __init__(self, fs=500, bandpass=(20, 249), notch_freq=50, notch_q=30):
        """
        fs: sample rate (Hz)
        bandpass: (low, high) ความถี่ตัดของ bandpass filter
                  - low ~20Hz ตัด motion artifact / DC offset
                  - high ควรน้อยกว่า fs/2 (Nyquist) เช่น fs=500 -> high <= 249
        notch_freq: ความถี่ไฟบ้านที่ต้องตัด (ไทย/ยุโรป = 50, สหรัฐ = 60)
        notch_q: quality factor ของ notch filter (ยิ่งสูง แถบตัดยิ่งแคบ)
        """
        self.fs = fs
        nyq = fs / 2.0

        # --- Bandpass filter (Butterworth, order 4) ---
        low, high = bandpass
        if high >= nyq:
            high = nyq - 1  # กันพลาดกรณี sample rate ต่ำเกินไป
        self.b_bp, self.a_bp = signal.butter(
            4, [low / nyq, high / nyq], btype="band"
        )
        self.zi_bp = signal.lfilter_zi(self.b_bp, self.a_bp)

        # --- Notch filter (IIR notch) ---
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, notch_q, fs)
        self.zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch)

        # ตัวแปรสำหรับ envelope (moving RMS)
        self._sq_buffer = []
        self.rms_window_size = max(1, int(fs * 0.1))  # หน้าต่าง RMS 100ms

    def process_sample(self, x):
        """
        รับ raw sample 1 ตัว -> คืนค่า (filtered_sample, rectified_sample)
        ใช้สำหรับ real-time streaming (เรียกทีละ sample)
        """
        # bandpass
        y, self.zi_bp = signal.lfilter(
            self.b_bp, self.a_bp, [x], zi=self.zi_bp
        )
        y_bp = y[0]

        # notch
        y2, self.zi_notch = signal.lfilter(
            self.b_notch, self.a_notch, [y_bp], zi=self.zi_notch
        )
        y_notch = y2[0]

        # rectify (full-wave)
        y_rect = abs(y_notch)

        return y_notch, y_rect

    def update_rms(self, rectified_sample):
        """
        อัปเดต moving RMS envelope ทีละ sample
        คืนค่า RMS ปัจจุบัน (ใช้เป็น 'กำลังกล้ามเนื้อ' สำหรับ threshold)
        """
        self._sq_buffer.append(rectified_sample ** 2)
        if len(self._sq_buffer) > self.rms_window_size:
            self._sq_buffer.pop(0)
        return float(np.sqrt(np.mean(self._sq_buffer)))
