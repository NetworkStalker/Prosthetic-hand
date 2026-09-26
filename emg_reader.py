"""
emg_reader.py
อ่านค่าจาก ADS1115 แบบ continuous conversion mode เพื่อให้ได้ sample rate
สูงและสม่ำเสมอ (ไม่ใช้ AnalogIn ปกติที่เป็น single-shot ซึ่งช้ากว่ามาก)

ต้องติดตั้งก่อน:
    pip install adafruit-circuitpython-ads1x15 --break-system-packages
"""

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn


# ADS1115 data rate ที่รองรับ (SPS) - เลือกค่าที่ >= ที่ต้องการใช้จริง
_SUPPORTED_RATES = [8, 16, 32, 64, 128, 250, 475, 860]


def _closest_supported_rate(target_sps):
    return min(_SUPPORTED_RATES, key=lambda r: abs(r - target_sps))


class EMGReader:
    def __init__(self, channel=0, address=0x48, data_rate=860, gain=1):
        """
        channel: 0-3 (A0-A3 บน ADS1115) ต่อกับ output ของ SEN0240
        data_rate: ความเร็วในการแปลงสัญญาณของ ADS1115 (SPS)
                   860 คือค่าสูงสุด ใช้เพื่อให้เก็บ sample ได้เร็วพอ
                   แล้วค่อยไป downsample/loop ที่ sample rate เป้าหมายทีหลัง
        gain: 1 = ช่วง +-4.096V (พอดีกับสัญญาณ EMG ที่ออก 0-3.3V)
        """
        i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(i2c, address=address)
        self.ads.gain = gain
        self.ads.data_rate = _closest_supported_rate(data_rate)

        self.chan = AnalogIn(self.ads, channel)

        # เปิด continuous conversion mode
        self.ads.mode = Mode.CONTINUOUS

    def read(self, retries=3, retry_delay=0.002):
        """คืนค่าแรงดัน (โวลต์) ของ sample ล่าสุด

        มี retry สำหรับ OSError 121 (Remote I/O error) ที่เกิดเป็นพักๆ
        จากสัญญาณรบกวนบนบัส I2C (เช่น จากมอเตอร์/servo ที่ทำงานพร้อมกัน)
        """
        last_err = None
        for attempt in range(retries):
            try:
                return self.chan.voltage
            except OSError as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        raise last_err

    def read_raw(self, retries=3, retry_delay=0.002):
        """คืนค่า raw ADC (16-bit signed) พร้อม retry เช่นเดียวกับ read()"""
        last_err = None
        for attempt in range(retries):
            try:
                return self.chan.value
            except OSError as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        raise last_err

    def stream(self, fs, callback, duration=None):
        """
        อ่านค่าต่อเนื่องที่ sample rate ประมาณ fs (Hz) แล้วเรียก callback(voltage)
        ทุกครั้งที่ได้ sample ใหม่

        fs: sample rate เป้าหมาย (Hz) เช่น 500
        callback: function(voltage: float, t: float) -> None
        duration: วินาทีที่จะรัน (None = รันไม่มีที่สิ้นสุด จนกว่าจะ Ctrl+C)
        """
        period = 1.0 / fs
        start = time.perf_counter()
        next_t = start

        while True:
            now = time.perf_counter()
            if now >= next_t:
                try:
                    v = self.chan.voltage
                except OSError:
                    # ข้าม sample นี้ไปถ้า I2C error ชั่วคราว (เช่น noise จากมอเตอร์)
                    # ไม่ทำให้ thread ทั้งเส้นตาย
                    next_t += period
                    continue
                callback(v, now - start)
                next_t += period

            if duration is not None and (now - start) >= duration:
                break
