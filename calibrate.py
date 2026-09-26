"""
calibrate.py
วัดค่า RMS จริงจากร่างกายผู้ใช้ แทนการเดา threshold เอง

อัพเดต: นอกจาก RMS แล้วยังคำนวณ MAV/Variance/Waveform Length/Zero Crossing ผ่าน sliding
window เดียวกับที่ emg_stream.py ใช้ตอน real-time (ดู emg_features.py ฟังก์ชันกลาง) แล้ว
บันทึกค่าเฉลี่ยของแต่ละ feature ต่อคลาส (REST/FIST) ลง calibration.json ไว้ด้วย เพื่อเป็น
ข้อมูลอ้างอิง (ไม่ได้ใช้คำนวณ threshold ของ servo โดยตรง — threshold ยังอิงจาก RMS เพียงตัว
เดียวเหมือนเดิม ตามหลักการเดิมของระบบ servo control ที่ไม่ต้องการเปลี่ยนพฤติกรรมเดิม)
"""

import json
import time
import statistics

from emg_reader import EMGReader
from emg_filters import EMGFilter
from emg_features import SlidingWindowFeatureExtractor, FEATURE_NAMES

FS = 500
CHANNEL = 0
BANDPASS = (20, 249)
NOTCH_FREQ = 50
CALIB_FILE = "calibration.json"

REST_SECONDS = 5
FIST_SECONDS = 5

# ต้องตรงกับดีฟอลต์ใน emg_stream.EMGStreamThread (feature_window_sec/feature_step_sec) —
# บันทึกลง calibration.json ด้วยเพื่อให้ปรับที่นี่ที่เดียวแล้วมีผลตอน real-time จริงด้วย
FEATURE_WINDOW_SEC = 0.2
FEATURE_STEP_SEC = 0.1

PRINT_EVERY_SEC = 0.1
PRINT_EVERY_SAMPLES = max(1, int(FS * PRINT_EVERY_SEC))


def collect_rms(reader, emg_filter, duration_sec, label):
    """เก็บ RMS (ต่อ sample เดี่ยวๆ ผ่าน EMGFilter.update_rms ตามเดิมทุกประการ — ใช้คำนวณ
    threshold ของ servo) พร้อมกับ feature 5 ตัวต่อ sliding window (ผ่าน
    SlidingWindowFeatureExtractor ตัวเดียวกับที่ emg_stream.py ใช้ตอน real-time)

    คืนค่า (rms_values, window_features) — window_features เป็น list ของ dict 5 ค่า
    (จำนวนน้อยกว่า rms_values มาก เพราะคำนวณทุก step ไม่ใช่ทุก sample — ดู emg_features.py)"""
    rms_values = []
    window_features = []
    sample_count = 0
    feature_extractor = SlidingWindowFeatureExtractor.from_seconds(
        fs=FS, window_sec=FEATURE_WINDOW_SEC, step_sec=FEATURE_STEP_SEC)

    def on_sample(voltage, t):
        nonlocal sample_count
        filtered, rectified = emg_filter.process_sample(voltage)
        rms = emg_filter.update_rms(rectified)
        rms_values.append(rms)
        sample_count += 1

        # ป้อนสัญญาณที่ผ่าน filter แล้ว (ไม่ใช่ rectified) เข้า feature window — เหตุผล
        # เดียวกับใน emg_stream.py: ต้องการสัญญาณที่แกว่งรอบ 0 จริง ให้ waveform_length/
        # zero_crossing มีความหมาย
        feats = feature_extractor.add_sample(filtered)
        if feats is not None:
            window_features.append(feats)

        if sample_count % PRINT_EVERY_SAMPLES == 0:
            print(f"    [{label}] sample #{sample_count}  raw={voltage:.5f}V  RMS={rms:.5f}V")

    print(f"\n>>> {label} เป็นเวลา {duration_sec} วินาที ... (เริ่มนับถอยหลัง)")
    for i in range(30, 0, -1):
        print(f"    เริ่มใน {i}...")
        time.sleep(1)
    print("    เริ่ม! กรุณาค้างท่าไว้จนกว่าจะจบ (จะโชว์ค่าดิบสดๆ ด้านล่าง)")

    reader.stream(fs=FS, callback=on_sample, duration=duration_sec)

    print(f"    เสร็จสิ้น เก็บได้ {len(rms_values)} samples ({len(window_features)} windows) ทั้งหมด")
    return rms_values, window_features


def _mean_features(window_features: list) -> dict:
    """เฉลี่ย feature แต่ละตัวข้าม window ทั้งหมดของคลาสนั้น — คืนค่า dict ว่างถ้าไม่มี
    window ไหนคำนวณเสร็จเลย (เช่น duration สั้นเกินไปจนไม่ครบแม้แต่ window เดียว)"""
    if not window_features:
        return {name: None for name in FEATURE_NAMES}
    return {name: statistics.mean(f[name] for f in window_features) for name in FEATURE_NAMES}


def main():
    print("=== EMG Calibration ===")
    print("ตรวจสอบก่อนเริ่ม:")
    print("  - Electrode แปะแนบผิวหนังแน่นดี ตรงแนวกล้ามเนื้อ")
    print("  - อยู่ห่างจากอุปกรณ์ไฟฟ้าที่อาจรบกวนสัญญาณ (UPS, adapter, จอมอนิเตอร์)")
    print("  - นั่งในท่าที่สบาย ไม่เกร็งกล้ามเนื้อส่วนอื่นโดยไม่ตั้งใจ")
    input("\nกด Enter เมื่อพร้อม...")

    reader = EMGReader(channel=CHANNEL, data_rate=860, gain=1)
    emg_filter = EMGFilter(fs=FS, bandpass=BANDPASS, notch_freq=NOTCH_FREQ)

    rest_rms, rest_windows = collect_rms(reader, emg_filter, REST_SECONDS, "ผ่อนคลายกล้ามเนื้อ / เปิดมือ")

    print("\nพักสัก 2 วินาทีก่อนไปขั้นถัดไป...")
    time.sleep(2)

    fist_rms, fist_windows = collect_rms(reader, emg_filter, FIST_SECONDS, "กำหมัดแน่นค้างไว้")

    def trim_warmup(values):
        cut = int(len(values) * 0.2)
        return values[cut:] if len(values) > cut else values

    rest_trimmed = trim_warmup(rest_rms)
    fist_trimmed = trim_warmup(fist_rms)

    rest_mean = statistics.mean(rest_trimmed)
    rest_max = max(rest_trimmed)
    fist_mean = statistics.mean(fist_trimmed)
    fist_min = min(fist_trimmed)

    # เฉลี่ย feature อีก 4 ตัว (นอกจาก rms) ต่อคลาส — ไม่ trim warmup แบบเดียวกับ rms เพราะ
    # window ตัวแรกๆ ถูกคำนวณช้ากว่าอยู่แล้ว (ต้องรอ buffer เต็มก่อน) มี lag ในตัวพอสมควรแล้ว
    rest_feat_mean = _mean_features(rest_windows)
    fist_feat_mean = _mean_features(fist_windows)

    print("\n=== ผลลัพธ์ ===")
    print(f"REST : mean={rest_mean:.5f}V  max={rest_max:.5f}V")
    print(f"FIST : mean={fist_mean:.5f}V  min={fist_min:.5f}V")
    print(f"REST feature เฉลี่ย: {rest_feat_mean}")
    print(f"FIST feature เฉลี่ย: {fist_feat_mean}")

    if fist_mean <= rest_mean:
        print("\n⚠️  คำเตือน: ค่า RMS ตอนกำหมัดไม่สูงกว่าตอนพักอย่างชัดเจน!")
        threshold = (rest_mean + fist_mean) / 2
    else:
        if fist_min > rest_max:
            threshold = (rest_max + fist_min) / 2
            print(f"\n✅ แยกโซนได้ชัดเจน (rest_max < fist_min)")
        else:
            threshold = (rest_mean + fist_mean) / 2
            print(f"\n⚠️  โซนมีทับซ้อนกันบ้าง ใช้ค่ากึ่งกลางของ mean แทน")

    print(f"\nThreshold ที่แนะนำ: {threshold:.5f} V")

    config = {
        "threshold": threshold,
        "rest_mean": rest_mean,
        "rest_max": rest_max,
        "fist_mean": fist_mean,
        "fist_min": fist_min,
        "fs": FS,
        "channel": CHANNEL,
        "bandpass": BANDPASS,
        "notch_freq": NOTCH_FREQ,
        "feature_window_sec": FEATURE_WINDOW_SEC,
        "feature_step_sec": FEATURE_STEP_SEC,
        # ค่าเฉลี่ย feature 5 ตัวต่อคลาส — เก็บไว้อ้างอิง/debug เท่านั้น (ไม่ได้ใช้คำนวณ
        # threshold ของ servo โดยตรง ซึ่งยังอิง RMS อย่างเดียวเหมือนเดิม)
        "rest_features_mean": rest_feat_mean,
        "fist_features_mean": fist_feat_mean,
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nบันทึกผลลง {CALIB_FILE} แล้ว")
    print("รัน main.py ได้เลย โปรแกรมจะโหลดค่านี้ไปใช้อัตโนมัติ")


if __name__ == "__main__":
    main()
