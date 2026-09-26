"""
ml_online.py
เทรนโมเดลจำแนกท่าทาง EMG (FIST/OPEN) แบบ Online — เทรนอัตโนมัติทันทีที่เลือกโมเดล
ไม่ต้องกดปุ่มเริ่มเทรนเอง แล้วเทรนต่อเนื่องทุกครั้งที่มี sample ใหม่เข้ามา

Feature ที่ใช้ (7 ตัว ต่อ sample — เดิมมีแค่ RMS ตัวเดียว): RMS, MAV, Variance,
Waveform Length, Zero Crossing, Slope Sign Change, IEMG (จากสัญญาณ EMG จริงล้วนๆ ดู
emg_features.py สำหรับสูตรคำนวณกลาง) รวมเป็น X ที่ป้อนเข้าโมเดลคือ
X = features_to_vector(features) เรียงคอลัมน์ตาม emg_features.FEATURE_NAMES เสมอ

⚠️ อัพเดต: เอา person_category (one-hot ประเภทบุคคล 5 คอลัมน์) ออกจาก feature vector ที่
ป้อนเข้าโมเดลแล้ว ตามที่ต้องการให้โมเดลตัดสิน FIST/OPEN จากรูปคลื่นสัญญาณ EMG ล้วนๆ
เท่านั้น ไม่ผสมบริบทประเภทบุคคลเข้าไปด้วย — add_sample()/predict()/predict_with_confidence()
ยังรับพารามิเตอร์ person_category ไว้เหมือนเดิม (ไม่ตัดออกจาก signature) เพื่อ backward-
compat กับโค้ดฝั่งเรียก (prosthetic_gui.py) แต่จะไม่ถูกนำไปคำนวณ vector อีกต่อไป ส่วน
label (y) ยังเป็น gesture (FIST/OPEN) เหมือนเดิม

Train / Validate / Test Split: แบ่งข้อมูลที่เข้ามาแบบสุ่มตั้งแต่รับเข้ามาเลย เป็น
train 70% / validate 10% / test 20% (ปรับสัดส่วนได้ผ่าน test_ratio/val_ratio ตอนสร้าง
OnlineTrainer) — train ใช้เทรนโมเดลตรงๆ, validate ใช้ติดตาม train vs validate
accuracy/error ต่อรอบเทรน (ดู OnlineTrainer.history สำหรับกราฟ loss/performance),
test เป็น holdout แท้ๆ ที่ไม่เคยถูกใช้ทั้งตอนเทรนและตอนดู validate เลย ใช้รายงานผลสุดท้าย
(accuracy/precision/recall/f1/auc/kappa/confusion matrix — ดู _evaluate())

Reproducibility: การแบ่ง train/validate/test ใช้ random.Random(split_seed) instance
แยกต่างหาก (ดีฟอลต์ split_seed=42) แทนการเรียก random module ระดับ global ตรงๆ แบบเดิม
(ของเดิมทำให้ผลลัพธ์ไม่นิ่งทุกครั้งที่รันใหม่ เพราะ global random ไม่ได้ตั้ง seed ตายตัวไว้
เลย) — ป้อนข้อมูลชุดเดิมตามลำดับเดิมซ้ำแล้วซ้ำอีก จะได้การแบ่ง train/val/test ชุดเดียวกัน
ทุกครั้ง ทำให้ผลลัพธ์ของแต่ละโมเดลนิ่งพอที่จะเอาไปสรุป/เปรียบเทียบกันได้จริง

Feature Scaling: ใช้ StandardScaler ของ scikit-learn (แทนตัวคูณ _FEATURE_SCALE แบบเดิมที่
ใช้ได้กับ RMS ตัวเดียวเท่านั้น) — fit จาก training data เท่านั้น (ไม่ใช้ test data มา fit
เพื่อกัน data leakage) แล้วใช้ scaler ตัวเดียวกันทั้งตอน train/test/predict เสมอ (ดู
OnlineTrainer.scaler) เพราะ 7 feature มีหน่วย/สเกลต่างกันมาก (เช่น zero_crossing เป็นเลข
จำนวนเต็มหลักสิบ ในขณะที่ variance เป็นเลขทศนิยมหลัก 1e-4) โมเดลที่ไวต่อสเกล (SVM/ANN/k-NN)
จะทำงานผิดเพี้ยนถ้าไม่ normalize ให้อยู่หน่วยเดียวกันก่อน

── โมเดลที่รองรับ ─────────────────────────────────────────────────────────────
Base model เดี่ยว: ANN, k-NN, SVM, Random Forest, C4.5 Decision Tree, Random Tree,
Naive Bayes, XGBoost, CatBoost (2 ตัวหลังต้องติดตั้งไลบรารีเพิ่มเอง — ดูด้านล่าง)

Ensemble wrapper 3 แบบ (รวมกับ base model ไหนก็ได้): None, Bagging, AdaBoost

⚠️ ก่อนหน้านี้เคยมี REPTree, LADTree, MultiBoosting ด้วย แต่เป็นอัลกอริทึมเฉพาะของ Weka
ที่ scikit-learn ไม่มีให้ใช้จริง (ของเดิมเป็นแค่ "ตัวประมาณใกล้เคียง" ไม่ใช่ของแท้) จึง
เอาออกทั้งหมด — ตอนนี้ทุกโมเดล/ensemble ในรายการเป็นของจริงล้วนๆ ไม่มีตัวประมาณอีกต่อไป
- 7 ตัวจาก scikit-learn ตรงๆ (ANN, k-NN, SVM, Random Forest, C4.5, Random Tree, Naive Bayes)
- Bagging/AdaBoost จาก scikit-learn ตรงๆ
- XGBoost/CatBoost เป็นไลบรารีแยกที่ implement sklearn-compatible API (ไม่ใช่ของประมาณ
  เป็นอัลกอริทึม gradient boosting ของจริงจากไลบรารีต้นฉบับ) ติดตั้งด้วย:
      pip install xgboost catboost --break-system-packages
  ถ้าไม่ได้ติดตั้ง จะไม่โผล่ในรายการเลือกโมเดล (ไม่ error ไม่ crash)

── สถาปัตยกรรม Real-time ──────────────────────────────────────────────────────
เทรนหนักๆ (โดยเฉพาะ ensemble ที่ห่อโมเดลแพงๆ ซ้อนกัน) รันอยู่ใน background thread
แยกจาก GUI thread เสมอ เพื่อไม่ให้หน้าจอค้างเวลาโมเดลใช้เวลาเทรนนาน — GUI แค่ป้อน
sample เข้าคิว แล้วโพลผลลัพธ์ (accuracy ฯลฯ) มาแสดงเป็นระยะ ไม่ต้องรอเทรนเสร็จ
"""

import random
import threading
import time
import warnings

import numpy as np
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                              precision_score, recall_score, roc_auc_score)

from emg_features import FEATURE_NAMES, features_to_vector

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[ml_online] ไม่ได้ติดตั้ง xgboost → ไม่มีตัวเลือก XGBoost ใน Machine Learning tab")
    print("             ติดตั้งด้วย: pip install xgboost --break-system-packages")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[ml_online] ไม่ได้ติดตั้ง catboost → ไม่มีตัวเลือก CatBoost ใน Machine Learning tab")
    print("             ติดตั้งด้วย: pip install catboost --break-system-packages")

# ANN (MLPClassifier) ถูกเทรนซ้ำบ่อยๆ แบบเร็วๆ เพื่อ real-time ซึ่งจะไม่ทันลู่เข้าเต็มที่
# ทุกรอบ (max_iter จำกัดไว้ให้เร็ว) เป็นทางเลือกที่ตั้งใจไว้ (เร็ว > แม่นสมบูรณ์แบบทุกรอบ)
# จึงปิด warning นี้เพื่อไม่ให้ console รกโดยไม่จำเป็น
warnings.filterwarnings("ignore", category=ConvergenceWarning)

CLASSES = np.array(["OPEN", "FIST"])
POS_LABEL = "FIST"

BASE_MODELS = ["ANN", "k-NN", "SVM", "Random Forest", "C4.5 Decision Tree",
               "Random Tree", "Naive Bayes"]
if XGBOOST_AVAILABLE:
    BASE_MODELS.append("XGBoost")
if CATBOOST_AVAILABLE:
    BASE_MODELS.append("CatBoost")

ENSEMBLE_METHODS = ["None", "Bagging", "AdaBoost"]

# โมเดลที่รองรับ partial_fit จริง (online จริงๆ) — ใช้ได้เฉพาะตอน ensemble = "None"
# เพราะ Bagging/AdaBoost ใน scikit-learn ไม่มี partial_fit ให้เลย
# หมายเหตุ: ANN (MLPClassifier) ไม่ได้อยู่ในกลุ่มนี้ แม้ scikit-learn จะมี partial_fit ให้ก็ตาม
# เพราะทดสอบแล้วพบว่าเรียก partial_fit ซ้ำๆ ด้วยข้อมูลสะสม ทำให้ผลแย่ลงเรื่อยๆ (ไม่ลู่เข้า)
# เมื่อเทียบกับ fit() ปกติ จึงให้ ANN ใช้ full-refit เหมือน Random Forest แทนเพื่อความถูกต้อง
# XGBoost/CatBoost ก็ไม่มี partial_fit สไตล์ sklearn ให้เช่นกัน จึงใช้ full-refit เหมือนกัน
PARTIAL_FIT_BASE_ONLY = {"SVM", "Naive Bayes"}

# ป้ายกำกับที่มาจากปุ่มในแท็บ Calibration Tool (ภาษาไทย) → normalize เป็น FIST/OPEN
_FIST_ALIASES = {"FIST", "กำมือ", "กำมือ ✊", "กำมือ✊"}

# หมายเหตุ: เดิมมีตัวคูณ _FEATURE_SCALE = 100.0 สำหรับ RMS ตัวเดียว (โวลต์เล็กเกินไปสำหรับ
# gradient-based model) ตอนนี้แทนที่ด้วย StandardScaler เต็มรูปแบบแล้ว (ดู OnlineTrainer.scaler)
# เพราะ 7 feature มีสเกล/หน่วยต่างกันมาก คูณเลขคงที่ตัวเดียวไม่พอ ต้อง normalize แยกแต่ละ
# feature ตามค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานของมันเอง
_PARTIAL_FIT_WINDOW = 20  # จำนวน sample ล่าสุดที่ใช้ partial_fit ต่อรอบ (นอกจากรอบแรก)


def _normalize_gesture(gesture: str) -> str:
    return "FIST" if gesture in _FIST_ALIASES else "OPEN"


_XGB_SORTED_CLASSES = np.array(sorted(["FIST", "OPEN"]))  # ["FIST", "OPEN"] เรียงตามตัวอักษร
# เหมือนที่ sklearn.preprocessing.LabelEncoder ใช้ภายใน — สำคัญเพราะ BaggingClassifier จะเข้า
# รหัส y เป็น int (0/1) ให้เองก่อนส่งเข้า estimator.fit() ทุกครั้ง (ต่างจาก AdaBoostClassifier ที่
# ส่ง y แบบ string ดิบๆ ไม่แปลงเอง) ต้องเข้ารหัสด้วยลำดับตัวอักษรเดียวกันเป๊ะๆ ไม่งั้นตอน Bagging
# รวมผล predict_proba จากแต่ละ estimator จะตีความคอลัมน์สลับความหมายกัน (FIST<->OPEN) แบบเงียบๆ
# โดยไม่มี error ให้เห็นเลย (ผลลัพธ์ผิดแต่ไม่ crash — อันตรายกว่า error ที่เห็นชัดๆ อีก)


class _XGBStringLabelClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper รอบ XGBClassifier แก้ปัญหา XGBoost (>= 1.6) ไม่รับ label เป็น string
    ('FIST'/'OPEN') ตรงๆ เหมือนโมเดลอื่นทุกตัวใน scikit-learn อีกต่อไป — ตั้งแต่เวอร์ชัน 1.6
    เป็นต้นมา XGBClassifier บังคับให้ y ต้องเป็นจำนวนเต็มเรียงจาก 0 เท่านั้น (เดิมมี
    use_label_encoder=True คอย auto-encode ให้ แต่ param นี้ถูกลบไปแล้ว) ถ้าป้อน string label
    ตรงๆ จะโดน 'ValueError: Invalid classes inferred from unique values of y' ทุกครั้งที่เทรน —
    นี่คือสาเหตุที่ XGBoost ไม่เคยมีค่า accuracy ขึ้นเลย (error เงียบๆ ถูกจับไว้ใน
    OnlineTrainer._worker_loop เก็บแค่ last_error/n_errors ไม่มี exception โผล่ให้เห็นตรงๆ)
    และ AdaBoost/Bagging + XGBoost ก็ error เหมือนกันทุกรอบ boosting/bagging เพราะสุดท้ายก็เรียก
    estimator.fit(X, y) ด้วย y (string) ตัวเดิมอยู่ดี

    wrapper นี้ทำหน้าที่ map string label -> int ก่อนป้อนเข้า XGBClassifier จริง แล้ว map
    ผลลัพธ์กลับเป็น string ตอน predict()/predict_proba() เพื่อให้โค้ดส่วนอื่นทั้งหมด (evaluate,
    predict_with_confidence, AdaBoost/Bagging ที่ห่อโมเดลนี้อยู่) ยังเห็นเป็น string label
    'FIST'/'OPEN' เหมือนโมเดลอื่นทุกตัว ไม่ต้องแก้โค้ดจุดอื่นเลย — สืบทอดจาก BaseEstimator/
    ClassifierMixin เพื่อให้ get_params()/set_params()/clone() ทำงานถูกต้อง (จำเป็นเพราะ
    OnlineTrainer._worker_loop เรียก sklearn.base.clone(self.model) ทุกรอบเทรนใหม่)"""

    def __init__(self, n_estimators=40, max_depth=4, learning_rate=0.3,
                 eval_metric="logloss", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.eval_metric = eval_metric
        self.random_state = random_state

    def _build(self):
        return XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                              learning_rate=self.learning_rate, eval_metric=self.eval_metric,
                              random_state=self.random_state)

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = _XGB_SORTED_CLASSES.copy()
        if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
            # เจอกรณีนี้ตอนถูกห่อด้วย BaggingClassifier เท่านั้น — Bagging เข้ารหัส y เป็น
            # int (0/1) ให้เองแล้วด้วย LabelEncoder ก่อนเรียก estimator.fit() (เรียงตัวอักษร
            # เหมือนกับ _XGB_SORTED_CLASSES ข้างบนพอดี) ใช้ตรงๆ ได้เลยไม่ต้อง map ซ้ำ
            y_int = y.astype(int)
        else:
            # AdaBoostClassifier ส่ง y แบบ string ('FIST'/'OPEN') ดิบๆ ไม่แปลงเอง (และตอน
            # เรียกใช้ตรงๆ ไม่ผ่าน ensemble ใดๆ ก็เป็น string เช่นกัน) — map เองตรงนี้
            label_to_idx = {label: i for i, label in enumerate(self.classes_)}
            y_int = np.array([label_to_idx[v] for v in y])
        self.model_ = self._build()
        if sample_weight is not None:
            # AdaBoost ส่ง sample_weight ที่ normalize รวมกันเป็น 1 เสมอ (ค่าเฉลี่ยต่อ sample
            # เล็กมาก เช่น ~0.008 ตอนมี 120 samples) แต่ XGBoost ตีความ sample_weight เป็น
            # "น้ำหนักจริง" ที่ไปเข้าสูตร min_child_weight ของแต่ละใบ (ค่าเริ่มต้น min_child_weight=1
            # หมายถึงผลรวม hessian ของ sample ในใบนั้นต้องถึง 1) ถ้าน้ำหนักรวมทั้งชุดมีแค่ 1 จะไม่มี
            # ใบไหน split ได้เลยสักใบ (ผลรวมไม่ถึง threshold) ต้นไม้เลยจบที่ root node ทำนายมั่วๆ
            # ทันที (วัดได้จริง: accuracy ตกจาก 100% เหลือ ~52% แค่เพราะน้ำหนักถูก normalize)
            # แก้โดย rescale ให้ค่าเฉลี่ยน้ำหนักต่อ sample = 1 เสมอ (คงสัดส่วนความสำคัญระหว่าง
            # sample ที่ AdaBoost ต้องการเป๊ะ แค่ปรับสเกลรวมให้ไม่ชนกับ min_child_weight ของ XGBoost)
            sw = np.asarray(sample_weight, dtype=float)
            sw_total = sw.sum()
            if sw_total > 0:
                sw = sw * (len(sw) / sw_total)
            self.model_.fit(X, y_int, sample_weight=sw)
        else:
            self.model_.fit(X, y_int)
        return self

    def predict(self, X):
        pred_int = self.model_.predict(np.asarray(X))
        return self.classes_[pred_int]

    def predict_proba(self, X):
        # คอลัมน์เรียงตาม self.classes_ (= CLASSES) เสมอ เพราะ label_to_idx ใน fit() ผูกกับ
        # CLASSES ตรงๆ ไม่ใช่ np.unique(y) ของรอบนั้นๆ
        return self.model_.predict_proba(np.asarray(X))


def _make_base_estimator(name: str):
    if name == "ANN":
        return MLPClassifier(hidden_layer_sizes=(8,), random_state=42)
    if name == "k-NN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "SVM":
        # ใช้ SGDClassifier(hinge) แทน SVC เสมอ (แม้ตอนห่อด้วย ensemble) เพราะ SVC ของจริง
        # (kernel method) ช้ามากเมื่อข้อมูลสะสมเยอะ ไม่เหมาะกับการเทรนซ้ำแบบ real-time
        return SGDClassifier(loss="hinge", random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    if name == "C4.5 Decision Tree":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    if name == "Random Tree":
        return ExtraTreeClassifier(max_depth=5, random_state=42)
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "XGBoost":
        return _XGBStringLabelClassifier(n_estimators=40, max_depth=4, learning_rate=0.3,
                                          eval_metric="logloss", random_state=42)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=60, depth=4, verbose=False,
                                   allow_writing_files=False, random_state=42)
    raise ValueError(f"ไม่รู้จัก base model: {name}")


def _is_expensive(base_name: str) -> bool:
    """ใช้กำหนดขนาด ensemble / ความถี่ retrain ให้เหมาะสม กันหน้าจอค้าง"""
    return base_name in ("ANN", "SVM", "Random Forest", "XGBoost", "CatBoost")


def _make_model(base_name: str, ensemble_name: str):
    """คืนค่า (model, is_partial, retrain_every_n, display_name)"""
    # เซฟตี้เน็ตอีกชั้น (นอกจาก GUI ที่กันไว้แล้ว) — KNeighborsClassifier ไม่รองรับ
    # sample_weight ที่ AdaBoostClassifier ต้องใช้ ทำให้เทรนไม่สำเร็จ 100% ของเวลา
    if base_name == "k-NN" and ensemble_name == "AdaBoost":
        ensemble_name = "None"
    base = _make_base_estimator(base_name)
    expensive = _is_expensive(base_name)

    if ensemble_name == "None":
        is_partial = base_name in PARTIAL_FIT_BASE_ONLY
        if is_partial:
            retrain_every_n = 2
        elif base_name == "ANN":
            retrain_every_n = 30  # MLPClassifier.fit() ช้ากว่า tree-based มาก
        elif base_name in ("XGBoost", "CatBoost"):
            retrain_every_n = 20  # gradient boosting เทรนช้ากว่า tree/forest เดี่ยว
        else:
            retrain_every_n = 8
        return base, is_partial, retrain_every_n, base_name

    if ensemble_name == "Bagging":
        n_est = 6 if expensive else 10
        model = BaggingClassifier(estimator=base, n_estimators=n_est, random_state=42)
        return model, False, (40 if expensive else 20), f"Bagging + {base_name}"

    if ensemble_name == "AdaBoost":
        n_est = 6 if expensive else 15
        model = AdaBoostClassifier(estimator=base, n_estimators=n_est, random_state=42)
        return model, False, (40 if expensive else 20), f"AdaBoost + {base_name}"

    raise ValueError(f"ไม่รู้จัก ensemble method: {ensemble_name}")


class OnlineTrainer:
    """
    เทรนแบบ Online: เรียก add_sample(features, gesture) ทุกครั้งที่มีข้อมูลใหม่เข้ามา
    (features เป็น dict 7 ค่า — ดู emg_features.FEATURE_NAMES) ข้อมูลจะถูกเพิ่มเข้าคิว
    ทันที (เร็ว ไม่บล็อก) ส่วนการเทรนจริงรันใน background thread แยกต่างหาก เพื่อไม่ให้
    โมเดลที่หนัก (เช่น ensemble ซ้อน ensemble) ทำให้ GUI ค้าง — เรียก get_snapshot() เป็น
    ระยะเพื่ออ่านผลลัพธ์ล่าสุด

    แบ่งข้อมูลที่เข้ามาเป็น train 70% / validate 10% / test 20% แบบสุ่มตั้งแต่รับเข้ามาเลย
    (สัดส่วนปรับได้ผ่าน test_ratio/val_ratio):
      - train    ใช้เทรนโมเดลตรงๆ
      - validate ใช้ประเมิน train vs validate accuracy/error ทุกรอบที่ retrain (เก็บไว้ใน
                 self.history — เอาไปวาดกราฟ loss/performance ระหว่างเทรนได้) ไม่เคยถูก
                 เอาไป fit โมเดลเลย แต่ก็ไม่ใช่ตัวเลขที่ใช้รายงานผลสุดท้าย (เพื่อไม่ให้
                 เลือกโมเดล/ปรับ hyperparameter จนไป overfit กับ validate set เองอีกที)
      - test     เป็น holdout แท้ๆ ที่ไม่แตะเลยจนกว่าจะประเมินผลสุดท้าย (accuracy/precision/
                 recall/f1/auc/kappa/confusion matrix — ดู _evaluate()) เพื่อวัด metric
                 อย่างยุติธรรม (ไม่ประเมินจากข้อมูลที่โมเดลเคยเห็นหรือเคยใช้ตัดสินใจตอนเทรน)

    การแบ่ง 3 ทางนี้ใช้ self._split_rng (random.Random(split_seed) instance แยกของตัวเอง
    ไม่ใช่ global random module) ทำให้ผลการแบ่งซ้ำเดิมทุกครั้งที่ป้อนข้อมูลชุดเดิมตามลำดับ
    เดิม — แก้ปัญหาผลลัพธ์ไม่นิ่งจาก run ต่อ run (ดู add_sample)

    Feature scaling: มี StandardScaler ประจำ trainer 1 ตัว (self.scaler) fit ใหม่จาก
    X_train ปัจจุบันทุกรอบที่ retrain (ไม่แตะ X_val/X_test เลย กัน data leakage) แล้วใช้
    scaler ตัวเดียวกัน (snapshot ล่าสุด ณ ตอนนั้น) กับทั้ง X_val/X_test ตอนประเมินผล และ
    feature vector ที่ส่งเข้า predict()/predict_with_confidence() เสมอ — model กับ scaler
    ถูกสลับพร้อมกันแบบ atomic ภายใต้ lock เดียวกัน กันเคส predict() เจอ model ใหม่แต่
    scaler เก่าค้างอยู่ (หรือกลับกัน) ซึ่งจะทำให้ทำนายผิดเพี้ยนแบบเงียบๆ โดยไม่มี error
    ให้เห็นเลย
    """

    def __init__(self, base_model: str, ensemble_method: str = "None",
                 test_ratio: float = 0.2, val_ratio: float = 0.1, max_history: int = 20000,
                 min_retrain_interval: float = 0.5, split_seed: int = 42,
                 history_max: int = 200):
        """
        test_ratio:   สัดส่วนข้อมูลที่แยกไปเป็น test holdout (ดีฟอลต์ 0.2 = 20%)
        val_ratio:    สัดส่วนข้อมูลที่แยกไปเป็น validate (ดีฟอลต์ 0.1 = 10%) ส่วนที่เหลือ
                      (1 - test_ratio - val_ratio, ดีฟอลต์ 0.7 = 70%) เป็น train
        max_history:  จำนวน sample "รวมทั้ง 3 split" สูงสุดที่เก็บไว้ทั้งหมด (ดีฟอลต์ 20000
                      — เดิมเคยตั้งไว้แค่ 6000 แต่พบว่าข้อมูลจริงที่เก็บสะสมได้ (self.n_samples)
                      เกิน 6000 ไปแล้ว ทำให้ sample เก่าสุดถูกเขี่ยออกแบบ FIFO เงียบๆ โดยที่
                      GUI ไม่เคยส่ง max_history เข้ามาปรับเองเลย — ยกเพดานขึ้นให้ครอบคลุม
                      ข้อมูลที่เก็บได้จริงในตอนนี้พร้อมเผื่อให้เก็บเพิ่มได้อีก โดยไม่ตัดข้อมูล
                      ทิ้งก่อนเวลาอันควร) — แบ่งเป็น cap ย่อยต่อ split ตามสัดส่วน
                      train_ratio/val_ratio/test_ratio ที่ตั้งไว้ (เช่น max_history=20000 กับ
                      สัดส่วนดีฟอลต์ 70/10/20 → cap_train=14000, cap_val=2000, cap_test=4000)
                      ⚠️ ของเดิมเคยใช้ max_history เป็น cap เดียวกันตรงๆ กับทั้ง 3 split
                      (cap_train == cap_val == cap_test == max_history) ทำให้ train ซึ่งได้รับ
                      sample เร็วกว่า val/test มาก (70% ของทุก sample ที่เข้ามา) ชนเพดานก่อนแล้ว
                      สัดส่วนจริงเพี้ยนไปจาก 70/10/20 ที่ตั้งใจไว้ — เปลี่ยนมาคิด cap แยกต่อ
                      split ตามสัดส่วนแทน แก้ปัญหานี้โดยตรง ยิ่งเยอะยิ่งเทรนช้าลง โดยเฉพาะ
                      Random Forest/k-NN/ensemble ที่ scan ข้อมูลทั้งหมดทุกรอบ (สำคัญมากบน
                      Raspberry Pi ที่ CPU จำกัด) ปรับเพิ่ม/ลดได้ตามจำนวนข้อมูลจริงที่ใช้
        min_retrain_interval: ระยะห่างขั้นต่ำระหว่างรอบเทรน (วินาที) ไม่ว่าจะมี sample
                      ใหม่เข้ามาถี่แค่ไหนก็ตาม กันไม่ให้ background thread ใช้ CPU รัว
                      ต่อเนื่อง — สำคัญมากบน Raspberry Pi ที่ CPU จำกัดกว่า PC
        split_seed:   seed ของตัวสุ่มแบ่ง train/validate/test (ดีฟอลต์ 42) — ตั้งตายตัวไว้
                      เพื่อให้ผลลัพธ์ "นิ่ง" ทำซ้ำได้ทุกครั้งที่ป้อนข้อมูลชุดเดิมตามลำดับเดิม
        history_max:  จำนวนรอบเทรนล่าสุดสูงสุดที่เก็บไว้ใน self.history สำหรับวาดกราฟ
                      loss/performance (ดีฟอลต์ 200 รอบ)
        """
        if not (0 < test_ratio + val_ratio < 1):
            raise ValueError(
                f"test_ratio + val_ratio ต้องอยู่ระหว่าง 0-1 (ได้ {test_ratio + val_ratio}) "
                "เพื่อให้เหลือสัดส่วนสำหรับ train อย่างน้อย 1 ส่วน")

        self.base_model = base_model
        self.ensemble_method = ensemble_method
        self.model, self.is_partial, self.retrain_every_n, self.display_name = \
            _make_model(base_model, ensemble_method)
        self.scaler = StandardScaler()  # ยังไม่ fit จนกว่าจะเทรนรอบแรกสำเร็จ (ดู _worker_loop)
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = 1.0 - test_ratio - val_ratio
        self.max_history = max_history
        # cap ย่อยต่อ split ตามสัดส่วนจริง (ดู docstring ของ max_history ด้านบน) — กัน train
        # ชนเพดานก่อน val/test แล้วทำให้สัดส่วนจริงเพี้ยนไปจาก train_ratio/val_ratio/
        # test_ratio ที่ตั้งไว้ อย่างน้อย 1 เสมอ กันเคส ratio เล็กมากจน round() ได้ 0
        self._cap_train = max(1, round(max_history * self.train_ratio))
        self._cap_val = max(1, round(max_history * self.val_ratio))
        self._cap_test = max(1, round(max_history * self.test_ratio))
        self.min_retrain_interval = min_retrain_interval
        self.history_max = history_max
        self._last_retrain_end = 0.0

        # ตัวสุ่มแบ่ง train/validate/test แยกเป็น instance ของตัวเอง (ไม่ใช่ global random
        # module) พร้อม seed ตายตัว — ทำให้การแบ่งซ้ำเดิมได้ทุกครั้ง ผลลัพธ์จึงนิ่งพอจะเอาไป
        # สรุป/เปรียบเทียบระหว่างโมเดลได้จริง (ดู module docstring หัวข้อ Reproducibility)
        self._split_rng = random.Random(split_seed)

        self._lock = threading.Lock()
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self._pending_since_retrain = 0
        self._initialized = False

        self.n_samples = 0
        self.n_retrains = 0
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.auc = None
        self.kappa = None
        self.confusion = None
        self.n_errors = 0          # จำนวนครั้งที่เทรนไม่สำเร็จสะสม (ให้ GUI ตรวจจับ error ใหม่ได้)
        self.last_error = None     # ข้อความ exception ล่าสุด (str) — None ถ้ายังไม่เคย error

        # ประวัติ train/validate accuracy-error ต่อรอบเทรน — ใช้วาดกราฟ loss/performance
        # (แต่ละ item: {"round", "train_accuracy", "train_error", "val_accuracy",
        # "val_error", "n_train", "n_val"}) เก็บไว้สูงสุด history_max รอบล่าสุด
        self.history = []
        # label ที่โมเดลล่าสุดทำนายบน test set (เรียงคู่กับ self.y_test ตามลำดับ) — เก็บไว้
        # ใช้วาดกราฟเปรียบเทียบ predict (actual vs predicted) บนหน้าจอ อัพเดตทุกรอบ retrain
        self._last_test_pred = []

        self._stop_flag = threading.Event()
        self._retrain_requested = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ---------- Public API (เรียกจาก GUI thread ได้อย่างปลอดภัย) ----------

    def add_sample(self, features: dict, gesture: str, person_category: str = None):
        """เพิ่ม sample ใหม่เข้าคิว (เร็วมาก ไม่บล็อก) แล้วสั่งให้ background thread
        เทรนใหม่ถ้าถึงรอบ retrain_every_n

        features: dict 7 ค่า {"rms":..., "mav":..., "variance":..., "waveform_length":...,
                  "zero_crossing":...} จาก emg_features.extract_features() — เก็บเป็นเวกเตอร์
                  ดิบ (ยังไม่ scale) ลง buffer ตรงๆ เพราะ StandardScaler ต้อง fit จากข้อมูล
                  ทั้งชุดใน X_train (ดูตอน retrain ใน _worker_loop) จะ scale ทีละ sample ตอน
                  รับเข้ามาเลยไม่ได้ (ยังไม่รู้ mean/std ของทั้งชุด)

        person_category: รับพารามิเตอร์นี้ไว้เหมือนเดิมเพื่อ backward-compat กับโค้ดฝั่งเรียก
                  (prosthetic_gui.py ยังส่งมาอยู่) แต่ ⚠️ ไม่ถูกนำไปคำนวณ feature vector ที่
                  ป้อนเข้าโมเดลอีกต่อไป (ดู module docstring) — โมเดลตัดสิน FIST/OPEN จาก
                  รูปคลื่นสัญญาณ EMG 7 ค่าล้วนๆ (emg_features.features_to_vector) เท่านั้น"""
        gesture = _normalize_gesture(gesture)
        vector = features_to_vector(features)

        with self._lock:
            self.n_samples += 1
            # แบ่ง train/validate/test ด้วย self._split_rng (instance ของตัวเอง มี seed
            # ตายตัว) แทน random module ระดับ global แบบเดิม — ของเดิมทำให้สัดส่วนที่แต่ละ
            # sample ตกไปอยู่ split ไหนไม่ซ้ำเดิมทุกครั้งที่รันโปรแกรมใหม่ (แม้ป้อนข้อมูล
            # ชุดเดิมตามลำดับเดิมทุกประการ) ผลลัพธ์/metric สุดท้ายเลยไม่นิ่งพอจะเอาไปสรุป
            # เปรียบเทียบระหว่างโมเดลได้ — ใช้ instance นี้แทนแก้ปัญหาตรงนี้โดยตรง
            r = self._split_rng.random()
            if r < self.test_ratio:
                self.X_test.append(vector); self.y_test.append(gesture)
                if len(self.X_test) > self._cap_test:
                    self.X_test.pop(0); self.y_test.pop(0)
            elif r < self.test_ratio + self.val_ratio:
                self.X_val.append(vector); self.y_val.append(gesture)
                if len(self.X_val) > self._cap_val:
                    self.X_val.pop(0); self.y_val.pop(0)
            else:
                self.X_train.append(vector); self.y_train.append(gesture)
                if len(self.X_train) > self._cap_train:
                    self.X_train.pop(0); self.y_train.pop(0)

            self._pending_since_retrain += 1
            due = self._pending_since_retrain >= self.retrain_every_n

        if due:
            self._retrain_requested.set()

    def force_retrain_now(self):
        """สั่งเทรนทันที โดยไม่ต้องรอ min_retrain_interval — ใช้ตอนโหลดไฟล์ offline เสร็จ
        เพื่อให้ผลลัพธ์สุดท้ายที่แสดงสะท้อนข้อมูล 'ทั้งหมด' ที่โหลดมาจริงๆ ไม่ใช่แค่บางส่วน
        ที่ทันเทรนตอน throttle จำกัดความถี่ไว้ระหว่างที่กำลังโหลดไฟล์เร็วๆ (ไม่บล็อก — แค่
        ปลุก background thread ให้ทำงานทันที ผลลัพธ์ค่อยไปเช็คทีหลังผ่าน get_snapshot())"""
        with self._lock:
            self._last_retrain_end = 0.0  # รีเซ็ตตัวจับเวลา ให้รอบถัดไปไม่ต้องรอ interval
        self._retrain_requested.set()

    def get_snapshot(self) -> dict:
        """อ่านผลลัพธ์ล่าสุดแบบ thread-safe (เรียกจาก GUI เป็นระยะเพื่ออัพเดตหน้าจอ)

        เพิ่มใหม่: "history" (list ของ train/validate accuracy-error ต่อรอบเทรน สำหรับ
        วาดกราฟ loss/performance), "split_summary" (จำนวน/สัดส่วน sample จริงที่ตกอยู่ใน
        แต่ละ split ณ ตอนนี้ — เทียบกับสัดส่วนเป้าหมาย train70/val10/test20 ที่ตั้งไว้),
        "last_test_pred"/"last_test_true" (label ที่โมเดลล่าสุดทำนายจริงบน test set —
        ใช้วาดกราฟเปรียบเทียบ predict บนหน้าจอ)"""
        with self._lock:
            n_train, n_val, n_test = len(self.X_train), len(self.X_val), len(self.X_test)
            n_split_total = n_train + n_val + n_test
            return {
                "n_samples": self.n_samples,
                "n_retrains": self.n_retrains,
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "auc": self.auc,
                "kappa": self.kappa,
                "confusion": None if self.confusion is None else self.confusion.copy(),
                "initialized": self._initialized,
                "n_errors": self.n_errors,
                "last_error": self.last_error,
                "history": list(self.history),
                "split_summary": {
                    "n_train": n_train, "n_val": n_val, "n_test": n_test,
                    "pct_train": (n_train / n_split_total * 100) if n_split_total else 0.0,
                    "pct_val": (n_val / n_split_total * 100) if n_split_total else 0.0,
                    "pct_test": (n_test / n_split_total * 100) if n_split_total else 0.0,
                    "target_train": self.train_ratio * 100,
                    "target_val": self.val_ratio * 100,
                    "target_test": self.test_ratio * 100,
                },
                "last_test_true": list(self.y_test),
                "last_test_pred": list(self._last_test_pred) if self._last_test_pred else [],
            }

    def get_boundary_data(self, feat_x: str = "rms", feat_y: str = "mav"):
        """เตรียมข้อมูล + โมเดลตัวช่วยสำหรับวาดกราฟ Decision Boundary 2 มิติ (GUI เรียกใช้ตอน
        วาดกราฟในแท็บ Machine Learning) — โมเดลจริง (self.model) ตัดสินใจจากครบทั้ง 7
        features เสมอ (ดู module docstring) วาดขอบเขตการตัดสินใจแบบ 2 มิติตรงๆ จากโมเดลจริง
        ไม่ได้ ที่นี่จึงเลือก 2 ใน 7 features (ดีฟอลต์ RMS กับ MAV) มาเทรน "โมเดลตัวช่วย"
        แยกต่างหาก — สถาปัตยกรรม/hyperparameter ชุดเดียวกับโมเดลจริงที่กำลังเลือกใช้อยู่ตอนนี้
        (เรียก _make_model ซ้ำด้วย base_model/ensemble_method เดิม) ผ่าน
        Pipeline(StandardScaler → model) เหมือนที่โมเดลจริงทำกับ scaler ของตัวเอง — กราฟที่
        ได้จึงเป็นแค่ "ภาพประกอบ" ขอบเขตการตัดสินใจแบบง่ายบนระนาบ 2 มิตินั้น ไม่ใช่ขอบเขตจริง
        ของโมเดล 7 มิติที่ใช้งานจริงกับ servo (ไม่แตะ/ไม่แก้ self.model, self.scaler เลย)

        คืนค่า None ถ้าข้อมูล train ยังไม่พอ (ต้องมีอย่างน้อย 4 sample และครบ 2 class) หรือ
        ถ้าเทรนโมเดลตัวช่วยไม่สำเร็จ"""
        with self._lock:
            if len(self.X_train) < 4 or len(set(self.y_train)) < 2:
                return None
            X_train_full = np.array(self.X_train, dtype=float)
            y_train = np.array(self.y_train)
            base_model, ensemble_method = self.base_model, self.ensemble_method
            display_name = self.display_name

        names = list(FEATURE_NAMES)
        if feat_x not in names or feat_y not in names:
            return None
        ix, iy = names.index(feat_x), names.index(feat_y)
        X2 = X_train_full[:, [ix, iy]]

        try:
            model2d, _, _, _ = _make_model(base_model, ensemble_method)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", model2d)])
            pipe.fit(X2, y_train)
        except Exception:
            return None

        return {
            "pipeline": pipe, "X2": X2, "y": y_train,
            "feat_x": feat_x, "feat_y": feat_y, "display_name": display_name,
        }

    def predict(self, features: dict, person_category: str = None):
        """person_category: รับไว้เหมือนเดิมเพื่อ backward-compat กับโค้ดฝั่งเรียก แต่ไม่ถูก
        นำไปใช้คำนวณ feature vector อีกต่อไป (ดู module docstring / add_sample)"""
        with self._lock:
            if not self._initialized:
                return None
            model = self.model
            scaler = self.scaler
        X = scaler.transform([features_to_vector(features)])
        return model.predict(X)[0]

    def predict_with_confidence(self, features: dict, person_category: str = None):
        """เหมือน predict() แต่คืนค่าความมั่นใจ (%) มาด้วยถ้าโมเดลรองรับ predict_proba
        คืนค่า (label, confidence_pct) — confidence_pct เป็น None ถ้าโมเดลไม่รองรับ

        features: dict 7 ค่าจาก emg_features.extract_features() (สัญญาสัญญาเดียวกับ
        add_sample) — ใช้ scaler ตัวเดียวกับที่ fit ตอนเทรนโมเดลปัจจุบัน (อ่านพร้อมกับ
        model ภายใต้ lock เดียวกัน กันเคส scaler/model ไม่ตรงรุ่นกัน)

        person_category: รับไว้เหมือนเดิมเพื่อ backward-compat กับโค้ดฝั่งเรียก แต่ไม่ถูก
        นำไปใช้คำนวณ feature vector อีกต่อไป (ดู module docstring / add_sample) — โมเดล
        ตัดสินจากรูปคลื่นสัญญาณ EMG 7 ค่าล้วนๆ เท่านั้น"""
        with self._lock:
            if not self._initialized:
                return None, None
            model = self.model
            scaler = self.scaler
        X = scaler.transform([features_to_vector(features)])
        pred = model.predict(X)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                classes = list(model.classes_)
                conf = float(proba[classes.index(pred)]) * 100.0
            except Exception:
                conf = None
        return pred, conf

    def stop(self):
        self._stop_flag.set()
        self._retrain_requested.set()  # ปลุก worker ให้เช็ค stop flag แล้วจบทันที

    # ---------- Background worker ----------

    def _worker_loop(self):
        while not self._stop_flag.is_set():
            triggered = self._retrain_requested.wait(timeout=0.5)
            if self._stop_flag.is_set():
                return
            if not triggered:
                continue
            self._retrain_requested.clear()

            # บังคับเว้นระยะห่างขั้นต่ำระหว่างรอบเทรนเสมอ ไม่ว่า sample จะเข้ามาถี่แค่ไหน
            # กัน background thread กิน CPU รัวต่อเนื่อง — จุดสำคัญที่สุดสำหรับ Raspberry Pi
            wait_more = self.min_retrain_interval - (time.time() - self._last_retrain_end)
            if wait_more > 0:
                time.sleep(wait_more)
            if self._stop_flag.is_set():
                return

            with self._lock:
                self._pending_since_retrain = 0
                X_train = list(self.X_train)
                y_train = list(self.y_train)

            if len(X_train) < 2 or len(set(y_train)) < 2:
                continue  # ต้องมีทั้ง FIST และ OPEN ก่อนเทรนได้จริง

            X_raw = np.array(X_train)
            y = np.array(y_train)

            try:
                # Fit scaler จาก training data รอบนี้เท่านั้น (ไม่แตะ X_test เลย กัน data
                # leakage — ข้อกำหนดเรื่อง Feature Scaling) แล้วใช้ scaler ตัวนี้แปลง X_raw
                # ก่อนป้อนเข้าโมเดล — ทำเป็น local variable ก่อน ยังไม่เขียนทับ self.scaler
                # จนกว่าจะ fit โมเดลสำเร็จ (เผื่อ error กลางทาง จะได้ไม่เหลือ scaler ใหม่
                # ค้างอยู่คู่กับโมเดลเก่า ซึ่งจะทำให้ scale ไม่ตรงกับที่โมเดลเก่าเคยเรียนรู้)
                new_scaler = StandardScaler()
                X = new_scaler.fit_transform(X_raw)

                if self.is_partial:
                    # partial_fit เร็ว (โมเดลเล็ก) ถือ lock ได้ทั้งการเรียกโดยไม่หน่วง GUI
                    # thread มาก — กันไม่ให้ predict() อ่านค่ากลางคันตอนกำลังอัพเดต
                    with self._lock:
                        if not self._initialized:
                            self.model.partial_fit(X, y, classes=CLASSES)
                        else:
                            self.model.partial_fit(X, y)
                        self.scaler = new_scaler
                        self._initialized = True
                        self.n_retrains += 1
                else:
                    # เทรนบน "โมเดลใหม่ที่ clone มา" นอก lock (อาจใช้เวลานาน) แล้วค่อยสลับ
                    # pointer แบบ atomic ใน lock สั้นๆ — กัน predict() อ่านโมเดลที่กำลังถูก
                    # fit() เขียนทับอยู่กลางคัน (เคยเจอ IndexError จาก race condition นี้จริง)
                    # scaler ใหม่ก็สลับพร้อมกันในจุดเดียวกันนี้ (atomic คู่กับ model เสมอ)
                    new_model = clone(self.model)
                    new_model.fit(X, y)
                    with self._lock:
                        self.model = new_model
                        self.scaler = new_scaler
                        self._initialized = True
                        self.n_retrains += 1
            except Exception as e:
                # เดิมแค่ print() ออก console เฉยๆ — ผู้ใช้ที่ไม่ได้เปิด terminal คู่ไว้จะไม่มี
                # ทางรู้เลยว่าทำไมโมเดลนี้ไม่มีผลลัพธ์ขึ้นเลย (เช่น 'AdaBoost + XGBoost' บาง
                # เวอร์ชันไลบรารีเข้ากันไม่ได้) จึงเก็บข้อความ error ไว้ใน last_error/n_errors
                # ด้วย ให้ GUI (prosthetic_gui.py) โพลแล้วเอาไป log ให้เห็นในหน้าจอได้
                err_msg = f"{type(e).__name__}: {e}"
                print(f"[OnlineTrainer] เทรน {self.display_name} ไม่สำเร็จ: {err_msg}")
                with self._lock:
                    self.last_error = err_msg
                    self.n_errors += 1
                self._last_retrain_end = time.time()
                continue

            # อัพเดต train vs validate accuracy/error ของรอบเทรนนี้ — ใช้เป็น "loss/
            # performance" สำหรับกราฟใน GUI (ตอบข้อ 5 ที่เลือก: ใช้ Accuracy/Error
            # (1-accuracy) แทน loss จริง เพราะโมเดลนี้เป็น sklearn classifier ไม่ใช่ deep
            # learning ที่มี loss แบบ epoch ตรงๆ) — X, y ด้านบนคือชุด train (สเกลแล้ว) ที่
            # เพิ่งใช้ fit รอบนี้พอดี ส่วน validate ไม่เคยถูกใช้ fit เลย (กัน data leakage
            # เหมือน test set) แต่ประเมินได้ทุกรอบเพื่อดูแนวโน้ม overfitting ระหว่างเทรน
            with self._lock:
                model_now = self.model
                scaler_now = self.scaler
                X_val_raw = list(self.X_val)
                y_val_raw = list(self.y_val)

            train_pred = model_now.predict(X)
            train_acc = float(accuracy_score(y, train_pred))

            val_acc = None
            if len(X_val_raw) >= 1:
                try:
                    Xv = scaler_now.transform(np.array(X_val_raw))
                    val_pred = model_now.predict(Xv)
                    val_acc = float(accuracy_score(y_val_raw, val_pred))
                except Exception:
                    val_acc = None

            with self._lock:
                self.history.append({
                    "round": self.n_retrains,
                    "train_accuracy": train_acc,
                    "train_error": 1.0 - train_acc,
                    "val_accuracy": val_acc,
                    "val_error": (1.0 - val_acc) if val_acc is not None else None,
                    "n_train": len(X),
                    "n_val": len(X_val_raw),
                })
                if len(self.history) > self.history_max:
                    self.history.pop(0)

            self._evaluate()
            self._last_retrain_end = time.time()

    def _decision_scores(self, model, X):
        """คะแนนสำหรับคำนวณ AUC — ลองใช้ predict_proba ก่อน ตกไป decision_function
        แล้วสุดท้าย fallback เป็น hard prediction (ไม่ error ไม่ว่าโมเดลจะเป็นแบบไหน)
        รับ model เป็นพารามิเตอร์ (แทนที่จะอ่าน self.model ตรงๆ) เพื่อให้แน่ใจว่าใช้
        model snapshot เดียวกับที่ _evaluate() ล็อกไว้ตอนต้น ไม่ใช่ค่าที่อาจถูกสลับไปแล้ว
        โดย _worker_loop ระหว่างที่ _evaluate() กำลังทำงานอยู่"""
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                classes = list(model.classes_)
                return proba[:, classes.index(POS_LABEL)]
            except Exception:
                pass
        if hasattr(model, "decision_function"):
            try:
                raw = model.decision_function(X)
                # decision_function > 0 หมายถึง classes_[1] เสมอ (ธรรมเนียม sklearn)
                # ต้องเช็คว่า classes_[1] คือ FIST จริงไหม ไม่งั้นค่าจะกลับขั้ว (AUC ผิดเป็น ~0
                # ทั้งที่จริงแยกคลาสได้สมบูรณ์แบบ แค่ทายขั้วกลับด้าน)
                classes = list(model.classes_)
                return raw if classes[1] == POS_LABEL else -raw
            except Exception:
                pass
        pred = model.predict(X)
        return (pred == POS_LABEL).astype(float)

    def _evaluate(self):
        with self._lock:
            X_test = list(self.X_test)
            y_test = list(self.y_test)

        if len(X_test) < 2 or len(set(y_test)) < 2:
            with self._lock:
                self.accuracy = self.precision = self.recall = None
                self.f1 = self.auc = self.kappa = None
                self.confusion = None
                self._last_test_pred = []
            return

        with self._lock:
            model = self.model
            scaler = self.scaler

        Xt_raw = np.array(X_test)
        yt = np.array(y_test)
        Xt = scaler.transform(Xt_raw)  # scaler ตัวเดียวกับที่ fit คู่กับโมเดลปัจจุบัน (ดู _worker_loop)
        pred = model.predict(Xt)

        idx = {c: i for i, c in enumerate(CLASSES)}
        conf = np.zeros((2, 2), dtype=int)
        for a, p in zip(yt, pred):
            conf[idx[a], idx[p]] += 1

        accuracy = float(accuracy_score(yt, pred))
        precision = float(precision_score(yt, pred, pos_label=POS_LABEL, zero_division=0))
        recall = float(recall_score(yt, pred, pos_label=POS_LABEL, zero_division=0))
        f1 = float(f1_score(yt, pred, pos_label=POS_LABEL, zero_division=0))
        kappa = float(cohen_kappa_score(yt, pred))
        try:
            scores = self._decision_scores(model, Xt)
            auc = float(roc_auc_score((yt == POS_LABEL).astype(int), scores))
        except Exception:
            auc = None

        with self._lock:
            self.accuracy, self.precision, self.recall = accuracy, precision, recall
            self.f1, self.kappa, self.auc = f1, kappa, auc
            self.confusion = conf
            # เก็บ predicted label บน test set รอบล่าสุด (เรียงคู่กับ self.y_test) — ใช้วาด
            # กราฟเปรียบเทียบ predict (actual vs predicted) ในหน้าจอ ML — ก๊อปแค่ list ของ
            # label ธรรมดา (ไม่ใช่ numpy array) กันปัญหา thread อื่นถือ reference ไปยัง
            # array ที่อาจถูกแก้ไขทีหลัง (แม้ในทางปฏิบัติจะไม่เกิดเพราะไม่มีจุดไหนแก้ pred
            # array นี้ซ้ำ แต่ป้องกันไว้ก่อนเพื่อความชัดเจนของ ownership)
            self._last_test_pred = [str(p) for p in pred]