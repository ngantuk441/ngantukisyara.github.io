import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

# ── Load model ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bisindo_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

N_FEATURES = model.n_features_in_  # otomatis baca dari model (63 atau 127)
print(f"[INFO] Model loaded. Ekspektasi fitur: {N_FEATURES}")
print(f"[INFO] Label yang dikenal: {list(model.classes_)}")

# ── MediaPipe ─────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# ── Helper: buat feature vector dari hasil deteksi ────────────
def build_features(multi_hand_landmarks, multi_handedness):
    """
    Selalu return vector dengan panjang N_FEATURES.
    
    Kalau model expect 63  → pakai 1 tangan saja
    Kalau model expect 127 → 1 handedness + 63 h1 + 63 h2
                             (tangan kedua diisi 0 kalau tidak ada)
    """
    if not multi_hand_landmarks:
        return None

    # Pisah tangan berdasarkan handedness
    right_lm, left_lm = None, None
    for i, hd in enumerate(multi_handedness):
        side = hd.classification[0].label  # "Right" / "Left"
        lm = multi_hand_landmarks[i]
        if side == "Right" and right_lm is None:
            right_lm = lm
        elif side == "Left" and left_lm is None:
            left_lm = lm

    # Tangan dominan = Right (kalau ada), fallback ke Left
    dominant = right_lm if right_lm else left_lm
    secondary = left_lm if right_lm else None

    def lm_flat(lm_obj):
        flat = []
        for p in lm_obj.landmark:
            flat += [p.x, p.y, p.z]
        return flat  # 63 nilai

    feat_h1 = lm_flat(dominant)
    feat_h2 = lm_flat(secondary) if secondary else [0.0] * 63

    if N_FEATURES == 63:
        # Model lama: hanya 1 tangan
        return feat_h1
    elif N_FEATURES == 127:
        # Model baru: handedness + h1 + h2
        handedness_val = 1 if right_lm else 0
        return [handedness_val] + feat_h1 + feat_h2
    else:
        # Fallback: pad atau potong sesuai N_FEATURES
        raw = feat_h1 + feat_h2
        if len(raw) >= N_FEATURES:
            return raw[:N_FEATURES]
        else:
            return raw + [0.0] * (N_FEATURES - len(raw))


# ── State kalimat ─────────────────────────────────────────────
sentence    = ""
last_letter = ""
stable_count = 0
STABLE_THRESHOLD = 15   # frame stabil sebelum huruf ditambah

# ── Main loop ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Kamera aktif. Tekan ESC untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted  = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        # Gambar semua landmark
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            side = results.multi_handedness[i].classification[0].label
            node_color = (0, 255, 100) if side == "Right" else (255, 100, 0)
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=node_color, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1),
            )

        # Buat feature vector
        features = build_features(
            results.multi_hand_landmarks,
            results.multi_handedness,
        )

        if features is not None:
            feat_arr = np.array(features).reshape(1, -1)
            predicted  = model.predict(feat_arr)[0]
            confidence = max(model.predict_proba(feat_arr)[0]) * 100

            # Tampilkan prediksi di atas pergelangan tangan dominan
            wrist = results.multi_hand_landmarks[0].landmark[0]
            h, w, _ = frame.shape
            cx = int(wrist.x * w)
            cy = int(wrist.y * h)
            cv2.putText(frame, f"{predicted} ({confidence:.0f}%)",
                        (cx - 40, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # ── Auto-append ke kalimat kalau stabil ───────────────────
    if predicted and predicted == last_letter:
        stable_count += 1
        if stable_count == STABLE_THRESHOLD:
            sentence += predicted
            print(f"Tambah: '{predicted}' → '{sentence}'")
    else:
        stable_count  = 0
        last_letter   = predicted

    # ── Progress bar stabilitas ───────────────────────────────
    h, w, _ = frame.shape
    if predicted:
        bar_w = int((stable_count / STABLE_THRESHOLD) * 200)
        cv2.rectangle(frame, (10, 110), (210, 128), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 110), (10 + bar_w, 128), (0, 220, 80), -1)
        cv2.putText(frame, "Stabil", (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── UI bawah ──────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Kalimat: {sentence}",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "ESC=keluar | SPACE=hapus huruf terakhir | ENTER=reset kalimat",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    cv2.imshow("BISINDO — Real-time", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:       # ESC
        break
    elif key == ord(' '):   # SPACE → hapus huruf terakhir
        sentence = sentence[:-1]
    elif key == 13:         # ENTER → reset kalimat
        sentence = ""

cap.release()
cv2.destroyAllWindows()
print(f"Sesi selesai. Kalimat terakhir: '{sentence}'")