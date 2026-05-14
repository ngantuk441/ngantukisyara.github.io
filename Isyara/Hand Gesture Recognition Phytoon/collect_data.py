import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# ─────────────────────────────────────────────
# Setup MediaPipe
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# Ganti nama file sesuai sesi:
#   "dataset_alfabet.csv"
#   "dataset_angka.csv"
#   "dataset_kata.csv"
# ─────────────────────────────────────────────
DATA_FILE = "dataset_alfabet.csv"

LM_COLS_H1 = [f"{ax}{i}_h1" for i in range(21) for ax in ["x", "y", "z"]]
LM_COLS_H2 = [f"{ax}{i}_h2" for i in range(21) for ax in ["x", "y", "z"]]
HEADER = ["label", "hand_mode", "hand1_side"] + LM_COLS_H1 + ["hand2_side"] + LM_COLS_H2

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
current_label = "A"
collecting    = False
hand_mode     = 1       # 1 = 1 tangan | 2 = 2 tangan
count         = 0
TARGET        = 300

# Mode ketik label custom (backtick)
typing_mode   = False
typed_label   = ""

# ─────────────────────────────────────────────
# Helper ekstrak row
# ─────────────────────────────────────────────
def extract_row(label, mode, multi_hand_landmarks, multi_handedness):
    if not multi_hand_landmarks:
        return None

    detected = list(zip(multi_handedness, multi_hand_landmarks))
    rights = [(hd, lm) for hd, lm in detected if hd.classification[0].label == "Right"]
    lefts  = [(hd, lm) for hd, lm in detected if hd.classification[0].label == "Left"]

    def flat(lm_obj):
        out = []
        for p in lm_obj.landmark:
            out += [p.x, p.y, p.z]
        return out

    if mode == 1:
        chosen_hd, chosen_lm = rights[0] if rights else lefts[0]
        side_enc = 1 if chosen_hd.classification[0].label == "Right" else 0
        lm1 = flat(chosen_lm)
        lm2 = [0.0] * 63
        return [label, mode, side_enc] + lm1 + [-1] + lm2

    elif mode == 2:
        if len(detected) < 2:
            return None
        if rights and lefts:
            _, lm1 = rights[0]
            _, lm2 = lefts[0]
            enc1, enc2 = 1, 0
        elif len(rights) >= 2:
            _, lm1 = rights[0]
            _, lm2 = rights[1]
            enc1, enc2 = 1, 1
        else:
            _, lm1 = lefts[0]
            _, lm2 = lefts[1]
            enc1, enc2 = 0, 0
        return [label, mode, enc1] + flat(lm1) + [enc2] + flat(lm2)

    return None

# ─────────────────────────────────────────────
# Baca progress
# ─────────────────────────────────────────────
def get_done():
    done = {}
    if not os.path.exists(DATA_FILE):
        return done
    with open(DATA_FILE, "r") as f:
        for row in list(csv.reader(f))[1:]:
            if row:
                done[row[0]] = done.get(row[0], 0) + 1
    return done

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)

print("=" * 55)
print("   COLLECT DATA BISINDO")
print("=" * 55)
print(f"  File    : {DATA_FILE}")
print(f"  Target  : {TARGET} sampel per label")
print("-" * 55)
print("  A–Z          : huruf BISINDO")
print("  1–9          : angka 1-9  |  0 = angka 10")
print("  TAB          : toggle 1 tangan ↔ 2 tangan")
print("  ` (backtick) : ketik label custom (HALO, PEACE, dll)")
print("  SPACE        : stop recording")
print("  ESC          : keluar")
print("=" * 55)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    # ── Gambar landmark ──────────────────────────────────────
    if results.multi_hand_landmarks:
        for i, lm in enumerate(results.multi_hand_landmarks):
            side = results.multi_handedness[i].classification[0].label
            color_node = (0, 255, 100) if side == "Right" else (255, 120, 0)
            mp_drawing.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color_node, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1),
            )
            wrist = lm.landmark[0]
            tx, ty = int(wrist.x * w), int(wrist.y * h) - 15
            cv2.putText(frame, side, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_node, 2)

    # ── Rekam data ───────────────────────────────────────────
    warning_msg = ""
    if collecting and not typing_mode and results.multi_hand_landmarks:
        row = extract_row(current_label, hand_mode,
                          results.multi_hand_landmarks,
                          results.multi_handedness)
        if row is not None:
            with open(DATA_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
            count += 1
            if count >= TARGET:
                collecting = False
                print(f"✓ [{current_label}] selesai! ({TARGET} sampel, {hand_mode} tangan)")
                count = 0
        else:
            warning_msg = f"Butuh 2 tangan! Tampilkan kedua tangan ke kamera."

    # ── UI header ────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 108), (20, 20, 20), -1)

    if collecting:
        status_txt   = f"REC [{current_label}]   {count}/{TARGET}"
        status_color = (0, 60, 255)
    else:
        status_txt   = "Siap — tekan huruf / angka / ` untuk label custom"
        status_color = (0, 220, 80)

    cv2.putText(frame, status_txt, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    mode_color = (0, 200, 255) if hand_mode == 2 else (0, 255, 150)
    mode_txt   = f"{'2 TANGAN' if hand_mode == 2 else '1 TANGAN'}  (TAB=ganti)"
    cv2.putText(frame, f"Label : {current_label}",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
    cv2.putText(frame, mode_txt,
                (w - 240, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    # Progress bar
    if collecting:
        bar = int((count / TARGET) * w)
        cv2.rectangle(frame, (0, 106), (bar, 112), (0, 200, 80), -1)

    # Warning
    if warning_msg:
        cv2.putText(frame, warning_msg, (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)

    # ── Overlay ketik label custom ───────────────────────────
    if typing_mode:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2 - 70), (w, h//2 + 70), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "Ketik nama label lalu tekan ENTER (ESC=batal):",
                    (20, h//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, f"> {typed_label}_",
                    (20, h//2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 2)
        cv2.putText(frame, "Contoh: HALO  PEACE  SIP  ILY  OK",
                    (20, h//2 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # ── Progress label bawah ─────────────────────────────────
    done     = get_done()
    done_str = "  ".join([f"{l}:{n}" for l, n in sorted(done.items())])
    cv2.rectangle(frame, (0, h - 52), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Data: {done_str if done_str else '(belum ada)'}",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
    cv2.putText(frame, f"File: {DATA_FILE}   |   SPACE=stop   ESC=keluar",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    cv2.imshow("Collect Data BISINDO", frame)
    key = cv2.waitKey(1) & 0xFF

    # ════════════════════════════════════════════════════════
    # Key handling — MODE KETIK LABEL CUSTOM
    # ════════════════════════════════════════════════════════
    if typing_mode:
        if key == 27:                          # ESC = batal
            typing_mode = False
            typed_label = ""
        elif key == 13:                        # ENTER = konfirmasi
            if typed_label.strip():
                current_label = typed_label.strip().upper()
                collecting    = True
                count         = 0
                print(f"▶ Merekam label custom: '{current_label}' ({hand_mode} tangan)")
            typing_mode = False
            typed_label = ""
        elif key == 8:                         # BACKSPACE
            typed_label = typed_label[:-1]
        elif 32 <= key <= 126:                 # karakter biasa
            typed_label += chr(key)
        continue                               # skip key handling normal

    # ════════════════════════════════════════════════════════
    # Key handling — MODE NORMAL
    # ════════════════════════════════════════════════════════
    if key == 27:                              # ESC = keluar
        break

    elif key == ord(' '):                      # SPACE = stop
        collecting = False
        count = 0
        print("⏹ Stop.")

    elif key == 9:                             # TAB = toggle tangan
        hand_mode = 2 if hand_mode == 1 else 1
        collecting = False
        count = 0
        print(f"🔄 Mode: {hand_mode} tangan")

    elif key == ord('`'):                      # ` = ketik label custom
        typing_mode = True
        typed_label = ""
        collecting  = False
        count       = 0

    elif key == ord('0'):                      # 0 = angka 10
        current_label = "10"
        collecting    = True
        count         = 0
        print(f"▶ Merekam: {current_label} ({hand_mode} tangan)")

    elif ord('1') <= key <= ord('9'):          # angka 1-9
        current_label = chr(key)
        collecting    = True
        count         = 0
        print(f"▶ Merekam: {current_label} ({hand_mode} tangan)")

    elif 65 <= key <= 90 or 97 <= key <= 122: # A-Z
        current_label = chr(key).upper()
        collecting    = True
        count         = 0
        print(f"▶ Merekam: {current_label} ({hand_mode} tangan)")

cap.release()
cv2.destroyAllWindows()
print(f"\nSesi selesai. Data tersimpan di: {DATA_FILE}")