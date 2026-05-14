import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)
mp_drawing = mp.solutions.drawing_utils

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567891011121314151617181920")
DATA_FILE = "dataset.csv"

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        writer.writerow(header)

cap = cv2.VideoCapture(0)
current_label = "A"
collecting = False
count = 0
TARGET = 100

print("Tekan huruf A-Z dan angka 1-20 untuk mulai rekam, SPACE untuk stop, ESC untuk keluar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and collecting:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = [current_label]
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]

            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            count += 1
            if count >= TARGET:
                collecting = False
                print(f"Huruf {current_label} selesai! ({TARGET} sampel)")
                count = 0

    elif results.multi_hand_landmarks and not collecting:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # UI
    status = f"RECORDING [{current_label}]: {count}/{TARGET}" if collecting else f"Siap - Tekan huruf untuk rekam"
    color = (0, 0, 255) if collecting else (0, 255, 0)
    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, "Huruf: " + current_label, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "SPACE=stop | ESC=keluar | A-Z 1-20=rekam",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Progress huruf yang sudah selesai
    done = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            rows = list(csv.reader(f))
            for row in rows[1:]:
                if row and row[0] not in done:
                    done.append(row[0])
    cv2.putText(frame, f"Selesai: {sorted(done)}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imshow("Kumpul Data BISINDO", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC untuk keluar
        break
    elif key == ord(' '):
        collecting = False
        count = 0
        print("Stop rekam")
    elif chr(key).upper() in LABELS:
        current_label = chr(key).upper()
        collecting = True
        count = 0
        print(f"Merekam huruf: {current_label}")

cap.release()
cv2.destroyAllWindows()