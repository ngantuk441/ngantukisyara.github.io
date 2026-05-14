import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def y(landmarks, id): return landmarks[id].y
def x(landmarks, id): return landmarks[id].x

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            gesture = ""

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )

            lm = hand_landmarks.landmark

            thumb_up   = y(lm, mp_hands.HandLandmark.THUMB_TIP)         < y(lm, mp_hands.HandLandmark.THUMB_IP)
            index_up   = y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)  < y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP)
            middle_up  = y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) < y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
            ring_up    = y(lm, mp_hands.HandLandmark.RING_FINGER_TIP)   < y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
            pinky_up   = y(lm, mp_hands.HandLandmark.PINKY_TIP)         < y(lm, mp_hands.HandLandmark.PINKY_PIP)

            thumb_down  = not thumb_up
            index_down  = not index_up
            middle_down = not middle_up
            ring_down   = not ring_up
            pinky_down  = not pinky_up

            thumb_index_close = (
                abs(x(lm, mp_hands.HandLandmark.THUMB_TIP) - x(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05 and
                abs(y(lm, mp_hands.HandLandmark.THUMB_TIP) - y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05
            )

            if thumb_up and index_up and middle_up and ring_up and pinky_up:
                gesture = "Halo"
            elif index_down and middle_down and ring_down and pinky_down:
                gesture = "Fist"
            elif thumb_up and index_down and middle_down and ring_down and pinky_down:
                gesture = "Sip!"
            elif thumb_index_close and middle_up and ring_up and pinky_up:
                gesture = "OK"
            elif index_up and middle_up and ring_down and pinky_down and thumb_down:
                gesture = "Peace"
            elif thumb_up and index_up and pinky_up and middle_down and ring_down:
                gesture = "I Love You"
            elif index_up and middle_down and ring_down and pinky_down and thumb_down:
                gesture = "Menunjuk"
            elif thumb_up and pinky_up and index_down and middle_down and ring_down:
                gesture = "Telepon"
            elif index_up and pinky_up and middle_down and ring_down and thumb_down:
                gesture = "Rock!"
            elif index_up and middle_up and ring_up and pinky_down and thumb_down:
                gesture = "Tiga"
            elif index_up and middle_up and ring_up and pinky_up and thumb_down:
                gesture = "Empat"
            elif middle_up and ring_up and index_down and pinky_down and thumb_down:
                gesture = "Spider-Man"

            wrist = lm[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            cx = int(wrist.x * w)
            cy = int(wrist.y * h)

            color = (255, 100, 0) if hand_label == "Right" else (0, 100, 255)
            cv2.putText(frame, results.multi_handedness, (cx - 30, cy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if gesture:
                cv2.putText(frame, gesture, (cx - 60, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)

    cv2.putText(frame, "Press Q to Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()