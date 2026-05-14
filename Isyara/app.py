from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import cv2
import mediapipe as mp
import pickle
import threading
import os
from dotenv import load_dotenv
from google import genai
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

app = Flask(__name__)
app.secret_key = "isyara_secret_key_2026"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ── MODEL USER ───────────────────────────────────────────────
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role     = db.Column(db.String(20), default="siswa")

# ── MODEL TAMBAHAN ──────────────────────────────────────────
class Sekolah(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    nama    = db.Column(db.String(120), unique=True, nullable=False)
    kota    = db.Column(db.String(80), default="")
    created = db.Column(db.String(20), default="")

class Kelas(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    nama_kelas = db.Column(db.String(80), nullable=False)
    sekolah_id = db.Column(db.Integer, db.ForeignKey('sekolah.id'), nullable=True)
    guru_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created    = db.Column(db.String(20), default="")
    sekolah    = db.relationship('Sekolah', backref='kelas')
    guru       = db.relationship('User', backref='kelas')

with app.app_context():
    db.create_all()
    # Buat akun admin default kalau belum ada
    if not User.query.filter_by(username="admin").first():
        admin = User(
            username="admin",
            password=generate_password_hash("admin123"),
            role="admin",
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Akun admin default dibuat (username: admin, password: admin123)")

# ── LOAD MODEL ───────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "alfabet_model.pkl"), "rb") as f:
    alfabet_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "angka_model.pkl"), "rb") as f:
    angka_model = pickle.load(f)

# Mode aktif: 'alfabet' atau 'angka'
current_mode = "alfabet"

# ── MEDIAPIPE ────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ── GEMINI CLIENT ────────────────────────────────────────────
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── STATE DETEKSI ────────────────────────────────────────────
state = {"gesture": "", "confidence": 0, "sentence": "", "history": [], "mode": "alfabet"}
lock         = threading.Lock()   # ← harus ada di sini
stable_count = 0
last_letter  = ""
STABLE_THRESHOLD = 5

# ── HELPER MEDIAPIPE ─────────────────────────────────────────
def get_y(lm, id): return lm[id].y
def get_x(lm, id): return lm[id].x


def detect_basic(lm):
    thu = get_y(lm, mp_hands.HandLandmark.THUMB_TIP)         < get_y(lm, mp_hands.HandLandmark.THUMB_IP)
    idx = get_y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)  < get_y(lm, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    mid = get_y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP) < get_y(lm, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    rng = get_y(lm, mp_hands.HandLandmark.RING_FINGER_TIP)   < get_y(lm, mp_hands.HandLandmark.RING_FINGER_PIP)
    pnk = get_y(lm, mp_hands.HandLandmark.PINKY_TIP)         < get_y(lm, mp_hands.HandLandmark.PINKY_PIP)
    ti_close = (
        abs(get_x(lm, mp_hands.HandLandmark.THUMB_TIP) - get_x(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05
        and abs(get_y(lm, mp_hands.HandLandmark.THUMB_TIP) - get_y(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)) < 0.05
    )
    if thu and idx and mid and rng and pnk:                 return "Halo"
    if thu and not idx and not mid and not rng and not pnk: return "Sip!"
    if ti_close and mid and rng and pnk:                    return "OK"
    if idx and mid and not rng and not pnk and not thu:     return "Peace"
    if thu and idx and pnk and not mid and not rng:         return "I Love You"
    if idx and not mid and not rng and not pnk and not thu: return "Menunjuk"
    if thu and pnk and not idx and not mid and not rng:     return "Telepon"
    if idx and pnk and not mid and not rng and not thu:     return "Rock!"
    return None

def process_frame(frame):
    global stable_count, last_letter, current_mode

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted  = ""
    confidence = 0
    left_lm    = None
    right_lm   = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            lbl = handedness.classification[0].label
            if lbl == "Left":
                left_lm = hand_landmarks.landmark
            else:
                right_lm = hand_landmarks.landmark

        check_lm = right_lm if right_lm else left_lm
        if check_lm:
            basic = detect_basic(check_lm)
            if basic:
                predicted  = basic
                confidence = 100

        if not predicted:
            zero    = [0.0] * 63
            hand_lm = right_lm if right_lm else left_lm
            row     = [c for lm in hand_lm for c in [lm.x, lm.y, lm.z]] if hand_lm else zero
            print(f"ROW LENGTH: {len(row)}, MODE: {current_mode}")
            active_model = alfabet_model if current_mode == "alfabet" else angka_model
            try:
                predicted  = active_model.predict([row])[0]
                confidence = round(max(active_model.predict_proba([row])[0]) * 100, 1)
                print(f"PREDICT: {predicted} {confidence}%")
            except Exception as e:
                print(f"ERROR MODEL: {e}")
                predicted  = ""
                confidence = 0

    with lock:
        if predicted and predicted == last_letter:
            stable_count += 1
            if stable_count == STABLE_THRESHOLD:
                state["sentence"] += str(predicted)
                state["history"].append(str(predicted))
                if len(state["history"]) > 30:
                    state["history"].pop(0)
        else:
            stable_count = 0
            last_letter  = predicted
        state["gesture"]    = predicted
        state["confidence"] = confidence
        state["mode"]       = current_mode

    return predicted, confidence

# ── DECORATOR ADMIN ──────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("username"):
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted, confidence = process_frame(frame.copy())

        frame_show = cv2.flip(frame, 1)
        rgb        = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB)
        results    = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_show, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                wrist   = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, _ = frame_show.shape
                cx, cy  = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame_show, f"{predicted} ({confidence}%)",
                            (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        _, buffer = cv2.imencode(".jpg", frame_show, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ── HELPER GEMINI ─────────────────────────────────────────────
def ask_gemini(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


# ── ROUTES UTAMA ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", user=session.get("username"))


@app.route("/dashboard")
def dashboard():
    username = session.get("username")
    if not username:
        return redirect(url_for("login"))
    role = session.get("role", "siswa")
    if role == "admin":
        return redirect(url_for("admin_dashboard"))
    if role == "guru":
        all_users = User.query.filter_by(role="siswa").all()
        return render_template("dashboard_guru.html", user=username, users=all_users)
    return render_template("dashboard_siswa.html", user=username)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        action   = request.form.get("action")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if action == "login":
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                session["username"] = username
                session["role"]     = user.role
                return redirect(url_for("dashboard"))
            else:
                error = "Username atau password salah!"

        elif action == "register":
            if User.query.filter_by(username=username).first():
                error = "Username sudah dipakai!"
            elif len(password) < 6:
                error = "Password minimal 6 karakter!"
            else:
                role = request.form.get("role", "siswa")
                # Register hanya boleh siswa atau guru, tidak bisa langsung admin
                if role not in ("siswa", "guru"):
                    role = "siswa"
                new_user = User(
                    username=username,
                    password=generate_password_hash(password),
                    role=role,
                )
                db.session.add(new_user)
                db.session.commit()
                session["username"] = username
                session["role"]     = role
                return redirect(url_for("dashboard"))

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("role", None)
    return redirect(url_for("login"))


@app.route("/game")
def game():
    if not session.get("username"):
        return redirect(url_for("login"))
    return render_template("game.html", user=session.get("username"))


@app.route("/kamus")
def kamus():
    if not session.get("username"):
        return redirect(url_for("login"))
    return render_template("kamus.html", user=session.get("username"))


@app.route("/dataset")
def dataset():
    import pandas as pd
    DATA_FILE = os.path.join(BASE_DIR, "dataset.csv")
    if not os.path.exists(DATA_FILE):
        return render_template("dataset.html", error="File dataset.csv tidak ditemukan.", stats=[], total=0)
    df    = pd.read_csv(DATA_FILE, on_bad_lines="skip")
    stats = df.groupby("label").size().reset_index(name="jumlah").to_dict(orient="records")
    return render_template("dataset.html", stats=stats, total=len(df), error=None)


# ── ROUTES KAMERA ─────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    with lock:
        return jsonify(state)


@app.route("/clear_sentence")
def clear_sentence():
    with lock:
        state["sentence"] = ""
        state["history"]  = []
    return jsonify({"status": "ok"})


@app.route("/backspace")
def backspace():
    with lock:
        state["sentence"] = state["sentence"][:-1]
        if state["history"]:
            state["history"].pop()
    return jsonify({"status": "ok"})


@app.route("/add_text", methods=["POST"])
def add_text():
    text = request.json.get("text", "").upper()
    with lock:
        state["sentence"] += text
        for ch in text:
            state["history"].append(ch)
        if len(state["history"]) > 30:
            state["history"] = state["history"][-30:]
    return jsonify({"status": "ok"})

@app.route("/set_mode/<mode>")
def set_mode(mode):
    global current_mode
    if mode in ("alfabet", "angka"):
        current_mode = mode
        with lock:
            state["sentence"] = ""
            state["history"]  = []
            state["mode"]     = mode
    return jsonify({"status": "ok", "mode": current_mode})


# ── ROUTES ADMIN ──────────────────────────────────────────────

@app.route("/admin")
@admin_required
def admin_dashboard():
    return render_template(
        "dashboard_admin.html",
        user=session.get("username"),
        total_siswa  = User.query.filter_by(role="siswa").count(),
        total_guru   = User.query.filter_by(role="guru").count(),
        total_admin  = User.query.filter_by(role="admin").count(),
        total        = User.query.count(),
    )


@app.route("/admin/api/users")
@admin_required
def admin_api_users():
    role  = request.args.get("role")
    q     = User.query
    if role in ("siswa", "guru", "admin"):
        q = q.filter_by(role=role)
    users = q.order_by(User.role, User.username).all()
    return jsonify([
        {"id": u.id, "username": u.username, "role": u.role}
        for u in users
    ])


@app.route("/admin/add_user", methods=["POST"])
@admin_required
def admin_add_user():
    data     = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    role     = data.get("role", "siswa")

    if not username or not password:
        return jsonify({"status": "error", "message": "Username dan password wajib diisi"}), 400
    if len(password) < 6:
        return jsonify({"status": "error", "message": "Password minimal 6 karakter"}), 400
    if role not in ("siswa", "guru", "admin"):
        return jsonify({"status": "error", "message": "Role tidak valid"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"status": "error", "message": "Username sudah dipakai"}), 409

    db.session.add(User(
        username=username,
        password=generate_password_hash(password),
        role=role,
    ))
    db.session.commit()
    return jsonify({"status": "ok", "message": f"Pengguna {username} berhasil ditambahkan"})


@app.route("/admin/edit_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_edit_user(user_id):
    user     = User.query.get_or_404(user_id)
    data     = request.json
    new_role = data.get("role")
    new_pass = data.get("password", "").strip()

    if new_role in ("siswa", "guru", "admin"):
        user.role = new_role
    if new_pass and len(new_pass) >= 6:
        user.password = generate_password_hash(new_pass)

    db.session.commit()
    return jsonify({"status": "ok", "message": f"{user.username} diperbarui"})


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.username == session.get("username"):
        return jsonify({"status": "error", "message": "Tidak bisa menghapus akun sendiri"}), 403
    db.session.delete(user)
    db.session.commit()
    return jsonify({"status": "ok", "message": f"{user.username} dihapus"})


@app.route("/admin/stats")
@admin_required
def admin_stats():
    return jsonify({
        "total":       User.query.count(),
        "total_siswa": User.query.filter_by(role="siswa").count(),
        "total_guru":  User.query.filter_by(role="guru").count(),
        "total_admin": User.query.filter_by(role="admin").count(),
    })


# ── ROUTES AI ────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data             = request.json
    user_message     = data.get("message", "")
    gesture_context  = data.get("gesture", "tidak ada")
    sentence_context = data.get("sentence", "kosong")

    prompt = f"""Kamu adalah asisten AI untuk platform ISYARA - platform pembelajaran bahasa isyarat Indonesia (BISINDO).
Tugasmu: menjawab pertanyaan BISINDO, memberi feedback, rekomendasi latihan, dan motivasi.

Konteks saat ini:
- Gesture terdeteksi: {gesture_context}
- Kalimat terbentuk: {sentence_context}

Jawab dalam Bahasa Indonesia, singkat, ramah, maksimal 3 kalimat.

User: {user_message}"""

    try:
        reply = ask_gemini(prompt)
        return jsonify({"reply": reply, "status": "ok"})
    except Exception:
        return jsonify({"reply": "Maaf, AI sedang tidak tersedia. Coba lagi ya!", "status": "error"})


@app.route("/ai_feedback", methods=["POST"])
def ai_feedback():
    data       = request.json
    gesture    = data.get("gesture", "")
    confidence = data.get("confidence", 0)
    sentence   = data.get("sentence", "")

    prompt = f"""Berikan feedback singkat untuk pengguna belajar bahasa isyarat BISINDO:
- Gesture: {gesture}
- Kepercayaan AI: {confidence}%
- Kalimat: {sentence}

Berikan penilaian, saran perbaikan jika perlu, dan motivasi. Maksimal 3 kalimat, ramah."""

    try:
        feedback = ask_gemini(prompt)
        return jsonify({"feedback": feedback, "status": "ok"})
    except Exception:
        return jsonify({"feedback": "Terus semangat berlatih!", "status": "error"})


@app.route("/ai_rekomendasi", methods=["POST"])
def ai_rekomendasi():
    data          = request.json
    history       = data.get("history", [])
    sentence      = data.get("sentence", "")
    huruf_dilatih = list(set(history)) if history else []

    prompt = f"""Kamu adalah AI tutor bahasa isyarat BISINDO.
Pengguna sudah berlatih huruf: {huruf_dilatih if huruf_dilatih else 'belum ada'}
Kalimat terakhir: {sentence if sentence else 'belum ada'}

Rekomendasikan 3 huruf berikutnya dan 1 kata sederhana yang bisa dicoba. Maksimal 4 kalimat, menarik."""

    try:
        rekomendasi = ask_gemini(prompt)
        return jsonify({"rekomendasi": rekomendasi, "status": "ok"})
    except Exception:
        return jsonify({"rekomendasi": "Coba latihan huruf A, B, C dulu ya!", "status": "error"})


if __name__ == "__main__":
    import os
    # Render akan mengisi variable PORT secara otomatis
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting ISYARA on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
