"""
PTU Student Face Recognition System - Streamlit Version
Pyay Technological University
"""
import streamlit as st
import sqlite3
import os
import pickle
import base64
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="PTU Face Recognition System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "database/ptu_students.db"
ADMIN_USER = "PTUAdmin"
ADMIN_PASS = "PTU2026"

MAJORS = [
    "Civil Engineering", "Electronic Engineering", "Mechanical Engineering",
    "Electrical Power Engineering", "Computer Engineering and Information Technology"
]
SEMESTERS = ["Seminar", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white; text-align: center;
}
.info-card {
    background: white; border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-left: 4px solid #0d6efd; margin-bottom: 1rem;
}
.result-card {
    background: #f0fff4; border-radius: 12px; padding: 1.5rem;
    border: 2px solid #28a745; margin-top: 1rem;
}
.stButton button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

def get_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, student_id TEXT UNIQUE NOT NULL,
            major TEXT NOT NULL, semester TEXT NOT NULL,
            roll_number TEXT NOT NULL, face_encodings BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL, image_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    return conn

def encode_face(img_array):
    try:
        import face_recognition
        encs = face_recognition.face_encodings(img_array)
        return encs[0] if encs else None
    except Exception:
        pass
    try:
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        roi = cv2.resize(gray[y:y+h, x:x+w], (128, 128)).flatten().astype(np.float64)
        n = np.linalg.norm(roi)
        return roi / n if n > 0 else roi
    except Exception:
        return None

def recognize(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    encoding = encode_face(np.array(img))
    if encoding is None:
        return None, "No face detected. Please use a clear front-facing photo."
    db = get_db()
    students = db.execute("SELECT * FROM students WHERE face_encodings IS NOT NULL").fetchall()
    best, best_dist = None, 0.6
    for s in students:
        try:
            stored = pickle.loads(s["face_encodings"])
            for enc in stored:
                dist = float(np.linalg.norm(np.array(enc) - np.array(encoding)))
                if dist < best_dist:
                    best_dist = dist
                    best = dict(s)
        except Exception:
            continue
    if best:
        best["confidence"] = round((1 - best_dist) * 100, 1)
        return best, None
    return None, "No matching student found in the database."

# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 PTU")
    st.markdown("**Face Recognition System**")
    st.markdown("*Pyay Technological University*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Search Student", "📷 Face Detection", "🔐 Admin Panel"],
        label_visibility="collapsed"
    )
    st.divider()
    db_stats = get_db()
    total = db_stats.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    trained = db_stats.execute("SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL").fetchone()[0]
    st.metric("Students", total)
    st.metric("Trained", trained)

# ── HOME ────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <div style="font-size:3rem;">🎓</div>
        <h1>PTU Student Face Recognition System</h1>
        <p>Pyay Technological University — Smart Attendance & Student Management</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="info-card">
            <h4>🤖 AI Face Recognition</h4>
            <p>Powered by OpenCV and face_recognition library for accurate detection and matching.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
            <h4>💾 Local SQLite Database</h4>
            <p>All student data and face encodings stored securely in a local SQLite database.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card">
            <h4>🔐 Admin Management</h4>
            <p>Secure admin panel for managing students, uploading face images, and training models.</p>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Use the sidebar to navigate between panels.")

# ── SEARCH ──────────────────────────────────────────────
elif page == "🔍 Search Student":
    st.markdown("## 🔍 Search Student by ID")
    st.markdown("Enter a student ID to retrieve their complete profile.")

    col1, col2 = st.columns([3, 1])
    with col1:
        student_id = st.text_input("Student ID", placeholder="e.g. PTU-2024-001", label_visibility="collapsed")
    with col2:
        search = st.button("🔍 Search", type="primary", use_container_width=True)

    if search:
        if student_id.strip():
            db = get_db()
            s = db.execute("SELECT * FROM students WHERE student_id = ?", (student_id.strip(),)).fetchone()
            if s:
                s = dict(s)
                st.success("✅ Student found!")
                st.markdown(f"""
                <div class="result-card">
                    <h3>👤 {s['name']}</h3>
                    <hr>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                        <div><b>Student ID:</b> <code>{s['student_id']}</code></div>
                        <div><b>Roll Number:</b> {s['roll_number']}</div>
                        <div><b>Major:</b> {s['major']}</div>
                        <div><b>Semester:</b> {s['semester']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ No student found with ID: `{student_id}`")
        else:
            st.warning("⚠️ Please enter a Student ID.")

# ── FACE DETECTION ──────────────────────────────────────
elif page == "📷 Face Detection":
    st.markdown("## 📷 Face Detection & Recognition")
    st.markdown("Upload a face image to identify a student.")

    uploaded = st.file_uploader(
        "Upload face image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear, front-facing photo for best results"
    )

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("🔍 Analyzing face..."):
                uploaded.seek(0)
                img_bytes = uploaded.read()
                result, error = recognize(img_bytes)

            if result:
                st.success(f"✅ Student Detected! Confidence: **{result['confidence']}%**")
                st.markdown("### Detected Student:")
                st.markdown(f"""
                <div class="result-card">
                    <p><b>Student Name:</b> {result['name']}</p>
                    <p><b>Student ID:</b> <code>{result['student_id']}</code></p>
                    <p><b>Major:</b> {result['major']}</p>
                    <p><b>Semester:</b> {result['semester']}</p>
                    <p><b>Roll Number:</b> {result['roll_number']}</p>
                </div>
                """, unsafe_allow_html=True)
                # Confidence bar
                st.progress(result['confidence'] / 100)
                st.caption(f"Match confidence: {result['confidence']}%")
            else:
                st.error(f"❌ {error}")

# ── ADMIN PANEL ─────────────────────────────────────────
elif page == "🔐 Admin Panel":
    if "admin_ok" not in st.session_state:
        st.markdown("## 🔐 Admin Login")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                st.markdown("### Administrator Access")
                u = st.text_input("Username", placeholder="PTUAdmin")
                p = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("🔐 Login", type="primary", use_container_width=True)
                if submitted:
                    if u == ADMIN_USER and p == ADMIN_PASS:
                        st.session_state.admin_ok = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")
    else:
        # Admin is logged in
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown("## ⚙️ Admin Panel")
        with col_right:
            if st.button("🚪 Logout"):
                del st.session_state.admin_ok
                st.rerun()

        admin_page = st.selectbox(
            "Section",
            ["📊 Dashboard", "👥 All Students", "➕ Add Student", "🧠 Train Faces", "🗑️ Manage Students"]
        )
        db = get_db()

        # Dashboard
        if admin_page == "📊 Dashboard":
            total = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            trained = db.execute("SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL").fetchone()[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("📚 Total Students", total)
            c2.metric("✅ Trained", trained, delta=None)
            c3.metric("⏳ Pending", total - trained)
            st.divider()
            st.subheader("Recent Students")
            rows = db.execute("SELECT * FROM students ORDER BY created_at DESC LIMIT 10").fetchall()
            if rows:
                for r in rows:
                    r = dict(r)
                    status = "✅ Trained" if r["face_encodings"] else "⏳ Untrained"
                    st.markdown(f"**{r['name']}** | `{r['student_id']}` | {r['major']} | {status}")
            else:
                st.info("No students yet.")

        # All Students
        elif admin_page == "👥 All Students":
            students = db.execute("SELECT * FROM students ORDER BY name").fetchall()
            st.write(f"**{len(students)} students registered**")
            for s in students:
                s = dict(s)
                with st.expander(f"{'✅' if s['face_encodings'] else '⏳'} {s['name']} — `{s['student_id']}`"):
                    col1, col2 = st.columns(2)
                    col1.write(f"**Major:** {s['major']}")
                    col1.write(f"**Semester:** {s['semester']}")
                    col2.write(f"**Roll No.:** {s['roll_number']}")
                    imgs = db.execute("SELECT COUNT(*) FROM face_images WHERE student_id=?", (s['id'],)).fetchone()[0]
                    col2.write(f"**Training Images:** {imgs}")

        # Add Student
        elif admin_page == "➕ Add Student":
            st.subheader("Add New Student")
            with st.form("add_form", clear_on_submit=True):
                name = st.text_input("Student Name *")
                col1, col2 = st.columns(2)
                with col1:
                    student_id = st.text_input("Student ID *", placeholder="PTU-2024-001")
                    major = st.selectbox("Major *", MAJORS)
                with col2:
                    roll = st.text_input("Roll Number *")
                    semester = st.selectbox("Semester *", SEMESTERS)
                submitted = st.form_submit_button("➕ Add Student", type="primary")
                if submitted:
                    if all([name, student_id, major, semester, roll]):
                        try:
                            db.execute(
                                "INSERT INTO students (name,student_id,major,semester,roll_number) VALUES (?,?,?,?,?)",
                                (name.strip(), student_id.strip(), major, semester, roll.strip())
                            )
                            db.commit()
                            st.success(f"✅ **{name}** added successfully! Now go to 'Train Faces' to add face images.")
                        except sqlite3.IntegrityError:
                            st.error(f"❌ Student ID `{student_id}` already exists.")
                    else:
                        st.error("❌ All fields are required.")

        # Train Faces
        elif admin_page == "🧠 Train Faces":
            st.subheader("Train Face Recognition")
            students = db.execute("SELECT id, name, student_id FROM students ORDER BY name").fetchall()
            if not students:
                st.info("No students yet. Add students first.")
            else:
                opts = {f"{s['name']} ({s['student_id']})": s['id'] for s in students}
                sel = st.selectbox("Select Student to Train", list(opts.keys()))
                sid = opts[sel]
                imgs_count = db.execute("SELECT COUNT(*) FROM face_images WHERE student_id=?", (sid,)).fetchone()[0]
                col1, col2 = st.columns(2)
                col1.metric("Current Images", imgs_count)
                col2.metric("Minimum Required", "2")
                if imgs_count < 2:
                    st.warning(f"⚠️ Need {2 - imgs_count} more image(s) for training.")

                files = st.file_uploader(
                    "Upload face images (minimum 2)",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True
                )
                if files:
                    cols = st.columns(min(len(files), 4))
                    for i, f in enumerate(files):
                        with cols[i % 4]:
                            st.image(f, use_column_width=True)

                if st.button("🧠 Upload & Train", type="primary") and files:
                    progress = st.progress(0)
                    saved = 0
                    for i, f in enumerate(files):
                        try:
                            img_bytes = f.read()
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            enc = encode_face(np.array(img))
                            if enc is None:
                                st.warning(f"⚠️ No face in {f.name}")
                                continue
                            b64 = base64.b64encode(img_bytes).decode()
                            db.execute("INSERT INTO face_images (student_id, image_data) VALUES (?,?)", (sid, b64))
                            saved += 1
                        except Exception as e:
                            st.warning(f"Error: {e}")
                        progress.progress((i + 1) / len(files))
                    db.commit()
                    if saved:
                        all_imgs = db.execute("SELECT image_data FROM face_images WHERE student_id=?", (sid,)).fetchall()
                        encodings = []
                        for ir in all_imgs:
                            try:
                                ib = base64.b64decode(ir["image_data"])
                                img = Image.open(io.BytesIO(ib)).convert("RGB")
                                enc = encode_face(np.array(img))
                                if enc is not None:
                                    encodings.append(enc)
                            except Exception:
                                continue
                        if encodings:
                            db.execute("UPDATE students SET face_encodings=? WHERE id=?", (pickle.dumps(encodings), sid))
                            db.commit()
                            st.success(f"✅ {saved} image(s) uploaded and trained! Total encodings: {len(encodings)}")
                        else:
                            st.error("❌ No valid face encodings generated.")
                    else:
                        st.error("❌ No valid faces detected. Please upload clear face photos.")

        # Manage Students
        elif admin_page == "🗑️ Manage Students":
            st.subheader("Manage Students")
            students = db.execute("SELECT * FROM students ORDER BY name").fetchall()
            for s in students:
                s = dict(s)
                with st.expander(f"{s['name']} — `{s['student_id']}`"):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_name = st.text_input("Name", s['name'], key=f"n_{s['id']}")
                        new_major = st.selectbox("Major", MAJORS, index=MAJORS.index(s['major']) if s['major'] in MAJORS else 0, key=f"m_{s['id']}")
                    with col2:
                        new_roll = st.text_input("Roll No.", s['roll_number'], key=f"r_{s['id']}")
                        new_sem = st.selectbox("Semester", SEMESTERS, index=SEMESTERS.index(s['semester']) if s['semester'] in SEMESTERS else 0, key=f"s_{s['id']}")
                    c1, c2 = st.columns(2)
                    if c1.button("💾 Update", key=f"upd_{s['id']}"):
                        db.execute("UPDATE students SET name=?,major=?,semester=?,roll_number=? WHERE id=?",
                                   (new_name, new_major, new_sem, new_roll, s['id']))
                        db.commit()
                        st.success("Updated!")
                        st.rerun()
                    if c2.button("🗑️ Delete", key=f"del_{s['id']}"):
                        db.execute("DELETE FROM students WHERE id=?", (s['id'],))
                        db.execute("DELETE FROM face_images WHERE student_id=?", (s['id'],))
                        db.commit()
                        st.success("Deleted!")
                        st.rerun()
