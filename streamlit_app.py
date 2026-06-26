"""
PTU Student Face Recognition System - Streamlit Version
Complete port of app.py with identical features, database schema, and workflow.
"""
import streamlit as st
import sqlite3
import os
import sys
import pickle
import base64
import numpy as np
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.face_utils import encode_face, compare_faces

st.set_page_config(
    page_title="PTU Face Recognition System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

ADMIN_USER = 'PTUAdmin'
ADMIN_PASS = 'PTU2026'

MAJORS = [
    'Civil Engineering',
    'Electronic Engineering',
    'Mechanical Engineering',
    'Electrical Power Engineering',
    'Computer Engineering and Information Technology',
]

SEMESTERS = ['Semester', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'ptu_students.db')

st.markdown("""
<style>
.main-header { background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white; text-align: center; }
.info-card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-left: 4px solid #0d6efd; margin-bottom: 1rem; }
.result-card { background: #f0fff4; border-radius: 12px; padding: 1.5rem; border: 2px solid #28a745; margin-top: 1rem; }
.stButton button { border-radius: 8px; font-weight: 600; }
div[data-testid="stMetric"] { background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ─── Database ────────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            major TEXT NOT NULL,
            semester TEXT NOT NULL,
            roll_number TEXT NOT NULL,
            face_encodings BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        );
    ''')
    conn.commit()
    conn.close()

@st.cache_resource
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_db():
    return get_connection()

init_db()

# ─── Face Recognition ────────────────────────────────────────────────────────

def recognize_from_bytes(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
        encoding = encode_face(img_array)
        if encoding is None:
            return None, 'No face detected in the image. Please try again with a clear face photo.'
        db = get_db()
        students = db.execute('SELECT * FROM students WHERE face_encodings IS NOT NULL').fetchall()
        best_match = None
        best_distance = 0.6
        for student in students:
            try:
                stored = pickle.loads(student['face_encodings'])
                for enc in stored:
                    dist = compare_faces(enc, encoding)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = dict(student)
            except Exception:
                continue
        if best_match:
            best_match['confidence'] = round((1 - best_distance) * 100, 1)
            return best_match, None
        return None, 'No matching student found in the database.'
    except Exception as e:
        return None, f'Error processing image: {str(e)}'

def retrain_student(sid, db):
    images = db.execute('SELECT image_data FROM face_images WHERE student_id = ?', (sid,)).fetchall()
    if len(images) < 1:
        db.execute('UPDATE students SET face_encodings = NULL WHERE id = ?', (sid,))
        db.commit()
        return False
    encodings = []
    for img_row in images:
        try:
            img_bytes = base64.b64decode(img_row['image_data'])
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(img)
            enc = encode_face(img_array)
            if enc is not None:
                encodings.append(enc)
        except Exception:
            continue
    if encodings:
        db.execute('UPDATE students SET face_encodings = ? WHERE id = ?', (pickle.dumps(encodings), sid))
        db.commit()
        return True
    return False

def show_student_details(s):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Student Name:** {s['name']}")
        st.markdown(f"**Student ID:** `{s['student_id']}`")
        st.markdown(f"**Major:** {s['major']}")
    with col2:
        st.markdown(f"**Semester:** {s['semester']}")
        st.markdown(f"**Roll Number:** {s['roll_number']}")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

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
    db_side = get_db()
    total_s = db_side.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    trained_s = db_side.execute("SELECT COUNT(*) FROM students WHERE face_encodings IS NOT NULL").fetchone()[0]
    st.metric("Total Students", total_s)
    st.metric("Trained", trained_s)
    st.metric("Pending", total_s - trained_s)

# ─── Home ────────────────────────────────────────────────────────────────────

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

    st.info("👈 Use the sidebar to navigate between panels. Search students by ID, detect faces with AI, or access the admin panel.")

# ─── Search Student ──────────────────────────────────────────────────────────

elif page == "🔍 Search Student":
    st.markdown("## 🔍 Search Student by ID")
    st.markdown("Enter a student ID to retrieve their complete profile.")

    col1, col2 = st.columns([3, 1])
    with col1:
        student_id_input = st.text_input("Student ID", placeholder="e.g. PTU-2024-001", label_visibility="collapsed")
    with col2:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    if search_clicked:
        if not student_id_input.strip():
            st.warning("⚠️ Please enter a Student ID.")
        else:
            db = get_db()
            s = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id_input.strip(),)).fetchone()
            if s:
                s = dict(s)
                st.success("✅ Student found!")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                show_student_details(s)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"❌ No student found with ID: `{student_id_input.strip()}`")

# ─── Face Detection ──────────────────────────────────────────────────────────

elif page == "📷 Face Detection":
    st.markdown("## 📷 Face Detection & Recognition")
    st.markdown("Upload a photo or use your camera to identify a student.")

    detect_tab_upload, detect_tab_camera = st.tabs(["📁 Upload Image", "📸 Use Camera"])

    with detect_tab_upload:
        uploaded_file = st.file_uploader(
            "Upload face image",
            type=["jpg", "jpeg", "png"],
            help="Use a clear, front-facing photo for best results",
            key="detect_upload"
        )
        if uploaded_file:
            img_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption="Uploaded Image", width=300)
            with st.spinner("🔍 Analyzing face..."):
                result, error = recognize_from_bytes(img_bytes)
            if result:
                st.success(f"✅ Student Detected! Confidence: **{result['confidence']}%**")
                st.progress(result['confidence'] / 100)
                st.caption(f"Match confidence: {result['confidence']}%")
                st.markdown("### Detected Student:")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                show_student_details(result)
                st.markdown('</div>', unsafe_allow_html=True)
            elif error:
                st.error(f"❌ {error}")

    with detect_tab_camera:
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            img_bytes = camera_image.getvalue()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption="Captured Photo", width=300)
            with st.spinner("🔍 Analyzing face..."):
                result, error = recognize_from_bytes(img_bytes)
            if result:
                st.success(f"✅ Student Detected! Confidence: **{result['confidence']}%**")
                st.progress(result['confidence'] / 100)
                st.caption(f"Match confidence: {result['confidence']}%")
                st.markdown("### Detected Student:")
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                show_student_details(result)
                st.markdown('</div>', unsafe_allow_html=True)
            elif error:
                st.error(f"❌ {error}")

# ─── Admin Panel ─────────────────────────────────────────────────────────────

elif page == "🔐 Admin Panel":
    if 'admin_logged_in' not in st.session_state:
        st.markdown("## 🔐 Admin Login")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                st.markdown("### Administrator Access")
                username = st.text_input("Username", placeholder="PTUAdmin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("🔐 Login", type="primary", use_container_width=True)
                if submitted:
                    if username == ADMIN_USER and password == ADMIN_PASS:
                        st.session_state.admin_logged_in = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
    else:
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown("## ⚙️ Admin Panel")
        with col_right:
            if st.button("🚪 Logout"):
                del st.session_state.admin_logged_in
                if 'admin_select' in st.session_state:
                    del st.session_state.admin_select
                st.rerun()

        admin_options = ["📊 Dashboard", "👥 All Students", "➕ Add Student", "🧠 Train Faces"]

        if 'admin_page_override' in st.session_state:
            st.session_state.admin_select = st.session_state.pop('admin_page_override')
        if 'admin_select' not in st.session_state:
            st.session_state.admin_select = "📊 Dashboard"

        admin_page = st.selectbox("Section", admin_options, key="admin_select")

        db = get_db()

        # ── Dashboard ────────────────────────────────────────────────────────
        if admin_page == "📊 Dashboard":
            total = db.execute('SELECT COUNT(*) as c FROM students').fetchone()['c']
            trained = db.execute('SELECT COUNT(*) as c FROM students WHERE face_encodings IS NOT NULL').fetchone()['c']
            recent = db.execute('SELECT * FROM students ORDER BY created_at DESC LIMIT 5').fetchall()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("📚 Total Students", total)
            with c2:
                st.metric("✅ Trained Students", trained)
            with c3:
                st.metric("⏳ Pending Training", total - trained)

            # Quick Actions
            st.subheader("Quick Actions")
            qa1, qa2, qa3, qa4 = st.columns(4)
            with qa1:
                if st.button("➕ Add Student", use_container_width=True):
                    st.session_state.admin_page_override = "➕ Add Student"
                    st.rerun()
            with qa2:
                if st.button("👥 View All Students", use_container_width=True):
                    st.session_state.admin_page_override = "👥 All Students"
                    st.rerun()
            with qa3:
                if st.button("🔄 Retrain All", use_container_width=True, type="primary"):
                    students_all = db.execute('SELECT id FROM students').fetchall()
                    count = 0
                    for s in students_all:
                        if retrain_student(s['id'], db):
                            count += 1
                    st.success(f"✅ Retrained {count} students successfully!")
                    st.rerun()
            with qa4:
                st.markdown("🔍 **Test Detection:** Use the sidebar to navigate to 📷 Face Detection.")

            # Recent Students
            st.subheader("Recently Added Students")
            if recent:
                for r in recent:
                    r = dict(r)
                    status = "✅ Trained" if r['face_encodings'] else "⏳ Pending"
                    cols = st.columns([2, 2, 2, 1, 1])
                    with cols[0]:
                        st.markdown(f"**{r['name']}**")
                    with cols[1]:
                        st.markdown(f"`{r['student_id']}`")
                    with cols[2]:
                        st.markdown(r['major'])
                    with cols[3]:
                        st.markdown(r['semester'])
                    with cols[4]:
                        st.markdown(status)
            else:
                st.info("No students yet.")

        # ── All Students ─────────────────────────────────────────────────────
        elif admin_page == "👥 All Students":
            students = db.execute('SELECT * FROM students ORDER BY created_at DESC').fetchall()
            st.write(f"**{len(students)} total records**")

            search_q = st.text_input("🔍 Search by name, ID, major...", placeholder="Type to filter...")
            filtered = [dict(s) for s in students]
            if search_q:
                q = search_q.lower()
                filtered = [s for s in filtered if q in s['name'].lower() or q in s['student_id'].lower() or q in s['major'].lower()]

            if not filtered:
                st.info("No students matching your search.")

            for s in filtered:
                with st.expander(f"{'✅' if s['face_encodings'] else '⏳'} {s['name']} — `{s['student_id']}`"):
                    cols = st.columns([3, 2, 2, 2])
                    with cols[0]:
                        st.markdown(f"**Major:** {s['major']}")
                        st.markdown(f"**Semester:** {s['semester']}")
                    with cols[1]:
                        st.markdown(f"**Roll No.:** {s['roll_number']}")
                        img_count = db.execute("SELECT COUNT(*) as c FROM face_images WHERE student_id=?", (s['id'],)).fetchone()['c']
                        st.markdown(f"**Images:** {img_count}")
                    with cols[2]:
                        if st.button("🧠 Train", key=f"train_{s['id']}"):
                            st.session_state['train_sid'] = s['id']
                            st.session_state.admin_page_override = "🧠 Train Faces"
                            st.rerun()
                    with cols[3]:
                        edit_col, del_col = st.columns(2)
                        with edit_col:
                            if st.button("✏️ Edit", key=f"edit_{s['id']}"):
                                st.session_state['edit_sid'] = s['id']
                                st.rerun()
                        with del_col:
                            if st.button("🗑️", key=f"del_{s['id']}"):
                                db.execute('DELETE FROM students WHERE id = ?', (s['id'],))
                                db.execute('DELETE FROM face_images WHERE student_id = ?', (s['id'],))
                                db.commit()
                                st.success(f"Deleted {s['name']}.")
                                st.rerun()

                    if 'edit_sid' in st.session_state and st.session_state['edit_sid'] == s['id']:
                        st.divider()
                        st.markdown("#### Edit Student")
                        with st.form(key=f"edit_form_{s['id']}"):
                            ename = st.text_input("Student Name", value=s['name'])
                            eid = st.text_input("Student ID", value=s['student_id'])
                            emajor = st.selectbox("Major", MAJORS, index=MAJORS.index(s['major']) if s['major'] in MAJORS else 0)
                            esem = st.selectbox("Semester", SEMESTERS, index=SEMESTERS.index(s['semester']) if s['semester'] in SEMESTERS else 0)
                            eroll = st.text_input("Roll Number", value=s['roll_number'])
                            sub1, sub2 = st.columns(2)
                            with sub1:
                                if st.form_submit_button("💾 Save Changes", type="primary"):
                                    db.execute(
                                        'UPDATE students SET name=?, student_id=?, major=?, semester=?, roll_number=? WHERE id=?',
                                        (ename.strip(), eid.strip(), emajor, esem, eroll.strip(), s['id'])
                                    )
                                    db.commit()
                                    st.success("✅ Student updated successfully!")
                                    del st.session_state['edit_sid']
                                    st.rerun()
                            with sub2:
                                if st.form_submit_button("Cancel"):
                                    del st.session_state['edit_sid']
                                    st.rerun()

        # ── Add Student ──────────────────────────────────────────────────────
        elif admin_page == "➕ Add Student":
            st.subheader("➕ Add New Student")
            with st.form("add_student_form", clear_on_submit=True):
                name = st.text_input("Student Name *")
                col1, col2 = st.columns(2)
                with col1:
                    student_id = st.text_input("Student ID *", placeholder="e.g. PTU-2024-001")
                    major = st.selectbox("Major *", MAJORS)
                with col2:
                    roll_number = st.text_input("Roll Number *")
                    semester = st.selectbox("Semester *", SEMESTERS)
                submitted = st.form_submit_button("➕ Add Student", type="primary")
                if submitted:
                    if not all([name, student_id, major, semester, roll_number]):
                        st.error("❌ All fields are required.")
                    else:
                        existing = db.execute('SELECT id FROM students WHERE student_id = ?', (student_id.strip(),)).fetchone()
                        if existing:
                            st.error(f"❌ Student ID `{student_id}` already exists.")
                        else:
                            db.execute(
                                'INSERT INTO students (name, student_id, major, semester, roll_number) VALUES (?, ?, ?, ?, ?)',
                                (name.strip(), student_id.strip(), major, semester, roll_number.strip())
                            )
                            db.commit()
                            st.success(f"✅ Student **{name}** added successfully!")

        # ── Train Faces ──────────────────────────────────────────────────────
        elif admin_page == "🧠 Train Faces":
            st.subheader("🧠 Train Face Recognition")

            students = db.execute('SELECT id, name, student_id FROM students ORDER BY name').fetchall()

            if not students:
                st.info("No students yet. Add students first.")

            default_idx = 0
            if 'train_sid' in st.session_state:
                for i, s in enumerate(students):
                    if s['id'] == st.session_state['train_sid']:
                        default_idx = i
                        break
                del st.session_state['train_sid']

            opts = {f"{s['name']} ({s['student_id']})": s['id'] for s in students}
            if opts:
                sel = st.selectbox("Select Student to Train", list(opts.keys()), index=default_idx if default_idx < len(opts) else 0)
                sid = opts[sel]

                student = dict(db.execute('SELECT * FROM students WHERE id = ?', (sid,)).fetchone())

                col_info, col_upload = st.columns([1, 1])

                with col_info:
                    st.markdown(f'<div class="info-card">', unsafe_allow_html=True)
                    st.markdown(f"### 👤 {student['name']}")
                    st.markdown(f"**ID:** `{student['student_id']}`")
                    st.markdown(f"**Major:** {student['major']}")
                    st.markdown(f"**Semester:** {student['semester']}")
                    st.markdown(f"**Roll No.:** {student['roll_number']}")
                    if student['face_encodings']:
                        st.success("✅ Face Trained")
                    else:
                        st.warning("⚠️ Not Trained Yet")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_upload:
                    st.markdown("### 📤 Upload Face Images")
                    st.info("Upload **at least 2 face images** for accurate recognition. Use clear, front-facing photos with good lighting.")

                    files = st.file_uploader(
                        "Choose face images",
                        type=["jpg", "jpeg", "png"],
                        accept_multiple_files=True,
                        key="train_upload"
                    )

                    if st.button("🧠 Upload & Train", type="primary", disabled=not files):
                        saved = 0
                        progress_bar = st.progress(0)
                        for i, f in enumerate(files):
                            try:
                                img_bytes = f.read()
                                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                                img_array = np.array(img)
                                enc = encode_face(img_array)
                                if enc is None:
                                    continue
                                img_b64 = base64.b64encode(img_bytes).decode()
                                db.execute(
                                    'INSERT INTO face_images (student_id, image_data) VALUES (?, ?)',
                                    (sid, img_b64)
                                )
                                saved += 1
                            except Exception:
                                continue
                            progress_bar.progress((i + 1) / len(files))
                        db.commit()

                        if saved > 0:
                            retrain_student(sid, db)
                            st.success(f"✅ {saved} image(s) uploaded and trained successfully!")
                            st.rerun()
                        else:
                            st.error("❌ No valid face images detected. Please upload clear face photos.")

                # Existing Images Gallery
                st.divider()
                st.subheader("Training Images")
                images = db.execute('SELECT * FROM face_images WHERE student_id = ?', (sid,)).fetchall()
                img_count = len(images)

                col_count, col_status = st.columns([1, 1])
                col_count.metric("Total Images", img_count)
                if img_count < 2:
                    col_status.warning(f"⚠️ Need {2 - img_count} more image(s) for reliable training.")
                else:
                    col_status.success("✅ Minimum images met")

                if images:
                    cols_per_row = 4
                    for row_start in range(0, len(images), cols_per_row):
                        row_imgs = images[row_start:row_start + cols_per_row]
                        cols = st.columns(cols_per_row)
                        for col_idx, img_row in enumerate(row_imgs):
                            with cols[col_idx]:
                                img = dict(img_row)
                                st.image(
                                    f"data:image/jpeg;base64,{img['image_data']}",
                                    use_container_width=True
                                )
                                st.caption(f"Image {row_start + col_idx + 1}")
                                if st.button("🗑️ Delete", key=f"del_img_{img['id']}"):
                                    db.execute('DELETE FROM face_images WHERE id = ? AND student_id = ?', (img['id'], sid))
                                    db.commit()
                                    retrain_student(sid, db)
                                    st.success("✅ Image deleted and model retrained.")
                                    st.rerun()
                else:
                    st.info("No training images yet. Upload face images above.")