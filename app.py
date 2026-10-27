import os
import sqlite3
from datetime import datetime, timezone
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from nlp_engine import (extract_text, extract_skills, predict_roles,
                        analyse_resume_against_job)

# ---------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "jobmatch.db")
ALLOWED_EXT = {"pdf", "docx", "txt"}
MAX_MB = 5

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "jobmatch-dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin','seeker')),
        is_active INTEGER NOT NULL DEFAULT 1,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        last_login TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        text TEXT NOT NULL,
        skills TEXT NOT NULL,
        uploaded_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        company TEXT NOT NULL,
        location TEXT,
        description TEXT NOT NULL,
        required_skills TEXT,
        is_open INTEGER NOT NULL DEFAULT 1,
        created_by INTEGER,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        job_id INTEGER NOT NULL,
        resume_id INTEGER,
        match_score REAL,
        status TEXT DEFAULT 'Applied',
        applied_at TEXT NOT NULL,
        UNIQUE(user_id, job_id)
    );
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT NOT NULL,
        detail TEXT,
        created_at TEXT NOT NULL
    );
    """)

    # Only ONE admin, seeded via DB
    cur.execute("SELECT id FROM users WHERE role='admin' LIMIT 1")
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users(name,email,password_hash,role,is_active,created_at)"
            " VALUES(?,?,?,?,?,?)",
            ("Administrator", "admin@jobmatch.ai",
             generate_password_hash("admin123"), "admin", 1, now_iso())
        )

    # seed jobs
    cur.execute("SELECT COUNT(*) AS c FROM jobs")
    if cur.fetchone()["c"] == 0:
        seeds = [
            ("Data Scientist", "Acme Analytics", "Chennai",
             "We are hiring a Data Scientist. Required skills: Python, machine learning, "
             "statistics, pandas, numpy, scikit-learn, SQL, data visualization, TensorFlow, "
             "NLP. Model deployment and Docker are a plus.",
             "Python, Machine Learning, SQL, Pandas, NumPy, scikit-learn, TensorFlow, NLP, Docker"),
            ("Machine Learning Engineer", "Nova AI", "Remote",
             "Build and deploy ML models. Required: Python, PyTorch or TensorFlow, Docker, "
             "Kubernetes, REST APIs, MLOps, deep learning, NLP.",
             "Python, TensorFlow, PyTorch, Docker, Kubernetes, MLOps, Deep Learning, NLP, REST API"),
            ("Data Analyst", "BrightMetrics", "Bangalore",
             "Analyse business data and build dashboards. Required: SQL, Excel, Tableau, "
             "Power BI, Python, statistics, communication.",
             "SQL, Excel, Tableau, Power BI, Python, Statistics, Data Analysis"),
            ("Web Developer", "PixelWorks", "Coimbatore",
             "Full-stack web development. Required: HTML, CSS, JavaScript, React, Node.js, "
             "REST API, Git, MongoDB or SQL.",
             "HTML, CSS, JavaScript, React, Node.js, REST API, Git, MongoDB, SQL"),
            ("DevOps Engineer", "CloudForge", "Hyderabad",
             "Design CI/CD pipelines and manage cloud infrastructure. Required: Linux, "
             "Docker, Kubernetes, AWS, CI/CD, Git.",
             "Linux, Docker, Kubernetes, AWS, CI/CD, Git"),
        ]
        for j in seeds:
            cur.execute(
                "INSERT INTO jobs(title,company,location,description,required_skills,"
                "created_by,created_at) VALUES(?,?,?,?,?,?,?)",
                (j[0], j[1], j[2], j[3], j[4], None, now_iso())
            )

    conn.commit()
    conn.close()


def log_event(event: str, detail: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT INTO audit_log(event,detail,created_at) VALUES(?,?,?)",
        (event, detail, now_iso())
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------
def login_required(role=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in first.", "warning")
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                flash("Access denied.", "danger")
                return redirect(url_for("dashboard"))
            return fn(*args, **kwargs)
        return wrapper
    return deco


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not (name and email and password):
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users(name,email,password_hash,role,is_active,created_at)"
                " VALUES(?,?,?,?,?,?)",
                (name, email, generate_password_hash(password), "seeker", 1, now_iso())
            )
            conn.commit()
            log_event("REGISTER", email)
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

        if not user:
            conn.close()
            log_event("LOGIN_FAIL", f"unknown email: {email}")
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        if not user["is_active"]:
            conn.close()
            log_event("LOGIN_BLOCKED", f"inactive user: {email}")
            flash("Your account has been deactivated. Contact the administrator.", "danger")
            return render_template("login.html")

        if not check_password_hash(user["password_hash"], password):
            conn.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1 WHERE id=?",
                (user["id"],)
            )
            conn.commit()
            conn.close()
            log_event("LOGIN_FAIL", f"bad password: {email}")
            flash("Invalid email or password.", "danger")
            return render_template("login.html")

        conn.execute(
            "UPDATE users SET failed_attempts = 0, last_login=? WHERE id=?",
            (now_iso(), user["id"])
        )
        conn.commit()
        conn.close()

        session.clear()
        session["user_id"] = user["id"]
        session["name"] = user["name"]
        session["role"] = user["role"]
        log_event("LOGIN_OK", f"{user['role']}: {email}")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required()
def dashboard():
    if session["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("seeker_dashboard"))


# ---------------------------------------------------------------
# Job Seeker routes
# ---------------------------------------------------------------
@app.route("/seeker/dashboard")
@login_required("seeker")
def seeker_dashboard():
    conn = get_db()
    user_id = session["user_id"]

    resume = conn.execute(
        "SELECT * FROM resumes WHERE user_id=? ORDER BY uploaded_at DESC LIMIT 1",
        (user_id,)
    ).fetchone()

    jobs = conn.execute("SELECT * FROM jobs WHERE is_open=1 ORDER BY created_at DESC").fetchall()

    job_results = []
    if resume:
        resume_skills = set(resume["skills"].split(",")) if resume["skills"] else set()
        for job in jobs:
            analysis = analyse_resume_against_job(resume["text"], resume_skills, job)
            job_results.append({"job": dict(job), **analysis})
        job_results.sort(key=lambda x: x["match_score"], reverse=True)

    top_roles = predict_roles(
        set(resume["skills"].split(",")) if resume and resume["skills"] else set()
    ) if resume else []

    applied_ids = {r["job_id"] for r in conn.execute(
        "SELECT job_id FROM applications WHERE user_id=?", (user_id,)
    ).fetchall()}

    applications = conn.execute("""
        SELECT a.*, j.title, j.company
        FROM applications a JOIN jobs j ON a.job_id = j.id
        WHERE a.user_id=? ORDER BY a.applied_at DESC
    """, (user_id,)).fetchall()

    conn.close()
    return render_template(
        "seeker_dashboard.html",
        resume=resume,
        job_results=job_results,
        top_roles=top_roles,
        applied_ids=applied_ids,
        applications=applications,
    )


@app.route("/seeker/upload", methods=["POST"])
@login_required("seeker")
def upload_resume():
    f = request.files.get("resume")
    if not f or f.filename == "":
        flash("Please choose a resume file.", "warning")
        return redirect(url_for("seeker_dashboard"))
    if not allowed_file(f.filename):
        flash("Only PDF, DOCX or TXT files are allowed.", "danger")
        return redirect(url_for("seeker_dashboard"))

    ext = f.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{session['user_id']}_{int(datetime.now().timestamp())}_{secure_filename(f.filename)}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(save_path)

    try:
        text = extract_text(save_path, ext)
    except Exception as e:
        flash(f"Could not read file: {e}", "danger")
        return redirect(url_for("seeker_dashboard"))

    if not text or len(text.split()) < 20:
        flash("The uploaded resume appears to be empty or unreadable.", "warning")
        return redirect(url_for("seeker_dashboard"))

    skills = extract_skills(text)
    conn = get_db()
    conn.execute(
        "INSERT INTO resumes(user_id,filename,text,skills,uploaded_at) VALUES(?,?,?,?,?)",
        (session["user_id"], safe_name, text, ",".join(sorted(skills)), now_iso())
    )
    conn.commit()
    conn.close()
    flash(f"Resume analysed. {len(skills)} skills detected.", "success")
    return redirect(url_for("seeker_dashboard"))


@app.route("/seeker/apply/<int:job_id>", methods=["POST"])
@login_required("seeker")
def apply_job(job_id):
    conn = get_db()
    user_id = session["user_id"]

    resume = conn.execute(
        "SELECT * FROM resumes WHERE user_id=? ORDER BY uploaded_at DESC LIMIT 1",
        (user_id,)
    ).fetchone()
    if not resume:
        flash("Please upload a resume first.", "warning")
        conn.close()
        return redirect(url_for("seeker_dashboard"))

    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not job:
        flash("Job not found.", "danger")
        conn.close()
        return redirect(url_for("seeker_dashboard"))

    skills = set(resume["skills"].split(",")) if resume["skills"] else set()
    analysis = analyse_resume_against_job(resume["text"], skills, job)

    try:
        conn.execute(
            "INSERT INTO applications(user_id,job_id,resume_id,match_score,status,applied_at)"
            " VALUES(?,?,?,?,?,?)",
            (user_id, job_id, resume["id"], analysis["match_score"], "Applied", now_iso())
        )
        conn.commit()
        flash(f"Applied to '{job['title']}' with match {analysis['match_score']}%.", "success")
    except sqlite3.IntegrityError:
        flash("You have already applied to this job.", "warning")
    finally:
        conn.close()

    return redirect(url_for("seeker_dashboard"))


@app.route("/seeker/search")
@login_required("seeker")
def seeker_search():
    q = request.args.get("q", "").strip()
    min_score = float(request.args.get("min_score", 0) or 0)
    sort = request.args.get("sort", "match")

    conn = get_db()
    resume = conn.execute(
        "SELECT * FROM resumes WHERE user_id=? ORDER BY uploaded_at DESC LIMIT 1",
        (session["user_id"],)
    ).fetchone()
    jobs = conn.execute("SELECT * FROM jobs WHERE is_open=1").fetchall()

    results = []
    if resume:
        skills = set(resume["skills"].split(",")) if resume["skills"] else set()
        for job in jobs:
            analysis = analyse_resume_against_job(resume["text"], skills, job)
            haystack = (job["title"] + " " + job["company"] + " " +
                        (job["required_skills"] or "")).lower()
            if q and q.lower() not in haystack:
                continue
            if analysis["match_score"] < min_score:
                continue
            results.append({"job": dict(job), **analysis})

    if sort == "match":
        results.sort(key=lambda x: x["match_score"], reverse=True)
    elif sort == "newest":
        results.sort(key=lambda x: x["job"]["created_at"], reverse=True)
    elif sort == "title":
        results.sort(key=lambda x: x["job"]["title"])

    conn.close()
    return render_template("seeker_search.html", results=results, q=q,
                           min_score=min_score, sort=sort, resume=resume)


@app.route("/seeker/report/<int:job_id>")
@login_required("seeker")
def seeker_report(job_id):
    conn = get_db()
    resume = conn.execute(
        "SELECT * FROM resumes WHERE user_id=? ORDER BY uploaded_at DESC LIMIT 1",
        (session["user_id"],)
    ).fetchone()
    job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not (resume and job):
        conn.close()
        flash("Upload a resume and pick a valid job.", "warning")
        return redirect(url_for("seeker_dashboard"))

    skills = set(resume["skills"].split(",")) if resume["skills"] else set()
    analysis = analyse_resume_against_job(resume["text"], skills, job)
    conn.close()
    return render_template("report.html", user=session,
                           resume=resume, job=job, analysis=analysis)


@app.route("/seeker/stats")
@login_required("seeker")
def seeker_stats():
    conn = get_db()
    resume = conn.execute(
        "SELECT * FROM resumes WHERE user_id=? ORDER BY uploaded_at DESC LIMIT 1",
        (session["user_id"],)
    ).fetchone()
    buckets = {"High (75+)": 0, "Medium (50-74)": 0, "Low (<50)": 0}
    if resume:
        skills = set(resume["skills"].split(",")) if resume["skills"] else set()
        for job in conn.execute("SELECT * FROM jobs WHERE is_open=1").fetchall():
            s = analyse_resume_against_job(resume["text"], skills, job)["match_score"]
            if s >= 75:
                buckets["High (75+)"] += 1
            elif s >= 50:
                buckets["Medium (50-74)"] += 1
            else:
                buckets["Low (<50)"] += 1
    conn.close()
    return {"buckets": buckets}


# ---------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------
@app.route("/admin/dashboard")
@login_required("admin")
def admin_dashboard():
    conn = get_db()
    stats = {
        "users": conn.execute("SELECT COUNT(*) c FROM users WHERE role='seeker'").fetchone()["c"],
        "jobs": conn.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"],
        "applications": conn.execute("SELECT COUNT(*) c FROM applications").fetchone()["c"],
        "avg_match": conn.execute(
            "SELECT IFNULL(ROUND(AVG(match_score),1),0) m FROM applications"
        ).fetchone()["m"],
    }
    recent = conn.execute("""
        SELECT a.*, u.name AS user_name, u.email AS user_email,
               j.title AS job_title, j.company
        FROM applications a
        JOIN users u ON u.id = a.user_id
        JOIN jobs j ON j.id = a.job_id
        ORDER BY a.applied_at DESC LIMIT 8
    """).fetchall()
    conn.close()
    return render_template("admin_dashboard.html", stats=stats, recent=recent)


@app.route("/admin/jobs", methods=["GET", "POST"])
@login_required("admin")
def admin_jobs():
    conn = get_db()
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        company = request.form.get("company", "").strip()
        location = request.form.get("location", "").strip()
        description = request.form.get("description", "").strip()
        required_skills = request.form.get("required_skills", "").strip()
        if not (title and company and description):
            flash("Title, company and description are required.", "danger")
        else:
            conn.execute(
                "INSERT INTO jobs(title,company,location,description,required_skills,"
                "created_by,created_at) VALUES(?,?,?,?,?,?,?)",
                (title, company, location, description, required_skills,
                 session["user_id"], now_iso())
            )
            conn.commit()
            flash("Job posted successfully.", "success")
        return redirect(url_for("admin_jobs"))

    jobs = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template("admin_jobs.html", jobs=jobs)


@app.route("/admin/jobs/delete/<int:job_id>", methods=["POST"])
@login_required("admin")
def admin_delete_job(job_id):
    conn = get_db()
    conn.execute("DELETE FROM applications WHERE job_id=?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()
    flash("Job deleted.", "success")
    return redirect(url_for("admin_jobs"))


@app.route("/admin/users")
@login_required("admin")
def admin_users():
    conn = get_db()
    users = conn.execute("""
        SELECT u.*,
               (SELECT COUNT(*) FROM resumes r WHERE r.user_id=u.id) AS resume_count,
               (SELECT COUNT(*) FROM applications a WHERE a.user_id=u.id) AS app_count
        FROM users u
        WHERE u.role='seeker'
        ORDER BY u.created_at DESC
    """).fetchall()
    conn.close()
    return render_template("admin_users.html", users=users)


@app.route("/admin/users/<int:user_id>")
@login_required("admin")
def admin_user_detail(user_id):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE id=? AND role='seeker'", (user_id,)
    ).fetchone()
    if not user:
        conn.close()
        flash("User not found.", "danger")
        return redirect(url_for("admin_users"))

    resumes = conn.execute(
        "SELECT * FROM resumes WHERE user_id=? ORDER BY uploaded_at DESC", (user_id,)
    ).fetchall()

    apps = conn.execute("""
        SELECT a.*, j.title, j.company
        FROM applications a JOIN jobs j ON a.job_id = j.id
        WHERE a.user_id=? ORDER BY a.applied_at DESC
    """, (user_id,)).fetchall()

    conn.close()
    return render_template("admin_user_detail.html", user=user, resumes=resumes, apps=apps)


@app.route("/admin/users/<int:user_id>/toggle", methods=["POST"])
@login_required("admin")
def admin_toggle_user(user_id):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE id=? AND role='seeker'", (user_id,)
    ).fetchone()
    if user:
        new_state = 0 if user["is_active"] else 1
        conn.execute("UPDATE users SET is_active=? WHERE id=?", (new_state, user_id))
        conn.commit()
        flash(f"User {'activated' if new_state else 'deactivated'}.", "success")
    conn.close()
    return redirect(url_for("admin_users"))


@app.route("/admin/applications")
@login_required("admin")
def admin_applications():
    conn = get_db()
    rows = conn.execute("""
        SELECT a.*, u.name AS user_name, u.email AS user_email,
               j.title AS job_title, j.company
        FROM applications a
        JOIN users u ON u.id = a.user_id
        JOIN jobs j ON j.id = a.job_id
        ORDER BY a.applied_at DESC
    """).fetchall()
    conn.close()
    return render_template("admin_applications.html", applications=rows)


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)