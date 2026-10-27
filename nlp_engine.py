"""
nlp_engine.py
--------------
Text extraction, skill parsing, role prediction, resume–job matching,
ATS scoring and skill-gap detection.

Matching uses TF-IDF + cosine similarity (fast, no GPU).
To switch to Sentence-BERT, replace `semantic_similarity()` with:

    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def semantic_similarity(a, b):
        ea, eb = model.encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(ea, eb)[0][0])
"""

import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Skill dictionary: canonical skill -> list of aliases (lowercase)
# ---------------------------------------------------------------
SKILL_ALIASES = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "JavaScript": ["javascript", "js "],
    "TypeScript": ["typescript"],
    "HTML": ["html"],
    "CSS": ["css"],
    "React": ["react", "reactjs", "react.js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Django": ["django"],
    "Flask": ["flask"],
    "SQL": ["sql", "mysql", "postgres", "postgresql", "sqlite", "oracle"],
    "MongoDB": ["mongodb", "mongo db"],
    "Excel": ["excel", "ms excel"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "sci-kit learn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Keras": ["keras"],
    "OpenCV": ["opencv"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "Deep Learning": ["deep learning", "deep-learning"],
    "Machine Learning": ["machine learning", "ml ", " ml,"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization", "data visualisation"],
    "Statistics": ["statistics", "statistical"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Git": ["git", "github", "gitlab"],
    "Linux": ["linux", "ubuntu"],
    "REST API": ["rest api", "restful", "rest-api"],
    "MLOps": ["mlops", "ml ops"],
    "CI/CD": ["ci/cd", "cicd", "jenkins"],
    "Agile": ["agile", "scrum"],
    "Communication": ["communication"],
    "Leadership": ["leadership"],
    "Problem Solving": ["problem solving", "problem-solving"],
    "Teamwork": ["teamwork", "team work"],
}

# ---------------------------------------------------------------
# Job role → required skills (for rule-based role prediction)
# ---------------------------------------------------------------
JOB_ROLES = {
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "Pandas", "NumPy",
                       "scikit-learn", "SQL", "TensorFlow", "NLP"],
    "Machine Learning Engineer": ["Python", "TensorFlow", "PyTorch", "Deep Learning",
                                  "Docker", "Kubernetes", "REST API", "MLOps", "NLP"],
    "Data Analyst": ["SQL", "Excel", "Tableau", "Power BI", "Python", "Statistics",
                     "Data Analysis", "Data Visualization"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "REST API",
                      "Git", "SQL", "MongoDB"],
    "Backend Developer": ["Python", "Java", "Flask", "Django", "Node.js", "SQL",
                          "MongoDB", "REST API", "Docker", "Git"],
    "AI/ML Researcher": ["Python", "Deep Learning", "PyTorch", "TensorFlow", "NLP",
                         "Computer Vision", "Statistics"],
    "DevOps Engineer": ["Linux", "Docker", "Kubernetes", "AWS", "CI/CD", "Git", "GitHub"],
}

# ---------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------
def extract_text(file_path: str, ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == "docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")

# ---------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------
def extract_skills(text: str) -> set:
    lowered = " " + text.lower() + " "
    found = set()
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", lowered):
                found.add(canonical)
                break
    return found

# ---------------------------------------------------------------
# Role prediction (rule-based scoring, top-N)
# ---------------------------------------------------------------
def predict_roles(skills: set, top_n: int = 3):
    ranked = []
    for role, req in JOB_ROLES.items():
        req_set = set(req)
        overlap = len(skills & req_set)
        score = overlap / len(req_set) if req_set else 0
        ranked.append({"role": role, "score": round(score * 100, 1)})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]

# ---------------------------------------------------------------
# Semantic similarity (TF-IDF)
# ---------------------------------------------------------------
def semantic_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        m = vec.fit_transform([a, b])
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except ValueError:
        return 0.0

# ---------------------------------------------------------------
# ATS readiness score (heuristic / 100)
# ---------------------------------------------------------------
def ats_score(resume_text: str, resume_skills: set, jd_skills: set) -> dict:
    text = resume_text.lower()
    checks = {}

    # 1. Contact info
    checks["Email present"] = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text))
    checks["Phone present"] = bool(re.search(r"(\+?\d[\d\s\-]{8,}\d)", resume_text))

    # 2. Section headings
    for sec in ["skills", "education", "experience", "project"]:
        checks[f"'{sec.title()}' section"] = sec in text

    # 3. Word count
    wc = len(resume_text.split())
    checks["Length 150–900 words"] = 150 <= wc <= 900

    # 4. Keyword coverage
    if jd_skills:
        coverage = len(resume_skills & jd_skills) / len(jd_skills)
        checks["Skill keyword coverage ≥ 50%"] = coverage >= 0.5

    # 5. No table / image spam indicators
    checks["No excessive symbols/tables"] = resume_text.count("|") < 5

    passed = sum(checks.values())
    score = int(round(passed / len(checks) * 100))
    return {"score": score, "checks": checks}

# ---------------------------------------------------------------
# Full analysis for one job
# ---------------------------------------------------------------
def analyse_resume_against_job(resume_text: str, resume_skills: set, job_row) -> dict:
    jd_text = " ".join(filter(None, [
        job_row["title"], job_row["description"], job_row["required_skills"] or ""
    ]))
    jd_skills = extract_skills(jd_text)

    text_sim = semantic_similarity(resume_text, jd_text)
    skill_overlap = (len(resume_skills & jd_skills) / len(jd_skills)) if jd_skills else 0

    # weighted final score
    final = (0.55 * text_sim + 0.45 * skill_overlap) * 100
    final = round(min(final, 99.9), 1)

    if final >= 75:
        fit = "HIGH"
    elif final >= 50:
        fit = "MEDIUM"
    else:
        fit = "LOW"

    missing = sorted(jd_skills - resume_skills)
    ats = ats_score(resume_text, resume_skills, jd_skills)

    return {
        "match_score": final,
        "text_similarity": round(text_sim * 100, 1),
        "skill_overlap": round(skill_overlap * 100, 1),
        "fit_level": fit,
        "missing_skills": missing,
        "ats": ats,
    }