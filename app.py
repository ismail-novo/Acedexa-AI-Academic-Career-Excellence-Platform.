import json
import os
import sys
import time
from datetime import datetime
import customtkinter as ctk

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenCV 4 + 5 compatible
HAS_CV = False
cv2 = None
_CV_IMPORT_ERROR = ""
try:
    import cv2 as _cv2
    cv2 = _cv2
    if hasattr(cv2, "VideoCapture"):
        HAS_CV = True
except Exception as _e:
    _CV_IMPORT_ERROR = str(_e)

from PIL import Image, ImageTk, ImageDraw

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def ensure_cv2():
    global HAS_CV, cv2, _CV_IMPORT_ERROR
    if HAS_CV and cv2 is not None:
        return True
    try:
        import cv2 as _cv2
        cv2 = _cv2
        if hasattr(cv2, "VideoCapture"):
            HAS_CV = True
            return True
    except Exception as e:
        _CV_IMPORT_ERROR = str(e)
    HAS_CV = False
    return False


def make_circular_image(image_path_or_pil, size=(75, 75)):
    try:
        if isinstance(image_path_or_pil, str) and os.path.exists(image_path_or_pil):
            img = Image.open(image_path_or_pil).convert("RGBA")
        elif hasattr(image_path_or_pil, "convert"):
            img = image_path_or_pil.convert("RGBA")
        else:
            img = Image.new("RGBA", size, (137, 180, 250, 255))
        img = img.resize(size, Image.Resampling.LANCZOS)
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size[0], size[1]), fill=255)
        output = Image.new("RGBA", size, (0, 0, 0, 0))
        output.paste(img, (0, 0), mask=mask)
        return ctk.CTkImage(light_image=output, dark_image=output, size=size)
    except Exception:
        fallback = Image.new("RGBA", size, (137, 180, 250, 255))
        return ctk.CTkImage(light_image=fallback, dark_image=fallback, size=size)


# ==============================================================================
# DATA ENGINE
# ==============================================================================
class JSONDataManager:
    def __init__(self, db_path="acadexa_database.json"):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            self.save_all({"last_active_user": "", "accounts": [], "user_data": {}})

    def load_all(self):
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"last_active_user": "", "accounts": [], "user_data": {}}

    def save_all(self, data):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def set_last_active_user(self, student_id):
        all_db = self.load_all()
        all_db["last_active_user"] = student_id
        self.save_all(all_db)

    def get_last_active_user_pic(self):
        all_db = self.load_all()
        accounts = all_db.get("accounts", [])
        last_id = all_db.get("last_active_user", "")
        if last_id:
            for a in accounts:
                if a.get("student_id") == last_id and a.get("profile_pic"):
                    return a.get("profile_pic")
        for a in reversed(accounts):
            if a.get("profile_pic"):
                return a.get("profile_pic")
        return ""

    def get_user_profile(self, user_key):
        data = self.load_all()
        u = data.get("user_data", {}).get(user_key, {})
        if not u:
            return {
                "student_profile": {
                    "name": user_key, "student_id": user_key, "university": "",
                    "faculty": "", "department": "", "degree": "", "major": "",
                    "minor": "", "semester_year": "", "batch": "",
                    "graduation_year": "", "location": "", "academic_level": "",
                    "gpa": "0.0", "career_goal": "", "profile_pic": ""
                },
                "courses": [], "skills": [], "projects": [],
                "study_sessions": [], "topic_mastery": {}, "weak_topics": [],
                "daily_schedule": {}, "study_routine": [], "quiz_history": [],
                "tasks": [], "study_materials": [], "brain_scores": [], "cognitive_intelligence": {},
                "resume_history": [], "resume_draft": {},
            }
        u.setdefault("daily_schedule", {})
        u.setdefault("study_routine", [])
        u.setdefault("courses", [])
        u.setdefault("study_sessions", [])
        u.setdefault("topic_mastery", {})
        u.setdefault("weak_topics", [])
        u.setdefault("quiz_history", [])
        u.setdefault("skills", [])
        u.setdefault("projects", [])
        u.setdefault("tasks", [])
        u.setdefault("study_materials", [])
        u.setdefault("brain_scores", [])
        u.setdefault("cognitive_intelligence", {})
        u.setdefault("resume_history", [])
        u.setdefault("resume_draft", {})
        return u

    def save_user_profile(self, user_key, profile_data):
        data = self.load_all()
        data.setdefault("user_data", {})
        data["user_data"][user_key] = profile_data
        pic = profile_data.get("student_profile", {}).get("profile_pic", "")
        for acc in data.get("accounts", []):
            if acc.get("student_id") == user_key:
                acc["profile_pic"] = pic
        self.save_all(data)

    def toggle_account_active(self, student_id):
        """Toggle Active <-> Deactivated and return new status."""
        all_db = self.load_all()
        new_status = "Active"
        for acc in all_db.get("accounts", []):
            if acc.get("student_id") == student_id:
                current = acc.get("status", "Active")
                new_status = "Deactivated" if current == "Active" else "Active"
                acc["status"] = new_status
                break
        self.save_all(all_db)
        return new_status

    def delete_account(self, student_id):
        all_db = self.load_all()
        all_db["accounts"] = [a for a in all_db.get("accounts", []) if a.get("student_id") != student_id]
        if student_id in all_db.get("user_data", {}):
            del all_db["user_data"][student_id]
        if all_db.get("last_active_user") == student_id:
            all_db["last_active_user"] = ""
        self.save_all(all_db)

    def get_account(self, student_id):
        for a in self.load_all().get("accounts", []):
            if a.get("student_id") == student_id:
                return a
        return None


db = JSONDataManager()


# ==============================================================================
# INTELLIGENCE ENGINES
# ==============================================================================
class MasteryEngine:
    @staticmethod
    def calculate_mastery(accuracy, avg_response_time_sec=30, attempts=1):
        if attempts == 0:
            return 0.0, "Unassessed"
        score = accuracy * 0.7
        speed_factor = max(0, min(30, (60 - avg_response_time_sec) * 0.5))
        score += speed_factor
        score = round(min(100.0, max(0.0, score)), 1)
        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Strong"
        elif score >= 60:
            status = "Developing"
        elif score >= 40:
            status = "Weak"
        else:
            status = "Critical"
        return score, status


class StudyIntelligenceAgent:
    def evaluate(self, sessions):
        if not sessions:
            return {"total_hours": 0.0, "avg_distraction": 0.0, "status": "No Active Study Data",
                    "consistency_score": 0.0, "avg_concentration": 0.0, "unique_days": 0,
                    "subject_balance": 0.0}
        total_mins = sum(s.get("duration_minutes", 0) for s in sessions)
        avg_dist = float(np.mean([s.get("distraction_score", 0) for s in sessions]))
        avg_comp = float(np.mean([s.get("completion_rate", 0) for s in sessions]))
        avg_conc = float(np.mean([s.get("concentration_percent", 0) for s in sessions]))
        # unique calendar days with at least one session
        days = set()
        for s in sessions:
            d = str(s.get("date", ""))[:10]
            if d:
                days.add(d)
        unique_days = len(days)
        consistency_score = float(min(100.0, unique_days * 14.0 + min(20.0, len(sessions) * 2)))
        # subject balance: entropy-like — more even subject mix → higher
        subj_mins = {}
        for s in sessions:
            sub = (s.get("subject") or "General")[:40]
            subj_mins[sub] = subj_mins.get(sub, 0) + float(s.get("duration_minutes", 0) or 0)
        total = sum(subj_mins.values()) or 1.0
        probs = [v / total for v in subj_mins.values()]
        if len(probs) <= 1:
            subject_balance = 40.0 if total > 0 else 0.0
        else:
            # normalized entropy 0..1 → 0..100
            ent = -sum(p * np.log(p + 1e-12) for p in probs)
            max_ent = np.log(len(probs))
            subject_balance = float(min(100.0, 100.0 * (ent / max_ent))) if max_ent > 0 else 0.0
        status = "Optimal Focus" if avg_comp >= 75 and avg_dist <= 4 and avg_conc >= 80 else "Needs Focus"
        return {
            "total_hours": round(total_mins / 60, 1),
            "avg_distraction": round(avg_dist, 1),
            "status": status,
            "consistency_score": consistency_score,
            "avg_concentration": round(avg_conc, 1),
            "unique_days": unique_days,
            "subject_balance": round(subject_balance, 1),
        }


class UniversalJobMatcherAgent:
    def match(self, resume_text, job_desc_text):
        if not resume_text.strip() or not job_desc_text.strip():
            return 0.0, [], []
        docs = [resume_text, job_desc_text]
        vec = TfidfVectorizer(stop_words="english")
        m = vec.fit_transform(docs)
        score = round(float(cosine_similarity(m[0:1], m[1:2])[0][0]) * 100, 2)
        names = vec.get_feature_names_out()
        rv, jv = m[0].toarray()[0], m[1].toarray()[0]
        matched, missing = [], []
        for i, w in enumerate(names):
            if jv[i] > 0 and rv[i] == 0:
                missing.append(w)
            elif jv[i] > 0 and rv[i] > 0:
                matched.append(w)
        return score, matched[:12], missing[:12]


class AcadexaDecisionAgent:
    def evaluate_student(self, user_profile):
        profile = user_profile.get("student_profile", {})
        courses = user_profile.get("courses", [])
        sessions = user_profile.get("study_sessions", [])
        weak = user_profile.get("weak_topics", [])
        skills = user_profile.get("skills", [])
        projects = user_profile.get("projects", [])
        mastery_map = user_profile.get("topic_mastery", {})
        routine = user_profile.get("study_routine", [])

        tasks = user_profile.get("tasks", [])
        if tasks:
            done = sum(1 for t in tasks if t.get("done"))
            task_completion = round(100.0 * done / max(1, len(tasks)), 1)
        else:
            task_completion = 0.0

        study_eval = study_intelligence.evaluate(sessions)
        learning_intel = min(100.0, study_eval["consistency_score"])

        if mastery_map:
            knowledge = round(float(np.mean(list(mastery_map.values()))), 1)
        else:
            knowledge = 0.0

        skill_strength = min(100.0, len(skills) * 15.0)
        job_readiness = min(100.0, (len(projects) * 20) + (len(skills) * 5) + (knowledge * 0.3))
        if not (projects or skills or mastery_map):
            job_readiness = 0.0

        routine_eff = 0.0
        if routine:
            study_blocks = [r for r in routine if r.get("type") == "study"]
            routine_eff = min(100.0, len(study_blocks) * 8)

        # Quiz activity 0–100 (capped)
        quizzes = user_profile.get("quiz_history", [])
        quiz_activity = float(min(100.0, len(quizzes) * 12.0))

        avg_conc = float(study_eval.get("avg_concentration", 0) or 0)
        subject_balance = float(study_eval.get("subject_balance", 0) or 0)

        # Brain training score from recent games (0–100)
        brain_list = user_profile.get("brain_scores", []) or []
        if brain_list:
            recent = brain_list[-10:]
            brain_score = float(np.mean([float(b.get("score", 0)) for b in recent]))
        else:
            brain_score = 0.0

        # Habit Health Index (weighted composite 0–100)
        # consistency 20% + concentration 20% + tasks 15% + quiz 15% + subject balance 15% + brain 15%
        habit_health = round(
            0.20 * learning_intel
            + 0.20 * avg_conc
            + 0.15 * task_completion
            + 0.15 * quiz_activity
            + 0.15 * subject_balance
            + 0.15 * brain_score,
            1,
        )
        habit_breakdown = {
            "consistency": round(learning_intel, 1),
            "avg_concentration": round(avg_conc, 1),
            "task_completion": task_completion,
            "quiz_activity": round(quiz_activity, 1),
            "subject_balance": round(subject_balance, 1),
            "brain_training": round(brain_score, 1),
        }

        recs = []
        if not courses:
            recs.append("Add your subjects in Courses & Subjects.")
        if weak:
            recs.append(f"Focus on weak topics: {', '.join(weak[:3])}")
        if study_eval["avg_distraction"] > 5:
            recs.append("High distraction. Use shorter sessions.")
        if avg_conc < 80 and sessions:
            recs.append("Concentration often < 80%. Reduce table leaves.")
        if not routine:
            recs.append("Generate a Smart Study Routine for better planning.")
        if knowledge == 0:
            recs.append("Take quizzes to build Knowledge Mastery.")
        if tasks and task_completion < 50:
            recs.append("Task completion under 50%. Finish pending tasks in Task Manager.")
        if habit_health < 40 and sessions:
            recs.append("Habit Health is low — aim for shorter daily sessions (streak > intensity).")
        if subject_balance < 30 and sessions:
            recs.append("Subject balance is skewed — rotate neglected courses.")
        if not recs:
            recs.append("Habit profile looks balanced. Keep the streak going!")

        cog = user_profile.get("cognitive_intelligence") or {}
        cog_avg = round(float(np.mean(list(cog.values()))), 1) if cog else 0.0

        return {
            "academic_health": task_completion,  # legacy key maps to task completion
            "task_completion": task_completion,
            "habit_health": habit_health,
            "habit_breakdown": habit_breakdown,
            "learning_intelligence": learning_intel,
            "knowledge_mastery": knowledge,
            "skill_strength": skill_strength,
            "job_readiness": round(job_readiness, 1),
            "routine_efficiency": round(routine_eff, 1),
            "cognitive_avg": cog_avg,
            "cognitive_intelligence": cog,
            "recommendations": recs
        }


class SmartRoutineEngine:
    @staticmethod
    def parse_time(t_str):
        try:
            raw = str(t_str).strip().lower().replace(".", ":")
            am, pm = "am" in raw, "pm" in raw
            raw = raw.replace("am", "").replace("pm", "").strip()
            parts = raw.split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            if pm and h < 12:
                h += 12
            if am and h == 12:
                h = 0
            if not (0 <= h <= 23 and 0 <= m <= 59):
                return None
            return h * 60 + m
        except Exception:
            return None

    @staticmethod
    def mins_to_str(mins):
        mins = int(mins) % (24 * 60)
        return f"{mins // 60:02d}:{mins % 60:02d}"

    @classmethod
    def generate(cls, schedule, courses, session_pref_mins=45):
        wake = cls.parse_time(schedule.get("wake_up", "06:30")) or 390
        sleep = cls.parse_time(schedule.get("sleep_time", "23:00")) or 1380
        if sleep <= wake:
            sleep += 24 * 60
        class_days = set(d.lower() for d in schedule.get("class_days", []))
        cs = cls.parse_time(schedule.get("class_start", "09:00")) or 540
        ce = cls.parse_time(schedule.get("class_end", "16:00")) or 960
        if ce <= cs and ce < 12 * 60:
            ce += 12 * 60
        try:
            session_pref_mins = max(20, int(session_pref_mins))
        except Exception:
            session_pref_mins = 45
        fixed = []
        for key, default, label in [("breakfast", "07:30", "Breakfast"),
                                     ("lunch", "13:00", "Lunch"),
                                     ("dinner", "20:00", "Dinner")]:
            t = cls.parse_time(schedule.get(key, default))
            if t is not None:
                try:
                    dur = max(10, int(schedule.get("meal_mins", 30) or 30))
                except Exception:
                    dur = 30
                fixed.append((t, t + dur, label))
        st = cls.parse_time(schedule.get("shower_time", "07:00"))
        if st is not None:
            try:
                dur = max(5, int(schedule.get("shower_mins", 20) or 20))
            except Exception:
                dur = 20
            fixed.append((st, st + dur, "Shower / Fresh-up"))
        try:
            wash = max(0, int(schedule.get("washroom_mins", 10) or 10))
        except Exception:
            wash = 10
        subjects = []
        for c in courses:
            title = c.get("title", "Subject")
            code = c.get("code", "")
            name = f"{code} - {title}" if code else title
            topics = c.get("topics", [])
            if topics:
                for tp in topics[:10]:
                    subjects.append(f"{name} | {tp}")
            else:
                subjects.append(name)
        if not subjects:
            subjects = ["General Study"]
        rows, subj_idx = [], 0
        days = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for day in days:
            blocks = list(fixed)
            if day.lower() in class_days:
                blocks.append((cs, ce, "University / College Classes"))
            blocks = sorted([(s, e, l) for s, e, l in blocks if e > s], key=lambda x: x[0])
            cursor, free = wake, []
            for s, e, _ in blocks:
                if e <= wake or s >= sleep:
                    continue
                sc, ec = max(s, wake), min(e, sleep)
                if sc > cursor:
                    free.append((cursor, sc))
                cursor = max(cursor, ec)
            if sleep > cursor:
                free.append((cursor, sleep))
            day_rows = []
            for fs, fe in free:
                avail, safety = fe - fs, 0
                while avail >= session_pref_mins + wash and subj_idx < len(subjects) * 4 and safety < 20:
                    safety += 1
                    ss = fs
                    se = min(fs + session_pref_mins, fe - wash)
                    if se - ss < 20:
                        break
                    subj = subjects[subj_idx % len(subjects)]
                    day_rows.append({
                        "day": day, "start": cls.mins_to_str(ss), "end": cls.mins_to_str(se),
                        "activity": f"Study: {subj}", "duration_mins": se - ss,
                        "type": "study", "subject": subj
                    })
                    subj_idx += 1
                    fs = se + wash
                    avail = fe - fs
            for s, e, label in blocks:
                day_rows.append({
                    "day": day, "start": cls.mins_to_str(s), "end": cls.mins_to_str(e),
                    "activity": label, "duration_mins": e - s, "type": "fixed", "subject": ""
                })
            day_rows.sort(key=lambda x: cls.parse_time(x["start"]) or 0)
            rows.extend(day_rows)
        return rows


study_intelligence = StudyIntelligenceAgent()
job_matcher = UniversalJobMatcherAgent()
decision_agent = AcadexaDecisionAgent()
routine_engine = SmartRoutineEngine()


# ==============================================================================
# CHESS AI: Minimax + Alpha-Beta + Opening Book + Neural-style Eval
# ==============================================================================
class ChessAIEngine:
    """
    Classical chess AI pipeline used by Brain Testing:

    1) OPENING BOOK  — theory moves for the first few plies (no search)
    2) MINIMAX       — MAX maximizes score, MIN minimizes (perfect info)
    3) ALPHA-BETA    — same result as minimax, prunes branches when α≥β
    4) NN-STYLE EVAL — linear "1-layer net": features · weights (+ bias)
                       (inspired by early NN chess evals / Maia-lite ideas;
                        full AlphaZero needs deep nets + MCTS + self-play)

    Difficulty 1–10 maps to search depth, noise, and book usage.
    """

    # Compact opening book: FEN key (no counters) → list of (uci, weight)
    # Covers common 1.e4 / 1.d4 lines so CPU does not "invent" theory.
    OPENING_BOOK = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq": [
            ("e2e4", 40), ("d2d4", 35), ("c2c4", 12), ("g1f3", 13),
        ],
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq": [
            ("e7e5", 35), ("c7c5", 30), ("e7e6", 15), ("c7c6", 12), ("g8f6", 8),
        ],
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": [
            ("g1f3", 40), ("f1c4", 20), ("f1b5", 20), ("d2d4", 15), ("b1c3", 5),
        ],
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq": [
            ("b8c6", 45), ("g8f6", 30), ("d7d6", 15), ("f7f5", 5),
        ],
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq": [
            ("f1b5", 35), ("f1c4", 30), ("d2d4", 20), ("b1c3", 15),
        ],
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": [
            ("g1f3", 40), ("b1c3", 20), ("c2c3", 15), ("d2d4", 15),
        ],
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq": [
            ("g8f6", 35), ("d7d5", 35), ("e7e6", 15), ("c7c5", 10),
        ],
        "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq": [
            ("c2c4", 40), ("g1f3", 30), ("b1c3", 15), ("c1g5", 10),
        ],
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq": [
            ("c2c4", 40), ("g1f3", 30), ("b1c3", 15), ("c1f4", 10),
        ],
    }

    # Neural-style weights (hand-tuned linear layer on board features)
    NN_WEIGHTS = {
        "material": 1.0,
        "pst": 0.85,
        "mobility": 1.4,
        "center_control": 2.2,
        "king_safety": 3.0,
        "bias": 0.0,
    }

    @staticmethod
    def fen_key(board):
        """FEN without halfmove/fullmove counters — stable book key."""
        return " ".join(board.fen().split(" ")[:3])

    @classmethod
    def book_move(cls, board, rng):
        """Return a theory move from the opening book, or None."""
        key = cls.fen_key(board)
        entries = cls.OPENING_BOOK.get(key)
        if not entries:
            return None
        legal = {m.uci(): m for m in board.legal_moves}
        pool = [(uci, w) for uci, w in entries if uci in legal]
        if not pool:
            return None
        total = sum(w for _, w in pool)
        r = rng.uniform(0, total)
        acc = 0
        for uci, w in pool:
            acc += w
            if r <= acc:
                return legal[uci]
        return legal[pool[-1][0]]

    @classmethod
    def nn_style_eval(cls, board, chess_mod):
        """
        Linear 'neural' evaluation: score = Σ w_i * feature_i + bias
        Features approximate what a tiny NN chess head might learn.
        """
        if board.is_checkmate():
            return -50000 if board.turn == chess_mod.WHITE else 50000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        MAT = {chess_mod.PAWN: 100, chess_mod.KNIGHT: 320, chess_mod.BISHOP: 330,
               chess_mod.ROOK: 500, chess_mod.QUEEN: 900, chess_mod.KING: 0}
        # center squares
        CENTER = {chess_mod.D4, chess_mod.E4, chess_mod.D5, chess_mod.E5,
                  chess_mod.C3, chess_mod.F3, chess_mod.C6, chess_mod.F6}

        material = 0
        pst = 0
        center = 0
        for pt, val in MAT.items():
            for sq in board.pieces(pt, chess_mod.WHITE):
                material += val
                # simple center bonus as positional prior
                if sq in CENTER:
                    center += 8
                # rank advancement for pawns
                if pt == chess_mod.PAWN:
                    pst += 6 * (chess_mod.square_rank(sq) - 1)
            for sq in board.pieces(pt, chess_mod.BLACK):
                material -= val
                if sq in CENTER:
                    center -= 8
                if pt == chess_mod.PAWN:
                    pst -= 6 * (6 - chess_mod.square_rank(sq))

        # mobility (side to move perspective adjusted later)
        mob = board.legal_moves.count()
        # flip board turn to estimate opponent mobility cheaply
        board.push(chess_mod.Move.null()) if False else None
        # approximate: mobility bonus for white when white to move
        mobility = mob if board.turn == chess_mod.WHITE else -mob

        # king safety: fewer attackers near king ≈ safer
        wk = board.king(chess_mod.WHITE)
        bk = board.king(chess_mod.BLACK)
        king_safety = 0
        if wk is not None:
            king_safety -= 4 * len(board.attackers(chess_mod.BLACK, wk))
        if bk is not None:
            king_safety += 4 * len(board.attackers(chess_mod.WHITE, bk))

        w = cls.NN_WEIGHTS
        score = (
            w["material"] * material
            + w["pst"] * pst
            + w["mobility"] * mobility
            + w["center_control"] * center
            + w["king_safety"] * king_safety
            + w["bias"]
        )
        return score

    @classmethod
    def ordered_moves(cls, board):
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: (0 if board.is_capture(m) else 1,
                                  0 if board.gives_check(m) else 1))
        return moves

    @classmethod
    def alphabeta(cls, board, depth, alpha, beta, maximizing, chess_mod):
        """
        Alpha-Beta pruning over Minimax:
          α = best already guaranteed for MAX
          β = best already guaranteed for MIN
          prune when α ≥ β (branch cannot affect root decision)
        """
        if depth == 0 or board.is_game_over():
            return cls.nn_style_eval(board, chess_mod), None

        best_move = None
        moves = cls.ordered_moves(board)

        if maximizing:
            value = -10 ** 9
            for m in moves:
                board.push(m)
                sc, _ = cls.alphabeta(board, depth - 1, alpha, beta, False, chess_mod)
                board.pop()
                if sc > value:
                    value, best_move = sc, m
                alpha = max(alpha, value)
                if alpha >= beta:  # β cutoff
                    break
            return value, best_move
        else:
            value = 10 ** 9
            for m in moves:
                board.push(m)
                sc, _ = cls.alphabeta(board, depth - 1, alpha, beta, True, chess_mod)
                board.pop()
                if sc < value:
                    value, best_move = sc, m
                beta = min(beta, value)
                if alpha >= beta:  # α cutoff
                    break
            return value, best_move

    @classmethod
    def difficulty_params(cls, level):
        # depth / noise / top_n / use_book
        depth = 1 if level <= 2 else (2 if level <= 5 else (3 if level <= 8 else 4))
        noise = max(0.0, 0.55 - level * 0.05)
        top_n = max(1, 8 - level // 2)
        use_book = level >= 2  # level 1 plays without theory
        return depth, noise, top_n, use_book

    @classmethod
    def choose_move(cls, board, level, chess_mod, rng):
        """Full pipeline: book → else alpha-beta root search with noise."""
        depth, noise, top_n, use_book = cls.difficulty_params(level)

        if use_book and board.fullmove_number <= 10:
            bm = cls.book_move(board, rng)
            if bm is not None:
                return bm, "book"

        # Root search for side to move (CPU is Black in our UI)
        scored = []
        maximizing = board.turn == chess_mod.WHITE
        for m in cls.ordered_moves(board):
            board.push(m)
            sc, _ = cls.alphabeta(
                board, depth - 1, -10 ** 9, 10 ** 9, not maximizing, chess_mod
            )
            board.pop()
            scored.append((sc, m))

        if not scored:
            legal = list(board.legal_moves)
            return (legal[0] if legal else None), "search"

        # Black prefers lower eval; White prefers higher
        scored.sort(key=lambda x: x[0], reverse=maximizing)
        pool = scored[:top_n]
        if noise > 0.05 and len(pool) > 1 and rng.random() < noise:
            return rng.choice(pool)[1], "search-noise"
        return pool[0][1], "search"


# ==============================================================================
# AUTH WINDOW (with proper Activate / Deactivate)
# ==============================================================================
class AuthWindow(ctk.CTkFrame):
    def __init__(self, parent, on_login_success):
        super().__init__(parent, fg_color="#11111B")
        self.on_login_success = on_login_success
        self.pack(fill="both", expand=True)
        self.signup_pic_path = ""

        center = ctk.CTkFrame(self, fg_color="#181825", corner_radius=16, width=520, height=680,
                              border_width=1, border_color="#313244")
        center.place(relx=0.5, rely=0.5, anchor="center")
        center.pack_propagate(False)

        ctk.CTkLabel(center, text="ACADEXA AI", font=ctk.CTkFont(size=24, weight="bold"),
                     text_color="#89B4FA").pack(pady=(18, 2))
        ctk.CTkLabel(center, text="Universal Academic & Career Intelligence Ecosystem",
                     font=ctk.CTkFont(size=11), text_color="#A6ADC8").pack(pady=(0, 8))

        self.auth_tabs = ctk.CTkTabview(
            center, fg_color="transparent",
            segmented_button_fg_color="#11111B",
            segmented_button_selected_color="#89B4FA"
        )
        self.auth_tabs.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.tab_login = self.auth_tabs.add("Login")
        self.tab_signup = self.auth_tabs.add("Sign Up")
        self.tab_manage = self.auth_tabs.add("Manage Accounts")

        self._build_login_tab()
        self._build_signup_tab()
        self._build_manage_tab()

    def _build_login_tab(self):
        scroll = ctk.CTkScrollableFrame(self.tab_login, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        pic = db.get_last_active_user_pic()
        self.circle_avatar = ctk.CTkLabel(scroll, text="", image=make_circular_image(pic, size=(70, 70)))
        self.circle_avatar.pack(pady=(6, 8))

        ctk.CTkLabel(scroll, text="Full Name", text_color="#CDD6F4").pack(anchor="w", pady=(4, 1))
        self.login_name = ctk.CTkEntry(scroll, placeholder_text="Enter Full Name",
                                       fg_color="#11111B", border_color="#313244", corner_radius=8)
        self.login_name.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(scroll, text="Uni ID (Student ID)", text_color="#CDD6F4").pack(anchor="w", pady=(4, 1))
        self.login_id = ctk.CTkEntry(scroll, placeholder_text="Enter Student ID",
                                     fg_color="#11111B", border_color="#313244", corner_radius=8)
        self.login_id.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(scroll, text="Password", text_color="#CDD6F4").pack(anchor="w", pady=(4, 1))
        self.login_pass = ctk.CTkEntry(scroll, placeholder_text="••••••••", show="•",
                                       fg_color="#11111B", border_color="#313244", corner_radius=8)
        self.login_pass.pack(fill="x", pady=(0, 10))

        self.login_msg = ctk.CTkLabel(scroll, text="", font=ctk.CTkFont(size=12))
        self.login_msg.pack(pady=(0, 4))

        ctk.CTkButton(scroll, text="Sign In", fg_color="#89B4FA", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), corner_radius=8,
                      command=self._handle_login).pack(fill="x", pady=(4, 10))

        switch = ctk.CTkFrame(scroll, fg_color="transparent")
        switch.pack(pady=4)
        ctk.CTkLabel(switch, text="New member?", text_color="#A6ADC8").pack(side="left", padx=2)
        ctk.CTkButton(switch, text="Sign Up Now", fg_color="transparent", text_color="#89B4FA",
                      hover=False, width=90,
                      command=lambda: self.auth_tabs.set("Sign Up")).pack(side="left")

    def _build_signup_tab(self):
        scroll = ctk.CTkScrollableFrame(self.tab_signup, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        self.signup_entries = {}
        fields = [
            ("Name *", "name"), ("Student ID *", "student_id"),
            ("University/College", "university"), ("Faculty/School", "faculty"),
            ("Department", "department"), ("Program/Degree", "degree"),
            ("Major", "major"), ("Minor", "minor"),
            ("Semester/Year", "semester_year"), ("Batch", "batch"),
            ("Graduation Year", "graduation_year"), ("Location", "location"),
            ("Academic Level", "academic_level"), ("Password *", "password")
        ]
        for label, key in fields:
            ctk.CTkLabel(scroll, text=label, text_color="#CDD6F4").pack(anchor="w", pady=(3, 1))
            show = "•" if key == "password" else ""
            ent = ctk.CTkEntry(scroll, placeholder_text=f"Enter {label.replace('*', '').strip()}",
                               show=show, fg_color="#11111B", border_color="#313244", corner_radius=8)
            ent.pack(fill="x", pady=(0, 4))
            self.signup_entries[key] = ent

        ctk.CTkLabel(scroll, text="Profile Picture (optional)", text_color="#CDD6F4").pack(anchor="w", pady=(6, 2))
        self.btn_pic = ctk.CTkButton(scroll, text="Choose Image File...", fg_color="#313244",
                                     text_color="#CDD6F4", corner_radius=8, command=self._select_pic)
        self.btn_pic.pack(fill="x", pady=(0, 8))

        self.signup_msg = ctk.CTkLabel(scroll, text="", font=ctk.CTkFont(size=12))
        self.signup_msg.pack(pady=(0, 4))

        ctk.CTkButton(scroll, text="Create Ecosystem Account", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), corner_radius=8,
                      command=self._handle_signup).pack(fill="x", pady=8)

    def _build_manage_tab(self):
        """Rebuild manage tab every time so Activate/Deactivate stays in sync."""
        for w in self.tab_manage.winfo_children():
            w.destroy()

        scroll = ctk.CTkScrollableFrame(self.tab_manage, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll, text="Existing Ecosystem Accounts",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color="#89B4FA").pack(anchor="w", pady=(10, 10))

        accounts = db.load_all().get("accounts", [])
        if not accounts:
            ctk.CTkLabel(scroll, text="No accounts found.", text_color="#A6ADC8").pack(anchor="w")
            return

        for acc in accounts:
            status = acc.get("status", "Active")
            is_active = status == "Active"

            card = ctk.CTkFrame(scroll, fg_color="#11111B", corner_radius=8,
                                border_width=1, border_color="#313244")
            card.pack(fill="x", pady=5, padx=2)

            # Left info
            info_col = ctk.CTkFrame(card, fg_color="transparent")
            info_col.pack(side="left", fill="x", expand=True, padx=10, pady=8)
            name_color = "#CDD6F4" if is_active else "#A6ADC8"
            status_color = "#A6E3A1" if is_active else "#F38BA8"
            ctk.CTkLabel(info_col, text=f"ID: {acc.get('student_id', '')}",
                         font=ctk.CTkFont(size=12, weight="bold"), text_color=name_color).pack(anchor="w")
            ctk.CTkLabel(info_col, text=f"Name: {acc.get('name', '')}  •  [{status}]",
                         font=ctk.CTkFont(size=11), text_color=status_color).pack(anchor="w")

            # Buttons on the right — full width so text is not cut off
            btn_col = ctk.CTkFrame(card, fg_color="transparent")
            btn_col.pack(side="right", padx=8, pady=8)

            if is_active:
                toggle_text = "Deactivate"
                toggle_color = "#FAB387"
            else:
                toggle_text = "Activate"
                toggle_color = "#A6E3A1"

            ctk.CTkButton(
                btn_col, text=toggle_text, width=90, height=28,
                fg_color=toggle_color, text_color="#11111B",
                font=ctk.CTkFont(size=12, weight="bold"), corner_radius=6,
                command=lambda sid=acc.get("student_id"): self._toggle_acc(sid)
            ).pack(side="left", padx=4)

            ctk.CTkButton(
                btn_col, text="Delete", width=70, height=28,
                fg_color="#F38BA8", text_color="#11111B",
                font=ctk.CTkFont(size=12, weight="bold"), corner_radius=6,
                command=lambda sid=acc.get("student_id"): self._delete_acc(sid)
            ).pack(side="left", padx=4)

    def _select_pic(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp")])
        if path:
            self.signup_pic_path = path
            self.btn_pic.configure(text=f"✓ {os.path.basename(path)[:28]}")

    def _handle_login(self):
        name = self.login_name.get().strip()
        sid = self.login_id.get().strip()
        pwd = self.login_pass.get().strip()
        if not sid or not pwd:
            self.login_msg.configure(text="Enter Student ID and Password.", text_color="#F38BA8")
            return

        accounts = db.load_all().get("accounts", [])
        acc = next((a for a in accounts if a.get("student_id") == sid and a.get("password") == pwd), None)

        if not acc:
            # also try name match for older accounts
            acc = next((a for a in accounts if a.get("name", "").lower() == name.lower() and a.get("password") == pwd), None)

        if not acc:
            self.login_msg.configure(text="Invalid credentials.", text_color="#F38BA8")
            return

        status = acc.get("status", "Active")
        if status != "Active":
            self.login_msg.configure(
                text="Account is Deactivated. Go to Manage Accounts → Activate.",
                text_color="#FAB387"
            )
            return

        db.set_last_active_user(acc.get("student_id"))
        self.login_msg.configure(text="Logging in...", text_color="#A6E3A1")
        self.after(400, lambda: self.on_login_success(acc.get("student_id")))

    def _handle_signup(self):
        name = self.signup_entries["name"].get().strip()
        sid = self.signup_entries["student_id"].get().strip()
        pwd = self.signup_entries["password"].get().strip()
        if not name or not sid or not pwd:
            self.signup_msg.configure(text="Name, Student ID and Password are required.", text_color="#F38BA8")
            return

        all_db = db.load_all()
        if any(a.get("student_id") == sid for a in all_db.get("accounts", [])):
            self.signup_msg.configure(text="Student ID already registered.", text_color="#FAB387")
            return

        acc = {
            "name": name, "student_id": sid, "password": pwd,
            "status": "Active", "profile_pic": self.signup_pic_path,
            "created_at": datetime.now().strftime("%Y-%m-%d")
        }
        all_db.setdefault("accounts", []).append(acc)
        db.save_all(all_db)

        # seed profile
        profile = db.get_user_profile(sid)
        sp = profile["student_profile"]
        sp["name"] = name
        sp["student_id"] = sid
        sp["profile_pic"] = self.signup_pic_path
        for key in ["university", "faculty", "department", "degree", "major", "minor",
                    "semester_year", "batch", "graduation_year", "location", "academic_level"]:
            if key in self.signup_entries:
                sp[key] = self.signup_entries[key].get().strip()
        db.save_user_profile(sid, profile)

        self.signup_msg.configure(text="Account created! Please log in.", text_color="#A6E3A1")
        self.after(700, lambda: self.auth_tabs.set("Login"))
        self._build_manage_tab()

    def _toggle_acc(self, student_id):
        new_status = db.toggle_account_active(student_id)
        # Rebuild list so button text switches Activate <-> Deactivate
        self._build_manage_tab()
        # Small feedback label at top of manage tab
        for w in self.tab_manage.winfo_children():
            if isinstance(w, ctk.CTkLabel) and "toggled" in (w.cget("text") or "").lower():
                w.destroy()
        # The rebuild already happened; status is persisted.

    def _delete_acc(self, student_id):
        db.delete_account(student_id)
        self._build_manage_tab()


# ==============================================================================
# MAIN APP
# ==============================================================================
class AcadexaAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Acadexa AI — Universal Academic & Career Intelligence Ecosystem")
        self.geometry("1320x860")
        self.minsize(1100, 720)
        self.current_user_id = None
        self.user_data = {}

        # timer state
        self.timer_running = False
        self.timer_seconds = 0
        self.subject_queue = []
        self.current_subject_idx = 0
        self.phone_detection_counter = 0.0
        self.table_leave_count = 0
        self.session_failed = False
        self.concentration_percent = 0.0
        self._face_absent_streak = 0
        self._prev_gray = None
        self._leave_armed = True  # one leave per absence; re-arm only after return to desk
        self.session_timeline = []  # [{t_sec, conc, mode}] during active session
        self.cap = None
        self.face_cascade = None
        self.face_cascade_alt = None
        self.profile_cascade = None
        self.eye_cascade = None
        self.eye_cascade_glasses = None
        self.yunet_detector = None
        self._eye_absent_streak = 0
        self._last_eye_boxes = []
        self._load_face_models()

        # quiz state
        self.quiz_answers = []
        self.quiz_meta = {}

        self.auth_frame = AuthWindow(self, on_login_success=self._on_login)

    def _ensure_yunet_model(self):
        model_dir = os.path.join(os.path.expanduser("~"), ".acadexa")
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        if os.path.isfile(path) and os.path.getsize(path) > 10000:
            return path
        try:
            import urllib.request
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, path)
            if os.path.isfile(path) and os.path.getsize(path) > 10000:
                return path
        except Exception:
            pass
        return None

    def _load_face_models(self):
        self.face_cascade = self.face_cascade_alt = self.profile_cascade = None
        self.eye_cascade = self.eye_cascade_glasses = self.yunet_detector = None
        if not ensure_cv2():
            return
        CC = getattr(cv2, "CascadeClassifier", None)
        if CC is None:
            try:
                from cv2 import xobjdetect
                CC = getattr(xobjdetect, "CascadeClassifier", None)
            except Exception:
                CC = None
        if CC is not None:
            try:
                base = getattr(getattr(cv2, "data", None), "haarcascades", "") or ""
                for p in [base + "haarcascade_frontalface_default.xml", "haarcascade_frontalface_default.xml"]:
                    if not p:
                        continue
                    fc = CC(p)
                    if fc is not None and not fc.empty():
                        self.face_cascade = fc
                        break
                if base:
                    alt = CC(base + "haarcascade_frontalface_alt2.xml")
                    if alt is not None and not alt.empty():
                        self.face_cascade_alt = alt
                    prof = CC(base + "haarcascade_profileface.xml")
                    if prof is not None and not prof.empty():
                        self.profile_cascade = prof
                    # Eye trackers
                    eye = CC(base + "haarcascade_eye.xml")
                    if eye is not None and not eye.empty():
                        self.eye_cascade = eye
                    eye_g = CC(base + "haarcascade_eye_tree_eyeglasses.xml")
                    if eye_g is not None and not eye_g.empty():
                        self.eye_cascade_glasses = eye_g
            except Exception:
                pass
        if self.face_cascade is None and hasattr(cv2, "FaceDetectorYN"):
            try:
                model_path = self._ensure_yunet_model()
                if model_path:
                    self.yunet_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.7, 0.3, 5000)
            except Exception:
                self.yunet_detector = None

    def _on_login(self, student_id):
        self.current_user_id = student_id
        self.user_data = db.get_user_profile(student_id)
        self.auth_frame.destroy()
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self.tabview = ctk.CTkTabview(self, fg_color="#11111B",
                                      segmented_button_fg_color="#181825",
                                      segmented_button_selected_color="#89B4FA")
        self.tabview.grid(row=0, column=1, sticky="nsew", padx=12, pady=10)
        self.tab_dash = self.tabview.add("Dashboard")
        self.tab_courses = self.tabview.add("Courses & Subjects")
        self.tab_routine = self.tabview.add("Smart Routine")
        self.tab_quiz = self.tabview.add("Universal Quiz Engine")
        self.tab_timer = self.tabview.add("Focus Timer")
        self.tab_ats = self.tabview.add("ATS & Job Matcher")
        self.tab_resume = self.tabview.add("CV & Resume Builder")
        self.tab_tasks = self.tabview.add("Task Manager")
        self.tab_brain = self.tabview.add("Brain Testing")
        self.tab_profile = self.tabview.add("Student Intelligence Profile")
        self._build_dashboard()
        self._build_courses()
        self._build_routine()
        self._build_quiz()
        self._build_timer()
        self._build_ats()
        self._build_resume()
        self._build_tasks()
        self._build_brain()
        self._build_profile()
        self.refresh_ui()

    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color="#1E1E2E")
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        side.grid_rowconfigure(12, weight=1)

        sp = self.user_data.get("student_profile", {})
        pic = sp.get("profile_pic", "") or db.get_last_active_user_pic()
        self.side_avatar = ctk.CTkLabel(side, text="", image=make_circular_image(pic, size=(64, 64)))
        self.side_avatar.grid(row=0, column=0, padx=20, pady=(20, 6))

        ctk.CTkLabel(side, text="ACADEXA AI", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#89B4FA").grid(row=1, column=0, padx=16, sticky="w")
        self.side_user_lbl = ctk.CTkLabel(side, text=f"Active: {sp.get('name', self.current_user_id)}",
                                          font=ctk.CTkFont(size=11), text_color="#A6E3A1")
        self.side_user_lbl.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="w")

        nav = [
            ("Intelligence Dashboard", "Dashboard"),
            ("Universal Course Manager", "Courses & Subjects"),
            ("Smart Study Routine", "Smart Routine"),
            ("Adaptive Quiz Engine", "Universal Quiz Engine"),
            ("Focus Session Tracker", "Focus Timer"),
            ("ATS & Job Matcher", "ATS & Job Matcher"),
            ("CV & Resume Builder", "CV & Resume Builder"),
            ("Task Manager", "Task Manager"),
            ("Brain Testing", "Brain Testing"),
            ("Student Profile", "Student Intelligence Profile"),
        ]
        for i, (label, tab) in enumerate(nav, start=3):
            ctk.CTkButton(side, text=label, fg_color="#313244", text_color="#CDD6F4",
                          hover_color="#45475A", anchor="w", height=34,
                          command=lambda t=tab: self.tabview.set(t)).grid(
                row=i, column=0, padx=12, pady=3, sticky="ew")

        ctk.CTkButton(side, text="Logout", fg_color="#F38BA8", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), height=34,
                      command=self._logout).grid(row=14, column=0, padx=12, pady=16, sticky="ew")

    def _logout(self):
        """Return to Login / Sign Up / Manage Accounts — do not close the app."""
        self.timer_running = False
        try:
            self._release_camera()
        except Exception:
            pass
        # destroy main UI widgets
        for w in list(self.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self.current_user_id = None
        self.user_data = {}
        self.subject_queue = []
        self.timer_seconds = 0
        self.session_timeline = []
        self.auth_frame = AuthWindow(self, on_login_success=self._on_login)

    def _render_sidebar_profile(self):
        sp = self.user_data.get("student_profile", {})
        pic = sp.get("profile_pic", "")
        if hasattr(self, "side_avatar"):
            self.side_avatar.configure(image=make_circular_image(pic, size=(64, 64)))
        if hasattr(self, "side_user_lbl"):
            self.side_user_lbl.configure(text=f"Active: {sp.get('name', self.current_user_id)}")

    # ---------- DASHBOARD ----------
    def _build_dashboard(self):
        root = ctk.CTkScrollableFrame(self.tab_dash, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=4, pady=4)
        root.grid_columnconfigure(0, weight=1)

        # --- Header ---
        header = ctk.CTkFrame(root, fg_color="#181825", corner_radius=12)
        header.grid(row=0, column=0, sticky="ew", pady=(4, 10), padx=4)
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Intelligence Dashboard",
                     font=ctk.CTkFont(size=20, weight="bold"), text_color="#CDD6F4").grid(
            row=0, column=0, sticky="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(header, text="Click a metric card → chart updates in the panel below (stays open)",
                     font=ctk.CTkFont(size=11), text_color="#A6ADC8").grid(
            row=1, column=0, sticky="w", padx=16, pady=(2, 12))

        # --- Metrics grid (2 × 4) ---
        metrics = ctk.CTkFrame(root, fg_color="transparent")
        metrics.grid(row=1, column=0, sticky="ew", padx=4)
        for i in range(4):
            metrics.grid_columnconfigure(i, weight=1, uniform="m")

        accents = {
            "tasks": "#89B4FA", "mastery": "#A6E3A1", "readiness": "#FAB387", "routine": "#CBA6F7",
            "hours": "#94E2D5", "consistency": "#F5C2E7", "quizzes": "#F9E2AF", "habit": "#74C7EC",
        }
        self.card_academic = self._metric_card(metrics, "Task Completion", "0%", 0, 0, "tasks", accents["tasks"])
        self.card_mastery = self._metric_card(metrics, "Knowledge Mastery", "0%", 0, 1, "mastery", accents["mastery"])
        self.card_readiness = self._metric_card(metrics, "Career Readiness", "0%", 0, 2, "readiness", accents["readiness"])
        self.card_routine = self._metric_card(metrics, "Routine Efficiency", "0%", 0, 3, "routine", accents["routine"])
        self.card_hours = self._metric_card(metrics, "Study Hours", "0.0 hrs", 1, 0, "hours", accents["hours"])
        self.card_consistency = self._metric_card(metrics, "Learning Consistency", "0%", 1, 1, "consistency", accents["consistency"])
        self.card_quizzes = self._metric_card(metrics, "Quizzes Taken", "0", 1, 2, "quizzes", accents["quizzes"])
        self.card_status = self._metric_card(metrics, "Habit Health", "0%", 1, 3, "habit", accents["habit"])

        # --- Chart panel (primary visual) ---
        chart_wrap = ctk.CTkFrame(root, fg_color="#181825", corner_radius=12)
        chart_wrap.grid(row=2, column=0, sticky="ew", padx=4, pady=(12, 8))
        chart_wrap.grid_columnconfigure(0, weight=1)
        self.lbl_chart_title = ctk.CTkLabel(
            chart_wrap, text="Chart · Focus / Study Health",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#89B4FA")
        self.lbl_chart_title.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))
        self.dash_chart_host = ctk.CTkFrame(chart_wrap, fg_color="#11111B", corner_radius=10, height=260)
        self.dash_chart_host.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 12))
        self.dash_chart_host.grid_propagate(False)
        ctk.CTkLabel(self.dash_chart_host,
                     text="Click any metric card above to load its chart here.",
                     text_color="#A6ADC8").pack(expand=True, pady=40)
        self._current_chart_key = "focus"
        self._chart_canvas = None

        # --- Bottom: Recommendations | Assessments (2 columns) ---
        bottom = ctk.CTkFrame(root, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", padx=4, pady=(4, 4))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)

        rec_card = ctk.CTkFrame(bottom, fg_color="#181825", corner_radius=12)
        rec_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(rec_card, text="Recommendations", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", padx=14, pady=(12, 6))
        self.lbl_recommendations = ctk.CTkLabel(
            rec_card, text="—", font=ctk.CTkFont(size=12),
            text_color="#CDD6F4", justify="left", wraplength=420, anchor="nw")
        self.lbl_recommendations.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        assess_card = ctk.CTkFrame(bottom, fg_color="#181825", corner_radius=12)
        assess_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ctk.CTkLabel(assess_card, text="Recent Topic Assessments", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", padx=14, pady=(12, 6))
        self.assess_box = ctk.CTkFrame(assess_card, fg_color="#11111B", corner_radius=8)
        self.assess_box.pack(fill="both", expand=True, padx=10, pady=(0, 12))

        # --- Footer actions ---
        footer = ctk.CTkFrame(root, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="ew", padx=4, pady=(8, 16))
        ctk.CTkButton(
            footer, text="Export Dashboard PDF", fg_color="#89B4FA", text_color="#11111B",
            font=ctk.CTkFont(size=12, weight="bold"), width=170, height=32,
            command=self._export_dashboard_data
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            footer, text="Clear My History", fg_color="#313244", hover_color="#F38BA8",
            text_color="#CDD6F4", font=ctk.CTkFont(size=12), width=150, height=32,
            command=self._clear_my_history
        ).pack(side="left")
        self.lbl_clear_msg = ctk.CTkLabel(footer, text="", text_color="#A6ADC8", font=ctk.CTkFont(size=11))
        self.lbl_clear_msg.pack(side="left", padx=12)

    def _metric_card(self, parent, title, val, r, c, chart_key=None, accent="#89B4FA"):
        card = ctk.CTkFrame(parent, fg_color="#181825", corner_radius=12,
                            border_width=1, border_color="#313244")
        card.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
        # accent bar
        bar = ctk.CTkFrame(card, fg_color=accent, width=4, height=56, corner_radius=2)
        bar.pack(side="left", fill="y", padx=(8, 0), pady=10)
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        ctk.CTkLabel(body, text=title, font=ctk.CTkFont(size=11), text_color="#A6ADC8").pack(anchor="w")
        lbl = ctk.CTkLabel(body, text=val, font=ctk.CTkFont(size=20, weight="bold"), text_color=accent)
        lbl.pack(anchor="w", pady=(2, 0))
        hint = ctk.CTkLabel(body, text="Tap for chart", font=ctk.CTkFont(size=10), text_color="#585B70")
        hint.pack(anchor="w")
        if chart_key:
            def _bind(w, k=chart_key):
                w.bind("<Button-1>", lambda e, key=k: self._open_metric_chart(key))
                try:
                    w.configure(cursor="hand2")
                except Exception:
                    pass
            for w in (card, bar, body, lbl, hint):
                _bind(w)
        return lbl

    def _export_dashboard_data(self):
        """Export full dashboard as PDF (metrics + all charts). Not Excel."""
        from tkinter import filedialog
        import io
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                Table, TableStyle, PageBreak, KeepTogether
            )
            from reportlab.lib import colors
        except ImportError:
            if hasattr(self, "lbl_clear_msg"):
                self.lbl_clear_msg.configure(
                    text="pip install reportlab  needed for PDF export", text_color="#F38BA8")
            return

        ev = decision_agent.evaluate_student(self.user_data)
        study_ev = study_intelligence.evaluate(self.user_data.get("study_sessions", []))
        sp = self.user_data.get("student_profile", {})
        sessions = self.user_data.get("study_sessions", [])
        quizzes = self.user_data.get("quiz_history", [])
        mastery = self.user_data.get("topic_mastery", {})
        tasks = self.user_data.get("tasks", [])
        hb = ev.get("habit_breakdown") or {}

        default_name = f"acadexa_dashboard_{self.current_user_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        path = filedialog.asksaveasfilename(
            title="Export Dashboard PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF file", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            if hasattr(self, "lbl_clear_msg"):
                self.lbl_clear_msg.configure(text="Export cancelled.", text_color="#FAB387")
            return

        def fig_to_rl(fig, w=16*cm, h=7*cm):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            return RLImage(buf, width=w, height=h)

        try:
            doc = SimpleDocTemplate(path, pagesize=A4,
                                    leftMargin=1.5*cm, rightMargin=1.5*cm,
                                    topMargin=1.5*cm, bottomMargin=1.5*cm)
            styles = getSampleStyleSheet()
            title_st = ParagraphStyle("T", parent=styles["Heading1"], fontSize=18, spaceAfter=8)
            h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6)
            body = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=14)
            story = []

            story.append(Paragraph("Acadexa AI — Intelligence Dashboard Report", title_st))
            story.append(Paragraph(
                f"Student: {sp.get('name', '')} ({self.current_user_id}) &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
            story.append(Spacer(1, 0.4*cm))

            # Metrics table (8 cards)
            metrics_data = [
                ["Metric", "Value"],
                ["Task Completion", f"{ev.get('task_completion', 0)}%"],
                ["Knowledge Mastery", f"{ev.get('knowledge_mastery', 0)}%"],
                ["Career Readiness", f"{ev.get('job_readiness', 0)}%"],
                ["Routine Efficiency", f"{ev.get('routine_efficiency', 0)}%"],
                ["Study Hours", f"{study_ev.get('total_hours', 0)} hrs"],
                ["Learning Consistency", f"{ev.get('learning_intelligence', 0):.0f}%"],
                ["Quizzes Taken", str(len(quizzes))],
                ["Habit Health", f"{ev.get('habit_health', 0)}%"],
            ]
            t = Table(metrics_data, colWidths=[9*cm, 6*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#313244")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F5F7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A6ADC8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.5*cm))

            # Recommendations
            story.append(Paragraph("Recommendations", h2))
            for r in (ev.get("recommendations") or ["—"]):
                story.append(Paragraph(f"• {r}", body))

            # Habit breakdown chart
            story.append(Paragraph("Habit Health Breakdown", h2))
            fig, ax = plt.subplots(figsize=(7, 3.2))
            labels = list(hb.keys()) or ["no data"]
            vals = [float(hb.get(k, 0)) for k in labels] if hb else [0]
            ax.barh(labels, vals, color=["#89B4FA", "#A6E3A1", "#FAB387", "#F5C2E7", "#CBA6F7", "#74C7EC"][:len(labels)])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score")
            ax.set_title(f"Habit Health = {ev.get('habit_health', 0)}%")
            fig.tight_layout()
            story.append(fig_to_rl(fig))

            # Focus / concentration chart
            story.append(Paragraph("Focus Sessions (Concentration %)", h2))
            fig, ax = plt.subplots(figsize=(7, 3.2))
            if sessions:
                labels = [s.get("subject", "?")[:14] for s in sessions[-10:]]
                vals = [s.get("concentration_percent", 0) for s in sessions[-10:]]
                ax.bar(range(len(labels)), vals, color="#89B4FA")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
                ax.set_ylabel("Concentration %")
                ax.set_ylim(0, 105)
            else:
                ax.text(0.5, 0.5, "No focus sessions", ha="center", va="center")
                ax.set_axis_off()
            fig.tight_layout()
            story.append(fig_to_rl(fig))

            # Mastery pie
            story.append(Paragraph("Knowledge Mastery by Topic", h2))
            fig, ax = plt.subplots(figsize=(6, 4))
            items = list(mastery.items())[:8]
            if items:
                ax.pie([max(0.1, float(v)) for _, v in items],
                       labels=[k[:16] for k, _ in items], autopct="%1.0f%%",
                       textprops={"fontsize": 8})
            else:
                ax.text(0.5, 0.5, "No quiz mastery data", ha="center", va="center")
                ax.set_axis_off()
            fig.tight_layout()
            story.append(fig_to_rl(fig, w=12*cm, h=9*cm))

            # Task completion progress
            story.append(PageBreak())
            story.append(Paragraph("Task Completion Progress", h2))
            fig, ax = plt.subplots(figsize=(7, 3))
            if tasks:
                done_c, cum = 0, []
                for i, tsk in enumerate(tasks, 1):
                    if tsk.get("done"):
                        done_c += 1
                    cum.append(100.0 * done_c / len(tasks))
                ax.plot(range(1, len(cum) + 1), cum, marker="o", color="#89B4FA")
                ax.fill_between(range(1, len(cum) + 1), cum, alpha=0.25, color="#89B4FA")
                ax.set_ylim(0, 105)
                ax.set_xlabel("Task #")
                ax.set_ylabel("Completion %")
            else:
                ax.text(0.5, 0.5, "No tasks", ha="center", va="center")
                ax.set_axis_off()
            fig.tight_layout()
            story.append(fig_to_rl(fig))

            # Session table
            story.append(Paragraph("Recent Study Sessions", h2))
            rows = [["Date", "Subject", "Min", "Conc%", "Leaves"]]
            for s in sessions[-12:]:
                rows.append([
                    str(s.get("date", ""))[:16],
                    str(s.get("subject", ""))[:28],
                    str(s.get("duration_minutes", "")),
                    str(s.get("concentration_percent", "")),
                    str(s.get("table_leaves", "")),
                ])
            if len(rows) == 1:
                rows.append(["—", "No sessions", "—", "—", "—"])
            st = Table(rows, colWidths=[3.2*cm, 6*cm, 1.5*cm, 2*cm, 2*cm])
            st.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#313244")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(st)

            # Quiz history
            story.append(Paragraph("Quiz History", h2))
            qrows = [["Date", "Topic", "Score", "Status"]]
            for q in quizzes[-12:]:
                qrows.append([
                    str(q.get("date", ""))[:16],
                    str(q.get("topic", ""))[:30],
                    str(q.get("score", "")),
                    str(q.get("status", "")),
                ])
            if len(qrows) == 1:
                qrows.append(["—", "No quizzes", "—", "—"])
            qt = Table(qrows, colWidths=[3.2*cm, 7*cm, 2*cm, 3*cm])
            qt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#313244")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(qt)

            # Study materials inventory
            mats = self.user_data.get("study_materials", [])
            story.append(Paragraph("Pre-Study Materials", h2))
            mrows = [["Name", "Added", "Text chars"]]
            for m in mats[-15:]:
                mrows.append([
                    str(m.get("name", ""))[:40],
                    str(m.get("added", ""))[:16],
                    str(len(m.get("text") or "")),
                ])
            if len(mrows) == 1:
                mrows.append(["No materials uploaded", "—", "—"])
            mt = Table(mrows, colWidths=[8*cm, 4*cm, 3*cm])
            mt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#313244")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(mt)

            # Brain training scores
            brains = self.user_data.get("brain_scores", [])
            story.append(Paragraph("Brain Testing Scores", h2))
            brows = [["Date", "Game", "Level", "Score", "Detail"]]
            for b in brains[-12:]:
                brows.append([
                    str(b.get("date", ""))[:16],
                    str(b.get("game", "")),
                    str(b.get("level", "")),
                    str(b.get("score", "")),
                    str(b.get("detail", ""))[:40],
                ])
            if len(brows) == 1:
                brows.append(["—", "No games", "—", "—", "—"])
            bt = Table(brows, colWidths=[3*cm, 2.5*cm, 1.5*cm, 1.5*cm, 6.5*cm])
            bt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#313244")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(bt)

            doc.build(story)
            if hasattr(self, "lbl_clear_msg"):
                self.lbl_clear_msg.configure(
                    text=f"PDF saved: {os.path.basename(path)}", text_color="#A6E3A1")
        except Exception as e:
            if hasattr(self, "lbl_clear_msg"):
                self.lbl_clear_msg.configure(text=f"PDF export failed: {e}", text_color="#F38BA8")

    def _clear_my_history(self):
        """
        Dashboard analytics only:
          CLEARS → study_sessions, quiz_history, topic_mastery, weak_topics
          KEEPS  → login/accounts (Manage Accounts), password, student_profile,
                   courses, study_routine, daily_schedule
        """
        # Only wipe learning-history fields used by dashboard charts/metrics
        self.user_data["study_sessions"] = []
        self.user_data["quiz_history"] = []
        self.user_data["topic_mastery"] = {}
        self.user_data["weak_topics"] = []
        # NEVER touch accounts DB / login credentials here
        db.save_user_profile(self.current_user_id, self.user_data)
        self._current_chart_key = "focus"
        if hasattr(self, "lbl_clear_msg"):
            self.lbl_clear_msg.configure(
                text="Dashboard history cleared. Login & accounts unchanged.",
                text_color="#A6E3A1")
        self.refresh_ui()

    def _open_metric_chart(self, key):
        """Render chart inside dashboard panel — stays visible, no popup."""
        self._current_chart_key = key
        self._render_embedded_chart(key)

    def _draw_metric_figure(self, key, figsize=(8.2, 2.6)):
        sessions = self.user_data.get("study_sessions", [])
        mastery = self.user_data.get("topic_mastery", {})
        quizzes = self.user_data.get("quiz_history", [])
        routine = self.user_data.get("study_routine", [])
        ev = decision_agent.evaluate_student(self.user_data)

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        fig.patch.set_facecolor("#181825")
        ax.set_facecolor("#11111B")
        ax.tick_params(colors="#A6ADC8")
        for spine in ax.spines.values():
            spine.set_color("#313244")
        ax.title.set_color("#CDD6F4")
        ax.xaxis.label.set_color("#A6ADC8")
        ax.yaxis.label.set_color("#A6ADC8")

        title_map = {
            "tasks": "Task Completion progress",
            "habit": "Habit Health breakdown",
            "focus": "Focus timeline",
            "mastery": "Knowledge Mastery",
            "readiness": "Career Readiness",
            "routine": "Routine Efficiency",
            "hours": "Study Hours",
            "consistency": "Learning Consistency",
            "quizzes": "Quizzes Taken",
        }

        if key == "habit":
            hb = ev.get("habit_breakdown") or {
                "consistency": ev.get("learning_intelligence", 0),
                "avg_concentration": 0,
                "task_completion": ev.get("task_completion", 0),
                "quiz_activity": 0,
                "subject_balance": 0,
                "brain_training": 0,
            }
            labels = list(hb.keys())
            vals = [float(hb[k]) for k in labels]
            colors = ["#89B4FA", "#A6E3A1", "#FAB387", "#F5C2E7", "#CBA6F7", "#74C7EC"]
            ax.barh(labels, vals, color=colors[:len(labels)])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score")
            ax.axvline(ev.get("habit_health", 0), color="#F38BA8", linestyle="--", linewidth=1.5)
            ax.set_title(f"Habit Health {ev.get('habit_health', 0)}% (includes brain games)")
        elif key == "tasks":
            tasks = self.user_data.get("tasks", [])
            # cumulative completion % over task order (progress line goes up as more done)
            if tasks:
                labels, cum = [], []
                done = 0
                for i, t in enumerate(tasks, start=1):
                    if t.get("done"):
                        done += 1
                    labels.append(t.get("date", "")[:10] or f"#{i}")
                    cum.append(round(100.0 * done / len(tasks), 1))
                ax.plot(range(len(cum)), cum, color="#89B4FA", linewidth=2, marker="o", markersize=4)
                ax.fill_between(range(len(cum)), cum, alpha=0.25, color="#89B4FA")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
                ax.set_ylim(0, 105)
                ax.set_ylabel("Completion %")
                ax.set_xlabel("Tasks (by add order)")
            else:
                ax.text(0.5, 0.5, "No tasks yet — open Task Manager", ha="center",
                        va="center", color="#A6ADC8", transform=ax.transAxes)
                ax.set_axis_off()
        elif key == "focus":
            timelines = []
            for s in sessions[-5:]:
                tl = s.get("timeline") or []
                if tl:
                    timelines.append((s.get("subject", "Session")[:22], tl))
            if timelines:
                for name, tl in timelines[-3:]:
                    xs = [p.get("t_sec", i) / 60 for i, p in enumerate(tl)]
                    ys = [p.get("conc", 0) for p in tl]
                    ax.plot(xs, ys, label=name, linewidth=2)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Minutes into session")
                ax.set_ylabel("Concentration %")
                ax.axhline(80, color="#A6E3A1", linestyle="--", alpha=0.5)
                ax.legend(facecolor="#181825", labelcolor="#CDD6F4", fontsize=8)
            else:
                labels = [s.get("subject", "?")[:12] for s in sessions[-8:]] or ["No data"]
                vals = [s.get("concentration_percent", 0) for s in sessions[-8:]] or [0]
                ax.bar(range(len(labels)), vals, color="#89B4FA")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
                ax.set_ylabel("Concentration %")
        elif key == "hours":
            labels = [s.get("subject", "?")[:12] for s in sessions[-10:]] or ["No data"]
            vals = [s.get("duration_minutes", 0) for s in sessions[-10:]] or [0]
            ax.bar(range(len(labels)), vals, color="#A6E3A1")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Minutes")
        elif key == "consistency":
            labels = [s.get("date", "")[-11:] for s in sessions[-10:]] or ["No data"]
            vals = [s.get("completion_rate", 0) for s in sessions[-10:]] or [0]
            ax.bar(range(len(labels)), vals, color="#CBA6F7")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Completion %")
        elif key == "mastery":
            items = list(mastery.items())[:10] or [("No quizzes yet", 1)]
            labels = [k[:16] for k, _ in items]
            vals = [max(0.1, float(v)) for _, v in items]
            ax.pie(vals, labels=labels, autopct="%1.0f%%",
                   textprops={"color": "#CDD6F4", "fontsize": 8},
                   colors=plt.cm.Blues(np.linspace(0.4, 0.9, len(vals))))
        elif key == "readiness":
            parts = {
                "Mastery": max(0.1, ev.get("knowledge_mastery", 0)),
                "Skills": max(0.1, ev.get("skill_strength", 0)),
                "Job prep": max(0.1, ev.get("job_readiness", 0)),
            }
            ax.pie(list(parts.values()), labels=list(parts.keys()), autopct="%1.0f%%",
                   textprops={"color": "#CDD6F4", "fontsize": 9},
                   colors=["#89B4FA", "#A6E3A1", "#FAB387"])
        elif key == "routine":
            study_n = sum(1 for r in routine if r.get("type") == "study")
            fixed_n = sum(1 for r in routine if r.get("type") == "fixed")
            if study_n + fixed_n == 0:
                study_n, fixed_n = 0, 1
            ax.pie([max(0.1, study_n), max(0.1, fixed_n)],
                   labels=["Study blocks", "Fixed blocks"], autopct="%1.0f%%",
                   textprops={"color": "#CDD6F4"}, colors=["#89B4FA", "#585B70"])
        elif key == "quizzes":
            if quizzes:
                labels = [q.get("topic", "?")[:14] for q in quizzes[-8:]]
                vals = [q.get("score", 0) for q in quizzes[-8:]]
                ax.bar(range(len(labels)), vals, color="#F5C2E7")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
                ax.set_ylabel("Score %")
            else:
                ax.text(0.5, 0.5, "No quizzes yet", ha="center", va="center", color="#A6ADC8")
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "No data", ha="center", color="#CDD6F4")
            ax.set_axis_off()

        ax.set_title(title_map.get(key, key), color="#CDD6F4", fontsize=11)
        fig.tight_layout()
        return fig

    def _render_embedded_chart(self, key=None):
        """Draw selected chart permanently on dashboard (not a popup)."""
        if not hasattr(self, "dash_chart_host"):
            return
        key = key or getattr(self, "_current_chart_key", "focus")
        for w in self.dash_chart_host.winfo_children():
            w.destroy()
        if self._chart_canvas is not None:
            try:
                plt.close(self._chart_canvas.figure)
            except Exception:
                pass
            self._chart_canvas = None
        try:
            fig = self._draw_metric_figure(key)
            canvas = FigureCanvasTkAgg(fig, master=self.dash_chart_host)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
            self._chart_canvas = canvas
            if hasattr(self, "lbl_chart_title"):
                self.lbl_chart_title.configure(text=f"Showing: {key}  (click another card to switch)")
        except Exception as e:
            ctk.CTkLabel(self.dash_chart_host, text=f"Chart error: {e}", text_color="#F38BA8").pack(pady=20)

    def _render_dash_focus_chart(self):
        """Backward-compatible: show current or focus chart on dashboard."""
        self._render_embedded_chart(getattr(self, "_current_chart_key", "focus"))

    # ---------- COURSES ----------
    def _build_courses(self):
        frame = ctk.CTkScrollableFrame(self.tab_courses, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        ctk.CTkLabel(frame, text="Universal Course Manager (up to 10 subjects × 10 topics)",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#CDD6F4").pack(anchor="w", pady=(0, 8))

        form = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12)
        form.pack(fill="x", pady=6)
        row1 = ctk.CTkFrame(form, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=8)
        self.ent_course_code = ctk.CTkEntry(row1, placeholder_text="Code e.g. CSE-3101", width=140,
                                            fg_color="#11111B", border_color="#313244")
        self.ent_course_code.pack(side="left", padx=4)
        self.ent_course_title = ctk.CTkEntry(row1, placeholder_text="Title e.g. Operating System",
                                             fg_color="#11111B", border_color="#313244")
        self.ent_course_title.pack(side="left", fill="x", expand=True, padx=4)
        self.ent_course_domain = ctk.CTkEntry(row1, placeholder_text="Domain e.g. CSE", width=100,
                                              fg_color="#11111B", border_color="#313244")
        self.ent_course_domain.pack(side="left", padx=4)

        ctk.CTkLabel(form, text="Topics (comma separated, max 10)", text_color="#A6ADC8",
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=16)
        self.ent_course_topics = ctk.CTkEntry(form, placeholder_text="process, scheduling, deadlock, memory...",
                                              fg_color="#11111B", border_color="#313244")
        self.ent_course_topics.pack(fill="x", padx=12, pady=(2, 8))

        ctk.CTkButton(form, text="Add Subject", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._add_course).pack(fill="x", padx=12, pady=(0, 10))

        self.lbl_course_count = ctk.CTkLabel(frame, text="Subjects: 0 / 10", text_color="#A6ADC8")
        self.lbl_course_count.pack(anchor="w", pady=(8, 4))
        self.course_scroll = ctk.CTkScrollableFrame(frame, fg_color="#181825", corner_radius=10, height=320)
        self.course_scroll.pack(fill="both", expand=True)

    def _add_course(self):
        courses = self.user_data.setdefault("courses", [])
        if len(courses) >= 10:
            return
        title = self.ent_course_title.get().strip()
        if not title:
            return
        topics = [t.strip() for t in self.ent_course_topics.get().split(",") if t.strip()][:10]
        courses.append({
            "code": self.ent_course_code.get().strip(),
            "title": title,
            "discipline": self.ent_course_domain.get().strip() or "General",
            "topics": topics,
            "mastery": 0.0
        })
        db.save_user_profile(self.current_user_id, self.user_data)
        self.ent_course_code.delete(0, "end")
        self.ent_course_title.delete(0, "end")
        self.ent_course_topics.delete(0, "end")
        self.refresh_ui()

    def _remove_course(self, idx):
        courses = self.user_data.get("courses", [])
        if 0 <= idx < len(courses):
            courses.pop(idx)
            db.save_user_profile(self.current_user_id, self.user_data)
            self.refresh_ui()

    # ---------- ROUTINE ----------
    def _build_routine(self):
        outer = ctk.CTkFrame(self.tab_routine, fg_color="transparent")
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(outer, fg_color="#181825", corner_radius=12, width=340)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8), pady=6)

        def add_field(parent, label, default=""):
            ctk.CTkLabel(parent, text=label, text_color="#CDD6F4", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=12, pady=(6, 1))
            e = ctk.CTkEntry(parent, fg_color="#11111B", border_color="#313244")
            if default:
                e.insert(0, default)
            e.pack(fill="x", padx=12, pady=(0, 2))
            return e

        self.ent_wake = add_field(left, "Wake-up Time", "06:30")
        self.ent_sleep = add_field(left, "Sleep Time", "23:00")
        self.ent_breakfast = add_field(left, "Breakfast Time", "07:30")
        self.ent_lunch = add_field(left, "Lunch Time", "13:00")
        self.ent_dinner = add_field(left, "Dinner Time", "20:00")
        self.ent_meal_mins = add_field(left, "Meal Duration (minutes)", "30")
        self.ent_shower = add_field(left, "Shower / Fresh-up Time", "07:00")
        self.ent_shower_mins = add_field(left, "Shower Duration (mins)", "20")
        self.ent_wash_mins = add_field(left, "Avg Washroom Buffer (mins)", "10")

        ctk.CTkLabel(left, text="University Class Days (select)", text_color="#CDD6F4").pack(anchor="w", padx=12, pady=(10, 4))
        days_frame = ctk.CTkFrame(left, fg_color="transparent")
        days_frame.pack(fill="x", padx=12)
        self.class_day_vars = {}
        for i, d in enumerate(["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]):
            v = ctk.BooleanVar(value=(d in ("Sun", "Mon", "Tue", "Wed", "Thu")))
            self.class_day_vars[d] = v
            ctk.CTkCheckBox(days_frame, text=d, variable=v, width=50,
                            fg_color="#89B4FA", hover_color="#74A8F0").grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky="w")

        self.ent_class_start = add_field(left, "Class Start Time", "08:00")
        self.ent_class_end = add_field(left, "Class End Time", "14:30")
        ctk.CTkLabel(left, text="Preferred Study Session Length", text_color="#CDD6F4").pack(anchor="w", padx=12, pady=(8, 2))
        self.cmb_session_len = ctk.CTkOptionMenu(left, values=["25 Mins", "30 Mins", "45 Mins", "60 Mins", "90 Mins"],
                                                 fg_color="#313244", button_color="#45475A")
        self.cmb_session_len.set("60 Mins")
        self.cmb_session_len.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(left, text="Generate Smart Routine", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._generate_routine).pack(fill="x", padx=12, pady=6)
        ctk.CTkButton(left, text="Save Current Inputs", fg_color="#313244", text_color="#CDD6F4",
                      command=self._save_schedule_inputs).pack(fill="x", padx=12, pady=(0, 12))

        right = ctk.CTkFrame(outer, fg_color="#181825", corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", pady=6)
        ctk.CTkLabel(right, text="Generated Study Routine (Table)", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", padx=14, pady=(12, 4))
        self.lbl_routine_status = ctk.CTkLabel(right, text="No routine yet.", text_color="#A6ADC8")
        self.lbl_routine_status.pack(anchor="w", padx=14)
        self.txt_routine_table = ctk.CTkTextbox(right, fg_color="#11111B", border_color="#313244",
                                                font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self.txt_routine_table.pack(fill="both", expand=True, padx=12, pady=10)

    def _save_schedule_inputs(self):
        day_map = {"Sat": "Saturday", "Sun": "Sunday", "Mon": "Monday", "Tue": "Tuesday",
                   "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}
        class_days = [day_map[d] for d, v in self.class_day_vars.items() if v.get()]
        sched = {
            "wake_up": self.ent_wake.get().strip(),
            "sleep_time": self.ent_sleep.get().strip(),
            "breakfast": self.ent_breakfast.get().strip(),
            "lunch": self.ent_lunch.get().strip(),
            "dinner": self.ent_dinner.get().strip(),
            "meal_mins": self.ent_meal_mins.get().strip(),
            "shower_time": self.ent_shower.get().strip(),
            "shower_mins": self.ent_shower_mins.get().strip(),
            "washroom_mins": self.ent_wash_mins.get().strip(),
            "class_days": class_days,
            "class_start": self.ent_class_start.get().strip(),
            "class_end": self.ent_class_end.get().strip(),
        }
        self.user_data["daily_schedule"] = sched
        db.save_user_profile(self.current_user_id, self.user_data)
        self.lbl_routine_status.configure(text="Inputs saved.", text_color="#A6E3A1")

    def _generate_routine(self):
        self._save_schedule_inputs()
        pref = int(self.cmb_session_len.get().split()[0])
        rows = routine_engine.generate(self.user_data.get("daily_schedule", {}),
                                       self.user_data.get("courses", []), pref)
        self.user_data["study_routine"] = rows
        db.save_user_profile(self.current_user_id, self.user_data)

        study_n = sum(1 for r in rows if r.get("type") == "study")
        self.lbl_routine_status.configure(text=f"Routine generated • {study_n} study blocks", text_color="#A6E3A1")

        lines = [f"{'Day':<12} {'Start':<7} {'End':<7} {'Mins':<6} Activity", "-" * 72]
        cur_day = ""
        for r in rows:
            if r["day"] != cur_day:
                cur_day = r["day"]
                lines.append(f"\n=== {cur_day} ===")
            mark = "[S]" if r.get("type") == "study" else "[ ]"
            lines.append(f"{mark} {r['day'][:3]:<8} {r['start']:<7} {r['end']:<7} {r['duration_mins']:<6} {r['activity']}")
        self.txt_routine_table.delete("1.0", "end")
        self.txt_routine_table.insert("1.0", "\n".join(lines))
        self.refresh_ui()

    # ---------- QUIZ ----------
    def _build_quiz(self):
        frame = ctk.CTkFrame(self.tab_quiz, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        top = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12)
        top.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(top, text="Universal Dynamic Quiz Generator", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#89B4FA").pack(side="left", padx=14, pady=10)

        ctrl = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12)
        ctrl.pack(fill="x", pady=(0, 8))
        inner = ctk.CTkFrame(ctrl, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(inner, text="Topic:", text_color="#CDD6F4").pack(side="left", padx=(0, 4))
        self.ent_quiz_topic = ctk.CTkEntry(inner, width=180, fg_color="#11111B", border_color="#313244")
        self.ent_quiz_topic.pack(side="left", padx=4)

        ctk.CTkLabel(inner, text="Type:", text_color="#CDD6F4").pack(side="left", padx=(10, 4))
        self.cmb_quiz_type = ctk.CTkOptionMenu(inner, values=["MCQ", "Conceptual", "Numerical", "Scenario-based", "Short Answer"],
                                               fg_color="#313244", button_color="#45475A", width=130)
        self.cmb_quiz_type.set("MCQ")
        self.cmb_quiz_type.pack(side="left", padx=4)

        ctk.CTkLabel(inner, text="Questions:", text_color="#CDD6F4").pack(side="left", padx=(10, 4))
        self.cmb_quiz_n = ctk.CTkOptionMenu(inner, values=["3", "4", "5", "6", "8", "10", "12", "15"],
                                            fg_color="#313244", button_color="#45475A", width=70)
        self.cmb_quiz_n.set("5")
        self.cmb_quiz_n.pack(side="left", padx=4)

        ctk.CTkButton(inner, text="Generate Quiz", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._generate_quiz).pack(side="left", padx=12)

        self.quiz_display = ctk.CTkScrollableFrame(frame, fg_color="#181825", corner_radius=12)
        self.quiz_display.pack(fill="both", expand=True)

    def _find_materials_for_topic(self, topic):
        """Return study_materials whose name/text match academic subject/topic."""
        topic_l = (topic or "").lower()
        matched = []
        for m in self.user_data.get("study_materials", []):
            blob = f"{m.get('name', '')} {m.get('text', '')}".lower()
            # also match against registered courses
            if topic_l and topic_l in blob:
                matched.append(m)
                continue
            for c in self.user_data.get("courses", []):
                code = (c.get("code") or "").lower()
                title = (c.get("title") or "").lower()
                topics = " ".join(c.get("topics") or []).lower()
                if (code and code in topic_l) or (title and title in topic_l) or topic_l in topics:
                    if code in blob or title in blob or any(t.lower() in blob for t in (c.get("topics") or []) if t):
                        matched.append(m)
                        break
        # de-dup
        seen, out = set(), []
        for m in matched:
            mid = m.get("id") or m.get("name")
            if mid not in seen:
                seen.add(mid)
                out.append(m)
        return out

    def _generate_quiz(self):
        topic = self.ent_quiz_topic.get().strip() or "General"
        qtype = self.cmb_quiz_type.get()
        n = int(self.cmb_quiz_n.get())
        for w in self.quiz_display.winfo_children():
            w.destroy()
        self.quiz_answers = []
        self.quiz_meta = {"topic": topic, "type": qtype, "n": n}

        mats = self._find_materials_for_topic(topic)
        has_academic_material = any(len((m.get("text") or "")) > 80 for m in mats)

        if mats and not has_academic_material:
            # material files exist but no extractable text
            has_academic_material = False

        if not has_academic_material and topic.lower() not in ("general", "career", ""):
            # Prompt: can only do career-oriented
            box = ctk.CTkFrame(self.quiz_display, fg_color="#181825", corner_radius=10)
            box.pack(fill="x", padx=12, pady=12)
            ctk.CTkLabel(
                box,
                text=("Sorry, I can't generate academic-related questions for this subject "
                      "(no matching Pre-Study Material found).\n"
                      "I can generate career-oriented questions instead."),
                text_color="#FAB387", justify="left", wraplength=700,
            ).pack(anchor="w", padx=12, pady=10)

            def _ok_career():
                for w in self.quiz_display.winfo_children():
                    w.destroy()
                self._render_quiz_questions(topic, qtype, n, mode="career", materials=[])

            ctk.CTkButton(box, text="OK — Generate Career Quiz", fg_color="#A6E3A1",
                          text_color="#11111B", command=_ok_career).pack(padx=12, pady=10)
            return

        mode = "academic+career" if has_academic_material else "career"
        self._render_quiz_questions(topic, qtype, n, mode=mode, materials=mats)

    def _render_quiz_questions(self, topic, qtype, n, mode="career", materials=None):
        materials = materials or []
        for w in self.quiz_display.winfo_children():
            w.destroy()
        self.quiz_answers = []
        self.quiz_meta = {"topic": topic, "type": qtype, "n": n, "mode": mode}

        label = f"{qtype} Quiz: {topic}  [{mode}]"
        ctk.CTkLabel(self.quiz_display, text=label,
                     font=ctk.CTkFont(size=15, weight="bold"), text_color="#CDD6F4").pack(
            anchor="w", padx=12, pady=10)

        # Build material-based stems when available
        import random as _rnd
        mat_sents = []
        for m in materials:
            import re
            parts = re.split(r"(?<=[.!?])\s+|\n+", m.get("text") or "")
            for p in parts:
                p = re.sub(r"\s+", " ", p).strip()
                if 40 <= len(p) <= 220:
                    mat_sents.append(p)
        _rnd.shuffle(mat_sents)

        if mode.startswith("academic") and mat_sents and qtype == "MCQ":
            for i in range(min(n, len(mat_sents))):
                sent = mat_sents[i]
                words = [w for w in sent.replace(",", " ").split() if len(w) > 3]
                key = words[min(2, len(words) - 1)] if words else topic
                correct = f"Supported by material: {sent[:110]}{'…' if len(sent) > 110 else ''}"
                wrongs = [
                    f"Career-only claim unrelated to {key}",
                    f"Contradicts the uploaded notes on {topic}",
                    f"Irrelevant industry slogan about {key}",
                ]
                opts = [correct] + wrongs
                _rnd.shuffle(opts)
                self._add_mcq_row(i, f"From your study material on {topic}, which is accurate about “{key}”?",
                                 opts, opts.index(correct))
            # fill remaining with career MCQs if needed
            for i in range(len(mat_sents), n):
                self._add_career_mcq(i, topic)
            ctk.CTkButton(self.quiz_display, text="Submit & Evaluate Mastery", fg_color="#89B4FA",
                          text_color="#11111B", font=ctk.CTkFont(weight="bold"),
                          command=self._evaluate_quiz).pack(padx=12, pady=14)
            return

        # Topic-aware question bank (simple templates)
        mcq_bank = {
            "default": [
                (f"Which layer of the OSI model is most related to {topic} concepts?",
                 ["Application", "Transport", "Network", "Physical"], 1),
                (f"A key protocol commonly discussed with {topic} is:",
                 ["HTTP only", "TCP/UDP", "SMTP only", "FTP only"], 1),
                (f"What is a primary goal when studying {topic}?",
                 ["Ignore standards", "Understand principles & trade-offs", "Memorize only dates", "Avoid diagrams"], 1),
                (f"In {topic}, congestion control is mainly associated with:",
                 ["Physical media", "TCP", "DNS only", "HTML"], 1),
                (f"Which statement about {topic} is most accurate?",
                 ["It has no real-world use", "It underpins modern distributed systems", "It is obsolete", "It needs no algorithms"], 1),
            ]
        }
        text_templates = {
            "Conceptual": [
                f"Explain the fundamental principles underlying {topic}.",
                f"How does {topic} relate to other core subjects in your curriculum?",
                f"Define the most important terms used in {topic}.",
                f"Compare two major approaches used in {topic}.",
                f"Why is {topic} important for your target career?",
            ],
            "Numerical": [
                f"Solve a representative numerical problem related to {topic} and show steps.",
                f"Given typical parameters in {topic}, estimate a key performance metric.",
                f"How would you calculate efficiency/overhead in a {topic} scenario?",
            ],
            "Scenario-based": [
                f"Describe a real-world failure scenario involving {topic} and how you would fix it.",
                f"You are asked to design a system using {topic}. Outline your approach.",
                f"Evaluate trade-offs when applying {topic} under resource constraints.",
            ],
            "Short Answer": [
                f"In 3-5 sentences, summarize {topic}.",
                f"List 4 key components of {topic}.",
                f"What common mistake do students make with {topic}?",
            ],
        }

        import random as _rq
        _rq.seed(int(time.time() * 1000) % 10_000_000)
        if qtype == "MCQ":
            bank = list(mcq_bank["default"])
            # expand bank with career + variant stems so each Generate differs
            bank += [
                (f"A practical lab skill tied to {topic} is:",
                 ["Blind memorization only", "Measuring and validating results", "Avoiding notes", "Skipping peer review"], 1),
                (f"When debugging a {topic} issue, first step is usually:",
                 ["Delete everything", "Reproduce & isolate the fault", "Ignore logs", "Change random settings"], 1),
                (f"Industry roles using {topic} often require:",
                 ["Only social media skills", "Domain vocabulary + problem solving", "No teamwork", "Zero documentation"], 1),
                (f"Which metric might you track in a {topic} project?",
                 ["Likes only", "Latency / accuracy / cost trade-offs", "Font size", "Wallpaper color"], 1),
            ]
            _rq.shuffle(bank)
            for i in range(n):
                q, opts, ans_idx = bank[i % len(bank)]
                # shuffle options while tracking correct index
                pairs = list(enumerate(opts))
                _rq.shuffle(pairs)
                new_opts = [p[1] for p in pairs]
                new_ans = [p[0] for p in pairs].index(ans_idx)
                self._add_mcq_row(i, q, new_opts, new_ans)
        else:
            templates = list(text_templates.get(qtype, text_templates["Conceptual"]))
            _rq.shuffle(templates)
            for i in range(n):
                q = templates[i % len(templates)]
                card = ctk.CTkFrame(self.quiz_display, fg_color="#11111B", corner_radius=8)
                card.pack(fill="x", padx=12, pady=6)
                ctk.CTkLabel(card, text=f"{i+1}. {q}", text_color="#89B4FA",
                             font=ctk.CTkFont(size=13), wraplength=700, justify="left").pack(anchor="w", padx=10, pady=(8, 4))
                tb = ctk.CTkTextbox(card, height=60, fg_color="#181825", border_color="#313244")
                tb.pack(fill="x", padx=10, pady=(0, 8))
                self.quiz_answers.append({"type": "text", "box": tb, "topic": topic})

        ctk.CTkButton(self.quiz_display, text="Submit & Auto-Evaluate Mastery",
                      fg_color="#89B4FA", text_color="#11111B", font=ctk.CTkFont(weight="bold"),
                      command=self._submit_quiz).pack(pady=14)

    def _add_mcq_row(self, i, question, opts, correct_idx):
        card = ctk.CTkFrame(self.quiz_display, fg_color="#11111B", corner_radius=8)
        card.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(card, text=f"{i+1}. {question}", text_color="#89B4FA",
                     font=ctk.CTkFont(size=13), wraplength=700, justify="left").pack(
            anchor="w", padx=10, pady=(8, 4))
        var = ctk.StringVar(value="")
        for j, opt in enumerate(opts):
            ctk.CTkRadioButton(card, text=f"{chr(65+j)}. {opt[:140]}", variable=var, value=str(j),
                               text_color="#CDD6F4").pack(anchor="w", padx=20, pady=2)
        self.quiz_answers.append({"type": "mcq", "var": var, "correct": correct_idx, "opts": opts})

    def _add_career_mcq(self, i, topic):
        career_qs = [
            (f"Which soft skill best supports a career path involving {topic}?",
             ["Communication & teamwork", "Ignoring deadlines", "Avoiding documentation", "Never asking questions"], 0),
            (f"In job interviews about {topic}, employers often expect:",
             ["Memorized slogans only", "Practical problem-solving examples", "No portfolio", "Zero domain vocabulary"], 1),
            (f"A career-oriented project using {topic} should emphasize:",
             ["Impact, metrics, and trade-offs", "Only UI colors", "Copying without understanding", "Skipping testing"], 0),
            (f"Which resume bullet is strongest for {topic} roles?",
             ["Worked hard sometimes", "Built X using Y; improved Z by N%", "Knows computers", "Attended classes"], 1),
        ]
        q, opts, ans = career_qs[i % len(career_qs)]
        self._add_mcq_row(i, q, opts, ans)

    def _submit_quiz(self):
        if not self.quiz_answers:
            return
        topic = self.quiz_meta.get("topic", "General")
        qtype = self.quiz_meta.get("type", "MCQ")
        correct = 0
        total = len(self.quiz_answers)
        feedback_lines = []

        for i, item in enumerate(self.quiz_answers):
            if item["type"] == "mcq":
                try:
                    chosen = int(item["var"].get()) if item["var"].get() != "" else -1
                except Exception:
                    chosen = -1
                ok = chosen == item["correct"]
                if ok:
                    correct += 1
                ans_letter = chr(65 + item["correct"])
                feedback_lines.append(f"Q{i+1}: {'✓ Correct' if ok else '✗ Wrong'} (Answer: {ans_letter})")
            else:
                text = item["box"].get("1.0", "end").strip()
                # simple heuristic scoring
                score_i = 0
                if len(text) >= 40:
                    score_i += 40
                if len(text) >= 80:
                    score_i += 20
                keywords = [w.lower() for w in topic.replace("-", " ").split() if len(w) > 2]
                hits = sum(1 for k in keywords if k in text.lower())
                score_i += min(40, hits * 15)
                if score_i >= 50:
                    correct += 1
                feedback_lines.append(f"Q{i+1}: score ~{score_i}% ({'OK' if score_i >= 50 else 'Weak'})")

        accuracy = (correct / total) * 100 if total else 0
        mastery, status = MasteryEngine.calculate_mastery(accuracy, 25, total)

        # save mastery
        tm = self.user_data.setdefault("topic_mastery", {})
        prev = tm.get(topic)
        if prev is not None:
            tm[topic] = round((prev + mastery) / 2, 1)
        else:
            tm[topic] = mastery

        hist = self.user_data.setdefault("quiz_history", [])
        hist.append({
            "topic": topic, "type": qtype, "score": mastery, "status": status,
            "accuracy": round(accuracy, 1), "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        # update course mastery if matching title
        for c in self.user_data.get("courses", []):
            if topic.lower() in c.get("title", "").lower() or c.get("title", "").lower() in topic.lower():
                c["mastery"] = tm[topic]

        db.save_user_profile(self.current_user_id, self.user_data)

        fb = ctk.CTkLabel(self.quiz_display,
                          text=f"✓ Assessment Recorded! Mastery: {mastery}% ({status})\n" + "\n".join(feedback_lines),
                          font=ctk.CTkFont(size=13), text_color="#A6E3A1", justify="left")
        fb.pack(pady=8, padx=12, anchor="w")
        self.refresh_ui()

    # ---------- FOCUS TIMER ----------
    def _build_timer(self):
        frame = ctk.CTkFrame(self.tab_timer, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        left = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12, width=360)
        left.pack(side="left", fill="y", padx=(0, 8), pady=6)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="Focus Plan Setup", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(left, text="Leave only if you leave the chair (not head turn) • 3× = Fail",
                     font=ctk.CTkFont(size=11), text_color="#A6ADC8").pack(anchor="w", padx=14, pady=(0, 8))

        ctk.CTkLabel(left, text="1. Select Subject:", text_color="#CDD6F4").pack(anchor="w", padx=14)
        self.cmb_timer_subject = ctk.CTkOptionMenu(left, values=["Custom Subject Entry"],
                                                   fg_color="#313244", button_color="#45475A")
        self.cmb_timer_subject.pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(left, text="2. Or Manual Subject:", text_color="#CDD6F4").pack(anchor="w", padx=14, pady=(6, 0))
        self.ent_timer_custom = ctk.CTkEntry(left, placeholder_text="e.g. TOC / OS",
                                             fg_color="#11111B", border_color="#313244")
        self.ent_timer_custom.pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(left, text="Session Duration:", text_color="#CDD6F4").pack(anchor="w", padx=14, pady=(6, 0))
        self.cmb_timer_dur = ctk.CTkOptionMenu(
            left,
            values=["5 Mins", "10 Mins", "15 Mins", "20 Mins", "25 Mins", "30 Mins", "45 Mins", "60 Mins", "90 Mins"],
            fg_color="#313244", button_color="#45475A")
        self.cmb_timer_dur.set("25 Mins")
        self.cmb_timer_dur.pack(fill="x", padx=14, pady=4)

        ctk.CTkButton(left, text="Add Subject to Queue", fg_color="#89B4FA", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._add_to_queue).pack(fill="x", padx=14, pady=6)
        ctk.CTkButton(left, text="Load from Smart Routine", fg_color="#CBA6F7", text_color="#11111B",
                      command=self._load_from_routine).pack(fill="x", padx=14, pady=2)
        ctk.CTkButton(left, text="Clear Queue", fg_color="#313244", text_color="#F38BA8",
                      command=self._clear_queue).pack(fill="x", padx=14, pady=2)

        self.lbl_queue = ctk.CTkLabel(left, text="Queue: 0 Subjects", text_color="#A6ADC8",
                                      justify="left", wraplength=320)
        self.lbl_queue.pack(anchor="w", padx=14, pady=10)

        ctk.CTkLabel(left, text="Manual: I left the table", text_color="#CDD6F4").pack(anchor="w", padx=14)
        ctk.CTkButton(left, text="Log Table Leave", fg_color="#FAB387", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._manual_leave).pack(fill="x", padx=14, pady=(4, 12))

        right = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12)
        right.pack(side="right", fill="both", expand=True, pady=6)

        ctk.CTkLabel(right, text="Smart Concentration Focus Tracker",
                     font=ctk.CTkFont(size=16, weight="bold"), text_color="#89B4FA").pack(pady=(12, 2))
        self.lbl_current_subject = ctk.CTkLabel(right, text="Subject: Idle",
                                                font=ctk.CTkFont(size=13, weight="bold"), text_color="#FAB387")
        self.lbl_current_subject.pack()
        self.lbl_timer = ctk.CTkLabel(right, text="00:00", font=ctk.CTkFont(size=52, weight="bold"),
                                      text_color="#CDD6F4")
        self.lbl_timer.pack(pady=4)
        self.lbl_conc = ctk.CTkLabel(right, text="Concentration: 0%  |  Leaves: 0",
                                     font=ctk.CTkFont(size=13), text_color="#A6ADC8")
        self.lbl_conc.pack(pady=2)

        self.lbl_cam = ctk.CTkLabel(right, text="[Webcam offline — Start Session]",
                                    fg_color="#11111B", corner_radius=10, width=560, height=240)
        self.lbl_cam.pack(pady=6)

        self.lbl_status = ctk.CTkLabel(right, text="Ready.", font=ctk.CTkFont(size=12, weight="bold"),
                                       text_color="#A6E3A1")
        self.lbl_status.pack(pady=2)

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.pack(pady=6)
        ctk.CTkButton(btn_row, text="▶ Start Session", width=140, fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._start_timer).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="⏸ Pause", width=110, fg_color="#FAB387", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._pause_timer).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="🔄 Full Reset", width=120, fg_color="#F38BA8", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._full_reset).pack(side="left", padx=6)

        ctk.CTkLabel(right, text="Session History (after complete)",
                     font=ctk.CTkFont(size=12, weight="bold"), text_color="#89B4FA").pack(anchor="w", padx=12, pady=(4, 0))
        self.session_history_host = ctk.CTkFrame(right, fg_color="#11111B", corner_radius=8, height=160)
        self.session_history_host.pack(fill="x", padx=10, pady=(2, 8))
        self.session_history_host.pack_propagate(False)
        ctk.CTkLabel(self.session_history_host, text="Finish a session to see focus vs distraction timeline.",
                     text_color="#A6ADC8", font=ctk.CTkFont(size=11)).pack(expand=True)

    def _add_to_queue(self):
        sel = self.cmb_timer_subject.get()
        custom = self.ent_timer_custom.get().strip()
        name = custom if custom else (sel if sel != "Custom Subject Entry" else "General Study")
        try:
            dur = int(self.cmb_timer_dur.get().split()[0])
        except Exception:
            dur = 25
        self.subject_queue.append({"subject": name, "duration": dur})
        self.ent_timer_custom.delete(0, "end")
        self._update_queue_lbl()

    def _load_from_routine(self):
        routine = self.user_data.get("study_routine", [])
        study = [r for r in routine if r.get("type") == "study"]
        if not study:
            self.lbl_status.configure(text="No study blocks in routine. Generate routine first.", text_color="#F38BA8")
            return
        self.subject_queue.clear()
        for r in study[:12]:
            self.subject_queue.append({
                "subject": r.get("subject") or r.get("activity", "Study"),
                "duration": max(20, min(90, int(r.get("duration_mins", 45))))
            })
        self.current_subject_idx = 0
        self._update_queue_lbl()
        self.lbl_status.configure(text=f"Loaded {len(self.subject_queue)} blocks from routine.", text_color="#A6E3A1")

    def _clear_queue(self):
        self.subject_queue.clear()
        self.current_subject_idx = 0
        self._full_reset()
        self._update_queue_lbl()

    def _update_queue_lbl(self):
        if not self.subject_queue:
            self.lbl_queue.configure(text="Queue: 0 Subjects")
            return
        lines = [f"{i+1}. {x['subject'][:40]} ({x['duration']}m)" for i, x in enumerate(self.subject_queue)]
        self.lbl_queue.configure(text="Queue:\n" + "\n".join(lines))

    def _update_conc_label(self):
        if not self.timer_running and self.timer_seconds <= 0 and self.table_leave_count == 0:
            self.lbl_conc.configure(text="Concentration: 0%  |  Leaves: 0", text_color="#A6ADC8")
            return
        color = "#A6E3A1" if self.concentration_percent >= 80 else ("#FAB387" if self.concentration_percent >= 60 else "#F38BA8")
        self.lbl_conc.configure(
            text=f"Concentration: {self.concentration_percent:.0f}%  |  Leaves: {self.table_leave_count}",
            text_color=color)

    def _manual_leave(self):
        if not self.timer_running:
            self.lbl_status.configure(text="Start a session first.", text_color="#FAB387")
            return
        self._register_leave()

    def _register_leave(self):
        if self.session_failed or not self.timer_running:
            return
        # Prevent double-count: same absence can only score once until user returns
        if not getattr(self, "_leave_armed", True):
            return
        self._leave_armed = False
        self._face_absent_streak = 0
        self.table_leave_count += 1
        if self.table_leave_count == 1:
            self.concentration_percent = max(85.0, self.concentration_percent - 10)
            self.lbl_status.configure(text="Leave 1/3 (−10%) — return to desk", text_color="#FAB387")
        elif self.table_leave_count == 2:
            self.concentration_percent = max(70.0, self.concentration_percent - 15)
            self.lbl_status.configure(text="Leave 2/3 — one more = FAIL", text_color="#FAB387")
        else:
            self.concentration_percent = max(40.0, self.concentration_percent - 25)
            self.session_failed = True
            self.timer_running = False
            self.lbl_status.configure(text="SESSION FAILED — 3 leaves", text_color="#F38BA8")
            self._release_camera()
            self._save_failed_session()
        self._update_conc_label()

    def _release_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _open_camera(self):
        self._release_camera()
        if not ensure_cv2():
            self.lbl_cam.configure(text="[OpenCV missing — pip install opencv-python]\nTimer + Log Leave still work.", image="")
            return False
        if self.face_cascade is None and self.yunet_detector is None:
            self._load_face_models()
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for backend in backends:
            for idx in (0, 1):
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap is not None and cap.isOpened():
                        try:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        ret, _ = cap.read()
                        if ret:
                            self.cap = cap
                            return True
                        cap.release()
                except Exception:
                    pass
        self.lbl_cam.configure(text="[Webcam not found / permission denied]", image="")
        return False

    def _start_timer(self):
        if not self.subject_queue:
            self.lbl_status.configure(text="Add at least 1 subject to queue!", text_color="#F38BA8")
            return
        if self.timer_running:
            return
        if self.current_subject_idx >= len(self.subject_queue):
            self.current_subject_idx = 0
        self.timer_running = True
        self.session_failed = False
        if self.timer_seconds <= 0:
            self.table_leave_count = 0
            self._face_absent_streak = 0
            self._eye_absent_streak = 0
            self._leave_armed = True
            self.phone_detection_counter = 0.0
            self.concentration_percent = 100.0
            self.session_timeline = []
            self._session_elapsed = 0
            task = self.subject_queue[self.current_subject_idx]
            self.timer_seconds = int(task["duration"]) * 60
            self.lbl_current_subject.configure(
                text=f"Subject: {task['subject'][:42]} ({self.current_subject_idx+1}/{len(self.subject_queue)})")
        self._update_conc_label()
        cam_ok = self._open_camera()
        self.lbl_status.configure(
            text="Focus Active — stay at desk" if cam_ok else "Focus Active — camera off (use Log Leave)",
            text_color="#A6E3A1" if cam_ok else "#FAB387")
        self._timer_tick()
        self._video_tick()

    def _pause_timer(self):
        if not self.timer_running and self.timer_seconds <= 0:
            return
        self.timer_running = False
        self._release_camera()
        self.lbl_cam.configure(text="[Paused — press Start to resume]", image="")
        m, s = divmod(max(0, self.timer_seconds), 60)
        self.lbl_timer.configure(text=f"{m:02d}:{s:02d}")
        self._update_conc_label()
        self.lbl_status.configure(text="Paused", text_color="#FAB387")

    def _full_reset(self):
        self.timer_running = False
        self.session_failed = False
        self.timer_seconds = 0
        self.table_leave_count = 0
        self._face_absent_streak = 0
        self._eye_absent_streak = 0
        self._leave_armed = True
        self.phone_detection_counter = 0.0
        self.concentration_percent = 0.0
        self.current_subject_idx = 0
        self._release_camera()
        self.lbl_cam.configure(text="[Webcam offline — Start Session]", image="")
        self.lbl_timer.configure(text="00:00")
        self.lbl_current_subject.configure(text="Subject: Idle")
        self._update_conc_label()
        self.lbl_status.configure(text="Fully reset.", text_color="#A6E3A1")

    def _timer_tick(self):
        if not self.timer_running:
            return
        if self.timer_seconds > 0:
            m, s = divmod(self.timer_seconds, 60)
            self.lbl_timer.configure(text=f"{m:02d}:{s:02d}")
            # sample concentration every second for history graph
            self._session_elapsed = getattr(self, "_session_elapsed", 0) + 1
            mode = "focused" if self.concentration_percent >= 80 else (
                "distracted" if self.concentration_percent < 60 else "moderate")
            self.session_timeline.append({
                "t_sec": self._session_elapsed,
                "conc": round(self.concentration_percent, 1),
                "mode": mode,
                "leaves": self.table_leave_count,
            })
            self.timer_seconds -= 1
            self.after(1000, self._timer_tick)
        else:
            self._on_subject_complete()

    def _show_session_history_graph(self, timeline, subject=""):
        if not hasattr(self, "session_history_host"):
            return
        for w in self.session_history_host.winfo_children():
            w.destroy()
        if not timeline:
            ctk.CTkLabel(self.session_history_host, text="No timeline data.",
                         text_color="#A6ADC8").pack(expand=True)
            return
        fig, ax = plt.subplots(figsize=(5.6, 1.5), dpi=100)
        fig.patch.set_facecolor("#11111B")
        ax.set_facecolor("#11111B")
        xs = [p["t_sec"] / 60 for p in timeline]
        ys = [p["conc"] for p in timeline]
        ax.plot(xs, ys, color="#89B4FA", linewidth=1.8)
        ax.fill_between(xs, ys, alpha=0.3, color="#89B4FA")
        # mark low focus zones
        ax.axhline(80, color="#A6E3A1", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(60, color="#F38BA8", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(0, 105)
        ax.set_xlim(left=0)
        ax.set_xlabel("min", color="#A6ADC8", fontsize=8)
        ax.set_ylabel("%", color="#A6ADC8", fontsize=8)
        ax.set_title(f"Focus timeline: {subject[:36]}", color="#CDD6F4", fontsize=9)
        ax.tick_params(colors="#A6ADC8", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#313244")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.session_history_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_subject_complete(self):
        if self.session_failed or self.current_subject_idx >= len(self.subject_queue):
            return
        task = self.subject_queue[self.current_subject_idx]
        phone_penalty = min(20, self.phone_detection_counter * 2)
        final_conc = max(40.0, self.concentration_percent - phone_penalty)
        completion = 100 if final_conc >= 70 else (80 if final_conc >= 50 else 50)
        timeline_copy = list(self.session_timeline)
        self.user_data.setdefault("study_sessions", []).append({
            "subject": task["subject"], "duration_minutes": task["duration"],
            "distraction_score": round(self.phone_detection_counter + self.table_leave_count * 2, 1),
            "completion_rate": completion, "concentration_percent": round(final_conc, 1),
            "table_leaves": self.table_leave_count,
            "timeline": timeline_copy,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        db.save_user_profile(self.current_user_id, self.user_data)
        self._show_session_history_graph(timeline_copy, task["subject"])
        # Instant 5-mark MCQ on this subject for authenticity judgment
        try:
            self._launch_post_session_quiz(
                subject=task["subject"],
                duration_mins=task["duration"],
                final_conc=final_conc,
                timeline=timeline_copy,
                leaves=self.table_leave_count,
            )
        except Exception as _e:
            print("post-session quiz error:", _e)
        self.current_subject_idx += 1
        if self.current_subject_idx < len(self.subject_queue):
            self.table_leave_count = 0
            self._face_absent_streak = 0
            self._leave_armed = True
            self.phone_detection_counter = 0.0
            self.concentration_percent = 100.0
            self.session_failed = False
            self.session_timeline = []
            self._session_elapsed = 0
            self._update_conc_label()
            nxt = self.subject_queue[self.current_subject_idx]
            self.timer_seconds = int(nxt["duration"]) * 60
            self.lbl_current_subject.configure(
                text=f"Subject: {nxt['subject'][:42]} ({self.current_subject_idx+1}/{len(self.subject_queue)})")
            self._timer_tick()
        else:
            self.timer_running = False
            self.timer_seconds = 0
            self.concentration_percent = 0.0
            self.table_leave_count = 0
            self._release_camera()
            self.lbl_cam.configure(text="[All sessions done]", image="")
            self.lbl_timer.configure(text="00:00")
            self._update_conc_label()
            self.lbl_status.configure(text="All queue sessions completed!", text_color="#A6E3A1")
            self.refresh_ui()
            self._render_dash_focus_chart()

    def _save_failed_session(self):
        if self.current_subject_idx >= len(self.subject_queue):
            return
        task = self.subject_queue[self.current_subject_idx]
        timeline_copy = list(getattr(self, "session_timeline", []))
        self.user_data.setdefault("study_sessions", []).append({
            "subject": task["subject"], "duration_minutes": task["duration"],
            "distraction_score": round(self.phone_detection_counter + self.table_leave_count * 3, 1),
            "completion_rate": 0, "concentration_percent": round(self.concentration_percent, 1),
            "table_leaves": self.table_leave_count, "failed": True,
            "timeline": timeline_copy,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        db.save_user_profile(self.current_user_id, self.user_data)
        self._show_session_history_graph(timeline_copy, task["subject"] + " (failed)")
        self.refresh_ui()

    def _detect_eyes_in_face(self, gray, face_box):
        """Return list of eye boxes in full-frame coords inside the face ROI."""
        x, y, fw, fh = face_box
        # Eyes live in the upper ~60% of the face box
        roi_y2 = y + max(10, int(fh * 0.62))
        roi = gray[y:roi_y2, x:x + fw]
        if roi.size == 0:
            return []
        min_e = max(8, int(min(fw, fh) * 0.08))
        max_e = max(min_e + 4, int(min(fw, fh) * 0.45))
        eyes = []
        for cascade in (self.eye_cascade, self.eye_cascade_glasses):
            if cascade is None:
                continue
            try:
                found = cascade.detectMultiScale(
                    roi, scaleFactor=1.08, minNeighbors=3,
                    minSize=(min_e, min_e), maxSize=(max_e, max_e)
                )
                for (ex, ey, ew, eh) in found:
                    eyes.append((x + ex, y + ey, ew, eh))
            except Exception:
                pass
        # Keep at most 2 largest (left/right eye)
        eyes = sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[:2]
        return eyes

    def _detect_presence(self, frame_bgr, gray):
        """
        Returns (face_boxes, eye_boxes, mode, present).
        present=True ONLY if a real face is detected (or strong torso motion in center).
        Static windows/walls must NOT count as "at desk".
        """
        h, w = gray.shape[:2]
        face_boxes, eye_boxes = [], []

        try:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            ge = clahe.apply(gray)
        except Exception:
            ge = cv2.equalizeHist(gray)

        min_s = max(28, int(min(w, h) * 0.05))
        if self.face_cascade is not None or self.face_cascade_alt is not None:
            for cascade, scale, neigh in [
                (self.face_cascade, 1.07, 3),
                (self.face_cascade_alt, 1.08, 3),
                (self.profile_cascade, 1.08, 3),
            ]:
                if cascade is None:
                    continue
                try:
                    found = list(cascade.detectMultiScale(
                        ge, scaleFactor=scale, minNeighbors=neigh, minSize=(min_s, min_s)))
                    if found:
                        face_boxes = found
                        break
                except Exception:
                    pass

        if not face_boxes and self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                _, faces = self.yunet_detector.detect(frame_bgr)
                if faces is not None:
                    for f in faces:
                        x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        conf = float(f[4]) if len(f) > 4 else 1.0
                        if conf >= 0.45:
                            face_boxes.append((x, y, fw, fh))
            except Exception:
                pass

        # --- Real face found → present (focused or study by eyes)
        if face_boxes:
            main_face = max(face_boxes, key=lambda b: b[2] * b[3])
            eye_boxes = self._detect_eyes_in_face(ge, main_face)
            mode = "focused" if eye_boxes else "study"
            # update motion baseline while present
            self._prev_gray = gray.copy()
            return face_boxes, eye_boxes, mode, True

        # --- No face: only "study" if strong MOTION in center person zone
        # (ignores static bright windows / empty room texture)
        cx1, cx2 = int(w * 0.25), int(w * 0.75)
        cy1, cy2 = int(h * 0.15), int(h * 0.75)
        roi = gray[cy1:cy2, cx1:cx2]
        motion = 0.0
        if roi.size and self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            prev_roi = self._prev_gray[cy1:cy2, cx1:cx2]
            # pixel change ratio (more robust than mean absdiff alone)
            diff = cv2.absdiff(roi, prev_roi)
            motion = float(np.mean(diff))
            changed = float(np.count_nonzero(diff > 18)) / float(diff.size)
        else:
            changed = 0.0
        self._prev_gray = gray.copy()

        # Need clear movement (person shifting/writing), not window noise
        person_motion = motion > 6.5 and changed > 0.04
        if person_motion:
            bw, bh = int(w * 0.28), int(h * 0.35)
            box = [(w // 2 - bw // 2, int(h * 0.28) - bh // 2, bw, bh)]
            return box, [], "study", True

        # No face + no person motion → AWAY (leave will count)
        return [], [], "absent", False

    def _video_tick(self):
        if not self.timer_running:
            return
        # Keep camera alive while user multitasks (PDF / browser) — re-open if frames fail
        if HAS_CV:
            if self.cap is None or not self.cap.isOpened():
                self._cam_fail_streak = getattr(self, "_cam_fail_streak", 0) + 1
                if self._cam_fail_streak >= 8:
                    self._open_camera()
                    self._cam_fail_streak = 0
            else:
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    ret, frame = False, None
                if not ret or frame is None:
                    self._cam_fail_streak = getattr(self, "_cam_fail_streak", 0) + 1
                    if self._cam_fail_streak >= 10:
                        try:
                            self._release_camera()
                        except Exception:
                            pass
                        self._open_camera()
                        self._cam_fail_streak = 0
                    self.after(80, self._video_tick)
                    return
                self._cam_fail_streak = 0
                # keep preview smaller so UI stays responsive beside PDF windows
                frame = cv2.resize(frame, (480, 270))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_boxes, eye_boxes, mode, present = self._detect_presence(frame, gray)
                self._last_eye_boxes = eye_boxes

                # Draw face
                for (x, y, fw, fh) in face_boxes:
                    if mode == "focused":
                        col, label = (166, 227, 161), "FACE+EYES"
                    elif mode == "study":
                        col, label = (137, 180, 250), "STUDY"
                    else:
                        col, label = (243, 139, 168), "AWAY"
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), col, 2)
                    cv2.putText(frame, label, (x, max(18, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                # Draw eyes (cyan)
                for (ex, ey, ew, eh) in eye_boxes:
                    cx, cy = ex + ew // 2, ey + eh // 2
                    cv2.circle(frame, (cx, cy), max(4, ew // 3), (94, 234, 212), 2)
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (94, 234, 212), 1)

                # ~3.5s continuous absence — head turn alone should not count as leave
                LEAVE_FRAMES = 44

                if present:
                    self._face_absent_streak = 0
                    self._leave_armed = True  # back at desk → next absence can count
                    if mode == "focused":
                        self._eye_absent_streak = 0
                        self.concentration_percent = min(100.0, self.concentration_percent + 0.12)
                    else:
                        self._eye_absent_streak += 1
                        if self.concentration_percent < 92:
                            self.concentration_percent = min(92.0, self.concentration_percent + 0.03)
                        elif self.concentration_percent > 92:
                            self.concentration_percent = max(88.0, self.concentration_percent - 0.02)
                    self._update_conc_label()
                else:
                    self._face_absent_streak += 1
                    self._eye_absent_streak += 1
                    self.phone_detection_counter += 0.04
                    self.concentration_percent = max(35.0, self.concentration_percent - 0.25)
                    self._update_conc_label()
                    # Only fire leave if armed (user was present since last leave)
                    if self._leave_armed and self._face_absent_streak >= LEAVE_FRAMES:
                        self._register_leave()

                eye_txt = f"Eyes:{len(eye_boxes)}"
                cv2.putText(frame,
                            f"Leaves:{self.table_leave_count} Conc:{self.concentration_percent:.0f}% {eye_txt}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (137, 180, 250), 2)
                if not present:
                    left_in = max(0, LEAVE_FRAMES - self._face_absent_streak)
                    sec = left_in * 0.08
                    cv2.putText(frame, f"AWAY! Leave in {sec:.1f}s  (Leaves:{self.table_leave_count})",
                                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (243, 139, 168), 2)
                elif mode == "focused":
                    cv2.putText(frame, "Face+Eyes OK - focused", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (166, 227, 161), 1)
                else:
                    cv2.putText(frame, "Face OK - study mode (book/PDF)", (10, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (137, 180, 250), 1)

                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.lbl_cam.configure(image=imgtk, text="")
                self.lbl_cam.image = imgtk
        self.after(80, self._video_tick)

    # ---------- TASK MANAGER ----------
    def _build_tasks(self):
        frame = ctk.CTkScrollableFrame(self.tab_tasks, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        ctk.CTkLabel(frame, text="Task Manager", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#CDD6F4").pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(frame, text="Tasks + Pre-Study Materials (PDF/slides/notes) → material-based exams",
                     font=ctk.CTkFont(size=11), text_color="#A6ADC8").pack(anchor="w", pady=(0, 10))

        form = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12)
        form.pack(fill="x", pady=(0, 10))
        row = ctk.CTkFrame(form, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=12)
        self.ent_task_title = ctk.CTkEntry(row, placeholder_text="Task title e.g. Finish Chapter 3 PDF",
                                           width=320, fg_color="#11111B", border_color="#313244")
        self.ent_task_title.pack(side="left", padx=(0, 8))
        self.ent_task_date = ctk.CTkEntry(row, placeholder_text="Date YYYY-MM-DD", width=140,
                                          fg_color="#11111B", border_color="#313244")
        self.ent_task_date.pack(side="left", padx=(0, 8))
        self.ent_task_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        ctk.CTkButton(row, text="Add Task", fg_color="#89B4FA", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), width=100,
                      command=self._add_task).pack(side="left")

        self.lbl_task_pct = ctk.CTkLabel(frame, text="Completion: 0% (0/0)",
                                        font=ctk.CTkFont(size=14, weight="bold"), text_color="#A6E3A1")
        self.lbl_task_pct.pack(anchor="w", pady=(0, 8))

        self.task_list_host = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12)
        self.task_list_host.pack(fill="x", pady=(0, 12))
        self._refresh_task_list()

        # ---- Pre-Study Materials ----
        ctk.CTkLabel(frame, text="Pre-Study Materials", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", pady=(8, 4))
        ctk.CTkLabel(frame, text="Add PDF / slides / notes here. Adaptive Quiz Engine uses them for academic+career exams.",
                     font=ctk.CTkFont(size=11), text_color="#A6ADC8").pack(anchor="w", pady=(0, 6))
        mat_row = ctk.CTkFrame(frame, fg_color="transparent")
        mat_row.pack(fill="x", pady=4)
        ctk.CTkButton(mat_row, text="Add PDF / Slide / TXT", fg_color="#CBA6F7", text_color="#11111B",
                      command=self._add_study_material).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(mat_row, text="→ Use Adaptive Quiz Engine to exam from these materials",
                     text_color="#A6ADC8", font=ctk.CTkFont(size=11)).pack(side="left", padx=8)
        self.mat_list_host = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12)
        self.mat_list_host.pack(fill="x", pady=8)
        self._refresh_materials_list()

    def _add_task(self):
        title = self.ent_task_title.get().strip()
        date = self.ent_task_date.get().strip() or datetime.now().strftime("%Y-%m-%d")
        if not title:
            return
        tasks = self.user_data.setdefault("tasks", [])
        tasks.append({
            "id": f"t{int(time.time()*1000)}",
            "title": title,
            "date": date,
            "done": False,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        db.save_user_profile(self.current_user_id, self.user_data)
        self.ent_task_title.delete(0, "end")
        self._refresh_task_list()
        self.refresh_ui()

    def _toggle_task(self, task_id):
        for t in self.user_data.get("tasks", []):
            if t.get("id") == task_id:
                t["done"] = not t.get("done", False)
                break
        db.save_user_profile(self.current_user_id, self.user_data)
        self._refresh_task_list()
        self.refresh_ui()

    def _delete_task(self, task_id):
        self.user_data["tasks"] = [t for t in self.user_data.get("tasks", []) if t.get("id") != task_id]
        db.save_user_profile(self.current_user_id, self.user_data)
        self._refresh_task_list()
        self.refresh_ui()

    def _refresh_task_list(self):
        if not hasattr(self, "task_list_host"):
            return
        for w in self.task_list_host.winfo_children():
            w.destroy()
        tasks = self.user_data.get("tasks", [])
        done = sum(1 for t in tasks if t.get("done"))
        total = len(tasks)
        pct = round(100.0 * done / total, 1) if total else 0.0
        if hasattr(self, "lbl_task_pct"):
            self.lbl_task_pct.configure(text=f"Completion: {pct}% ({done}/{total})")
        if not tasks:
            ctk.CTkLabel(self.task_list_host, text="No tasks yet. Add one above.",
                         text_color="#A6ADC8").pack(padx=12, pady=16)
            return
        for t in tasks:
            row = ctk.CTkFrame(self.task_list_host, fg_color="#11111B", corner_radius=8)
            row.pack(fill="x", padx=10, pady=4)
            done_flag = bool(t.get("done"))
            title_color = "#A6E3A1" if done_flag else "#CDD6F4"
            mark = "✓ " if done_flag else "○ "
            ctk.CTkLabel(row, text=f"{mark}{t.get('title', '')}", text_color=title_color,
                         font=ctk.CTkFont(size=13)).pack(side="left", padx=10, pady=8)
            ctk.CTkLabel(row, text=t.get("date", ""), text_color="#A6ADC8",
                         font=ctk.CTkFont(size=11)).pack(side="left", padx=6)
            ctk.CTkButton(row, text="Done" if not done_flag else "Undo", width=70, height=28,
                          fg_color="#A6E3A1" if not done_flag else "#313244",
                          text_color="#11111B" if not done_flag else "#CDD6F4",
                          command=lambda i=t.get("id"): self._toggle_task(i)).pack(side="right", padx=6, pady=6)
            ctk.CTkButton(row, text="✕", width=32, height=28, fg_color="#F38BA8", text_color="#11111B",
                          command=lambda i=t.get("id"): self._delete_task(i)).pack(side="right", padx=(0, 4), pady=6)

    def _extract_text_from_file(self, path):
        """Extract readable text from PDF / TXT for material-based questions."""
        ext = os.path.splitext(path)[1].lower()
        text = ""
        try:
            if ext == ".txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            elif ext == ".pdf":
                try:
                    import PyPDF2
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        parts = []
                        for page in reader.pages[:30]:
                            parts.append(page.extract_text() or "")
                        text = "\n".join(parts)
                except Exception:
                    text = f"[PDF attached: {os.path.basename(path)} — install PyPDF2 for text extract]"
            else:
                # pptx/docx: store path only; use filename as topic seed
                text = f"Study material: {os.path.basename(path)}"
        except Exception as e:
            text = f"[Could not read file: {e}]"
        return (text or "")[:12000]

    def _add_study_material(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select study material",
            filetypes=[
                ("Documents", "*.pdf;*.txt;*.pptx;*.docx"),
                ("PDF", "*.pdf"), ("Text", "*.txt"), ("All", "*.*"),
            ],
        )
        if not path:
            return
        extracted = self._extract_text_from_file(path)
        mats = self.user_data.setdefault("study_materials", [])
        mats.append({
            "id": f"m{int(time.time()*1000)}",
            "name": os.path.basename(path),
            "path": path,
            "text": extracted,
            "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        db.save_user_profile(self.current_user_id, self.user_data)
        self._refresh_materials_list()

    def _delete_material(self, mid):
        self.user_data["study_materials"] = [
            m for m in self.user_data.get("study_materials", []) if m.get("id") != mid
        ]
        db.save_user_profile(self.current_user_id, self.user_data)
        self._refresh_materials_list()

    def _refresh_materials_list(self):
        if not hasattr(self, "mat_list_host"):
            return
        for w in self.mat_list_host.winfo_children():
            w.destroy()
        mats = self.user_data.get("study_materials", [])
        if not mats:
            ctk.CTkLabel(self.mat_list_host, text="No materials yet. Add PDF/notes above.",
                         text_color="#A6ADC8").pack(padx=12, pady=12)
            return
        for m in mats:
            row = ctk.CTkFrame(self.mat_list_host, fg_color="#11111B", corner_radius=8)
            row.pack(fill="x", padx=10, pady=4)
            preview_len = len(m.get("text") or "")
            ctk.CTkLabel(row, text=f"📄 {m.get('name', '')}  ({preview_len} chars)  ·  {m.get('added', '')}",
                         text_color="#CDD6F4", font=ctk.CTkFont(size=12)).pack(side="left", padx=10, pady=8)
            ctk.CTkButton(row, text="✕", width=32, height=28, fg_color="#F38BA8", text_color="#11111B",
                          command=lambda i=m.get("id"): self._delete_material(i)).pack(side="right", padx=8, pady=6)

    def _sentences_from_materials(self):
        """Build unique content sentences from uploaded materials."""
        import re
        blobs = []
        for m in self.user_data.get("study_materials", []):
            blobs.append(m.get("text") or m.get("name") or "")
        raw = "\n".join(blobs)
        # split into sentences
        parts = re.split(r"(?<=[.!?])\s+|\n+", raw)
        sents = []
        seen = set()
        for p in parts:
            p = re.sub(r"\s+", " ", p).strip()
            if len(p) < 40 or len(p) > 280:
                continue
            key = p[:60].lower()
            if key in seen:
                continue
            seen.add(key)
            sents.append(p)
        return sents

    def _exam_from_materials(self):
        """Generate academic/career style MCQs from material text — reshuffles each run."""
        import random
        import hashlib
        for w in self.mat_exam_host.winfo_children():
            w.destroy()
        sents = self._sentences_from_materials()
        if len(sents) < 3:
            ctk.CTkLabel(self.mat_exam_host,
                         text="Add more PDF/notes with readable text (need ≥3 sentences).",
                         text_color="#F38BA8").pack(pady=16)
            return

        # seed changes each exam so question set differs
        seed = int(time.time()) ^ len(sents)
        rng = random.Random(seed)
        pool = list(sents)
        rng.shuffle(pool)
        n = min(8, len(pool))
        chosen = pool[:n]

        questions = []
        for i, sent in enumerate(chosen):
            words = [w for w in sent.replace(",", " ").split() if len(w) > 3]
            key = words[min(2, len(words) - 1)] if words else "concept"
            # distractors from other sentences
            others = [s for s in pool if s != sent]
            rng.shuffle(others)
            wrongs = []
            for o in others[:6]:
                ow = [w for w in o.split() if len(w) > 4]
                if ow:
                    wrongs.append(" ".join(ow[:6]))
                if len(wrongs) >= 3:
                    break
            while len(wrongs) < 3:
                wrongs.append(f"Unrelated statement about {key} #{len(wrongs)}")
            correct = f"It states: {sent[:120]}{'…' if len(sent) > 120 else ''}"
            opts = [correct] + [f"Incorrect claim: {w[:100]}" for w in wrongs[:3]]
            rng.shuffle(opts)
            letters = ["A", "B", "C", "D"]
            correct_letter = letters[opts.index(correct)]
            qid = hashlib.md5(f"{seed}-{i}-{sent[:40]}".encode()).hexdigest()[:8]
            questions.append({
                "id": qid,
                "text": f"{i+1}. Based on your study material, which is accurate regarding “{key}”?",
                "options": [f"{letters[j]}) {opts[j]}" for j in range(4)],
                "correct": correct_letter,
                "source": sent[:80],
            })

        self._mat_exam_qs = questions
        self._mat_exam_vars = []
        self._mat_exam_seed = seed

        ctk.CTkLabel(self.mat_exam_host,
                     text=f"Material Exam · {len(questions)} Q · seed {seed} (changes each Generate)",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color="#CDD6F4").pack(anchor="w", padx=12, pady=8)

        for q in questions:
            card = ctk.CTkFrame(self.mat_exam_host, fg_color="#11111B", corner_radius=8)
            card.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(card, text=q["text"], text_color="#89B4FA", wraplength=700,
                         justify="left").pack(anchor="w", padx=10, pady=(8, 4))
            var = ctk.StringVar(value="")
            self._mat_exam_vars.append(var)
            for opt in q["options"]:
                ctk.CTkRadioButton(card, text=opt[:160], variable=var, value=opt[0],
                                   text_color="#CDD6F4", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=14)

        ctk.CTkButton(self.mat_exam_host, text="Submit Material Exam", fg_color="#89B4FA", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._submit_material_exam).pack(pady=12)

    def _submit_material_exam(self):
        qs = getattr(self, "_mat_exam_qs", [])
        vars_ = getattr(self, "_mat_exam_vars", [])
        if not qs:
            return
        scores = []
        for i, q in enumerate(qs):
            chosen = vars_[i].get().strip().upper() if i < len(vars_) else ""
            scores.append(100.0 if chosen == q.get("correct") else 0.0)
        avg = float(np.mean(scores)) if scores else 0.0
        topic = "MaterialExam"
        mats = self.user_data.get("study_materials", [])
        if mats:
            topic = f"Material:{mats[-1].get('name', 'Study')[:24]}"
        self.user_data.setdefault("topic_mastery", {})[topic] = round(avg, 1)
        self.user_data.setdefault("quiz_history", []).append({
            "topic": topic, "type": "Material-MCQ", "score": round(avg, 1),
            "status": "Strong" if avg >= 75 else ("Developing" if avg >= 50 else "Weak"),
            "count": len(qs), "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        db.save_user_profile(self.current_user_id, self.user_data)
        ctk.CTkLabel(self.mat_exam_host,
                     text=f"Score: {avg:.0f}%  — saved to mastery & quiz history",
                     text_color="#A6E3A1", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=8)
        self.refresh_ui()

    # ---------- BRAIN TESTING ----------
    def _build_brain(self):
        frame = ctk.CTkScrollableFrame(self.tab_brain, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        ctk.CTkLabel(frame, text="Brain Testing", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#CDD6F4").pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(frame, text="10 cognitive games · scores → Habit Health + Cognitive Intelligence Profile",
                     font=ctk.CTkFont(size=11), text_color="#A6ADC8").pack(anchor="w", pady=(0, 10))

        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="x")
        games = [
            ("Chess", "Strategy · planning\nBook + αβ engine", self._open_chess),
            ("Sudoku", "Logic · accuracy\nTime + mistakes", self._open_sudoku),
            ("Stop Object", "Reaction time\nClick red ball", self._open_stop_object),
            ("Minesweeper", "Risk · deduction\nFlags + time", self._open_minesweeper),
            ("2048", "Planning · patterns\nScore + moves", self._open_2048),
            ("N-Queens", "Constraint solve\nCS classic", self._open_nqueens),
            ("Tower Hanoi", "Recursion · plan\nMin moves", self._open_hanoi),
            ("Mastermind", "Hypothesis test\nCode break", self._open_mastermind),
            ("Connect Four", "Prediction\nWin/loss + moves", self._open_connect4),
            ("Nonogram", "Visual logic\nRow/col clues", self._open_nonogram),
        ]
        for i, (title, desc, cmd) in enumerate(games):
            r, c = divmod(i, 5)
            card = ctk.CTkFrame(grid, fg_color="#181825", corner_radius=12, width=200, height=130)
            card.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            card.grid_propagate(False)
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=13, weight="bold"),
                         text_color="#89B4FA").pack(pady=(12, 2))
            ctk.CTkLabel(card, text=desc, text_color="#A6ADC8", font=ctk.CTkFont(size=10)).pack()
            ctk.CTkButton(card, text="Play", fg_color="#A6E3A1", text_color="#11111B", width=90,
                          command=cmd).pack(pady=8)
        for c in range(5):
            grid.grid_columnconfigure(c, weight=1)

        self.lbl_cognitive = ctk.CTkLabel(frame, text="", text_color="#CDD6F4", justify="left")
        self.lbl_cognitive.pack(anchor="w", pady=(8, 4))

        self.lbl_brain_hist = ctk.CTkLabel(frame, text="", text_color="#CDD6F4", justify="left")
        self.lbl_brain_hist.pack(anchor="w", pady=12)
        ctk.CTkButton(frame, text="Clear Game History", fg_color="#313244", hover_color="#F38BA8",
                      text_color="#CDD6F4", command=self._clear_brain_history).pack(anchor="w", pady=4)
        self._refresh_brain_hist()

    def _refresh_brain_hist(self):
        if not hasattr(self, "lbl_brain_hist"):
            return
        scores = self.user_data.get("brain_scores", [])[-10:]
        if not scores:
            self.lbl_brain_hist.configure(text="No brain games played yet.")
            return
        # Reaction-time metrics from StopObject details (e.g. "0.85s")
        rts = []
        for s in self.user_data.get("brain_scores", []):
            if s.get("game") == "StopObject":
                d = str(s.get("detail", ""))
                try:
                    if d.endswith("s") and "ms" not in d:
                        rts.append(float(d.replace("s", "")))
                    elif d.endswith("ms"):
                        rts.append(float(d.replace("ms", "")) / 1000.0)
                except Exception:
                    pass
        lines = [f"• {s.get('game')} Lv{s.get('level',1)} → {s.get('score')}%  "
                 f"({s.get('detail', '')})  {s.get('date', '')}" for s in reversed(scores)]
        if rts:
            avg_rt = sum(rts) / len(rts)
            best_rt = min(rts)
            lines.append(f"\nReaction metrics: avg {avg_rt:.2f}s · best {best_rt:.2f}s · n={len(rts)}")
        self.lbl_brain_hist.configure(text="Recent brain scores:\n" + "\n".join(lines))
        if hasattr(self, "lbl_cognitive"):
            self.lbl_cognitive.configure(text=self._get_cognitive_summary())

    def _save_brain_score(self, game, level, score, detail="", metrics=None):
        """Save game result + optional cognitive metrics (mistakes, time, hints…)."""
        entry = {
            "game": game, "level": level, "score": round(float(score), 1),
            "detail": detail, "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        if metrics and isinstance(metrics, dict):
            entry["metrics"] = metrics
        self.user_data.setdefault("brain_scores", []).append(entry)
        # Update rolling cognitive intelligence profile
        self._update_cognitive_profile(game, float(score), metrics or {})
        db.save_user_profile(self.current_user_id, self.user_data)
        self._refresh_brain_hist()
        self.refresh_ui()

    def _update_cognitive_profile(self, game, score, metrics):
        """Map game performance → Cognitive Intelligence axes (0–100)."""
        cog = self.user_data.setdefault("cognitive_intelligence", {
            "logical_reasoning": 50.0,
            "pattern_recognition": 50.0,
            "problem_solving": 50.0,
            "decision_efficiency": 50.0,
            "spatial_reasoning": 50.0,
            "strategic_thinking": 50.0,
        })
        # game → primary axes weights
        mapping = {
            "Sudoku": {"logical_reasoning": 0.5, "pattern_recognition": 0.3, "problem_solving": 0.2},
            "Chess": {"strategic_thinking": 0.45, "decision_efficiency": 0.3, "problem_solving": 0.25},
            "StopObject": {"decision_efficiency": 0.6, "pattern_recognition": 0.4},
            "Minesweeper": {"logical_reasoning": 0.4, "pattern_recognition": 0.35, "decision_efficiency": 0.25},
            "2048": {"pattern_recognition": 0.4, "strategic_thinking": 0.35, "decision_efficiency": 0.25},
            "NQueens": {"logical_reasoning": 0.35, "problem_solving": 0.4, "spatial_reasoning": 0.25},
            "Hanoi": {"problem_solving": 0.5, "strategic_thinking": 0.3, "logical_reasoning": 0.2},
            "Mastermind": {"logical_reasoning": 0.45, "pattern_recognition": 0.35, "decision_efficiency": 0.2},
            "ConnectFour": {"strategic_thinking": 0.5, "decision_efficiency": 0.3, "pattern_recognition": 0.2},
            "Nonogram": {"pattern_recognition": 0.5, "logical_reasoning": 0.35, "spatial_reasoning": 0.15},
            "Rubiks": {"spatial_reasoning": 0.55, "pattern_recognition": 0.3, "problem_solving": 0.15},
        }
        weights = mapping.get(game, {"problem_solving": 1.0})
        alpha = 0.35  # blend factor toward new score
        for axis, w in weights.items():
            old = float(cog.get(axis, 50.0))
            blended = (1 - alpha * w) * old + (alpha * w) * score
            cog[axis] = round(min(100.0, max(0.0, blended)), 1)
        self.user_data["cognitive_intelligence"] = cog

    def _get_cognitive_summary(self):
        cog = self.user_data.get("cognitive_intelligence") or {}
        if not cog:
            return "Play brain games to build Cognitive Intelligence scores."
        lines = [f"  {k.replace('_',' ').title()}: {v}/100" for k, v in cog.items()]
        avg = sum(cog.values()) / max(1, len(cog))
        return f"🧠 Cognitive Intelligence (avg {avg:.0f}/100)\n" + "\n".join(lines)

    def _clear_brain_history(self):
        """Clear only brain game win/loss/score history (not login/courses)."""
        self.user_data["brain_scores"] = []
        db.save_user_profile(self.current_user_id, self.user_data)
        self._refresh_brain_hist()
        self.refresh_ui()

    def _open_stop_object(self):
        """Red ball moves horizontally; click to stop. Lv1–5 medium, Lv6–10 super fast."""
        win = ctk.CTkToplevel(self)
        win.title("Stop the Object")
        win.geometry("560x560")
        win.configure(fg_color="#11111B")
        win.transient(self)
        win.lift()
        win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        self._brain_win = win
        lvl_var = ctk.StringVar(value="5")
        ctk.CTkLabel(win, text="1. Select level  2. Start  3. Click the RED ball on the line",
                     text_color="#CDD6F4").pack(pady=(10, 4))
        ctk.CTkOptionMenu(win, variable=lvl_var, values=[str(i) for i in range(1, 11)],
                          fg_color="#313244").pack()
        info = ctk.CTkLabel(win, text="Level 1–5: medium speed · Level 6–10: super high speed",
                            text_color="#A6ADC8")
        info.pack(pady=4)
        canvas = ctk.CTkCanvas(win, width=520, height=200, bg="#1E1E2E", highlightthickness=0)
        canvas.pack(pady=8)
        # landscape line
        canvas.create_line(20, 100, 500, 100, fill="#585B70", width=3)
        state = {
            "running": False, "ball": None, "x": 20.0, "dir": 1,
            "start": None, "done": False, "after_id": None, "r": 14,
        }
        results = []  # seconds history this window

        def speed_for_level(level):
            # pixels per tick (~16ms). 1–5 medium, 6–10 super
            if level <= 5:
                return 3.0 + level * 1.2   # ~4.2 … 9
            return 12.0 + (level - 5) * 3.5  # ~15.5 … 29.5

        def tick():
            if not state["running"] or state["done"]:
                return
            level = int(lvl_var.get())
            sp = speed_for_level(level)
            state["x"] += state["dir"] * sp
            if state["x"] >= 500 - state["r"]:
                state["x"] = 500 - state["r"]
                state["dir"] = -1
            elif state["x"] <= 20 + state["r"]:
                state["x"] = 20 + state["r"]
                state["dir"] = 1
            r = state["r"]
            y = 100
            canvas.coords(state["ball"], state["x"] - r, y - r, state["x"] + r, y + r)
            state["after_id"] = win.after(16, tick)

        def start_run():
            if state["after_id"]:
                try:
                    win.after_cancel(state["after_id"])
                except Exception:
                    pass
            canvas.delete("ball")
            canvas.delete("line")
            canvas.create_line(20, 100, 500, 100, fill="#585B70", width=3, tags="line")
            r = max(10, 18 - int(lvl_var.get()))
            state["r"] = r
            state["x"] = 30.0
            state["dir"] = 1
            state["done"] = False
            state["running"] = True
            state["start"] = time.time()
            state["ball"] = canvas.create_oval(
                state["x"] - r, 100 - r, state["x"] + r, 100 + r,
                fill="#F38BA8", outline="", tags="ball")
            info.configure(text="Click the moving RED ball!", text_color="#FAB387")
            tick()

        def on_click(event):
            if not state["running"] or state["done"] or state["ball"] is None:
                return
            items = canvas.find_overlapping(event.x - 3, event.y - 3, event.x + 3, event.y + 3)
            if state["ball"] not in items:
                return
            state["done"] = True
            state["running"] = False
            if state["after_id"]:
                try:
                    win.after_cancel(state["after_id"])
                except Exception:
                    pass
            sec = time.time() - (state["start"] or time.time())
            level = int(lvl_var.get())
            # score: faster + higher level = better
            target = 1.2 if level <= 5 else 0.7
            score = max(0, min(100, 100 * (target / max(0.15, sec)) * (0.7 + level * 0.03)))
            results.append(sec)
            info.configure(
                text=f"Stopped in {sec:.2f} s  ·  Score {score:.0f}%  (Lv{level})",
                text_color="#A6E3A1")
            self._save_brain_score("StopObject", level, score, f"{sec:.2f}s")

        def show_graph():
            if not results:
                info.configure(text="Play at least once to see graph", text_color="#FAB387")
                return
            gwin = ctk.CTkToplevel(win)
            gwin.title("Stop-the-Object results")
            gwin.geometry("480x320")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(1, len(results) + 1), results, marker="o", color="#F38BA8")
            ax.set_xlabel("Attempt")
            ax.set_ylabel("Seconds to stop")
            ax.set_title("Reaction times (lower is better)")
            fig.tight_layout()
            canvas_g = FigureCanvasTkAgg(fig, master=gwin)
            canvas_g.draw()
            canvas_g.get_tk_widget().pack(fill="both", expand=True)

        canvas.bind("<Button-1>", on_click)
        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(pady=8)
        ctk.CTkButton(btn_row, text="Start", fg_color="#A6E3A1", text_color="#11111B",
                      command=start_run).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="Show Result Graph", fg_color="#89B4FA", text_color="#11111B",
                      command=show_graph).pack(side="left", padx=6)

    def _open_sudoku(self):
        """Sudoku with backtracking solver for generation + unique-solution carving."""
        import random as _rnd
        win = ctk.CTkToplevel(self)
        win.title("Sudoku — Solver-backed")
        win.geometry("440x560")
        win.configure(fg_color="#11111B")
        win.transient(self)
        win.lift()
        win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        self._brain_win = win
        lvl_var = ctk.StringVar(value="5")
        ctk.CTkLabel(win, text="Level 1–10 (solver carves unique puzzles)", text_color="#CDD6F4").pack(pady=6)
        ctk.CTkOptionMenu(win, variable=lvl_var, values=[str(i) for i in range(1, 11)],
                          fg_color="#313244").pack()
        status = ctk.CTkLabel(win, text="", text_color="#A6ADC8")
        status.pack()
        grid_frame = ctk.CTkFrame(win, fg_color="#181825")
        grid_frame.pack(pady=8)
        entries = [[None] * 9 for _ in range(9)]
        solution = {"g": None}
        t0 = {"t": None}

        def valid(g, r, c, n):
            if any(g[r][j] == n for j in range(9)):
                return False
            if any(g[i][c] == n for i in range(9)):
                return False
            br, bc = 3 * (r // 3), 3 * (c // 3)
            for i in range(br, br + 3):
                for j in range(bc, bc + 3):
                    if g[i][j] == n:
                        return False
            return True

        def find_empty(g):
            for i in range(9):
                for j in range(9):
                    if g[i][j] == 0:
                        return i, j
            return None

        def solve(g):
            """Backtracking Sudoku solver. Returns True if solved in-place."""
            pos = find_empty(g)
            if not pos:
                return True
            r, c = pos
            nums = list(range(1, 10))
            _rnd.shuffle(nums)
            for n in nums:
                if valid(g, r, c, n):
                    g[r][c] = n
                    if solve(g):
                        return True
                    g[r][c] = 0
            return False

        def count_solutions(g, limit=2):
            """Count solutions up to limit (for uniqueness check)."""
            pos = find_empty(g)
            if not pos:
                return 1
            r, c = pos
            count = 0
            for n in range(1, 10):
                if valid(g, r, c, n):
                    g[r][c] = n
                    count += count_solutions(g, limit)
                    g[r][c] = 0
                    if count >= limit:
                        return count
            return count

        def generate_full():
            g = [[0] * 9 for _ in range(9)]
            # seed diagonal boxes for faster solve
            for k in range(3):
                nums = list(range(1, 10))
                _rnd.shuffle(nums)
                for i in range(3):
                    for j in range(3):
                        g[k * 3 + i][k * 3 + j] = nums[i * 3 + j]
            solve(g)
            return g

        def carve(sol, level):
            """Remove cells while keeping unique solution. Higher level → more blanks."""
            g = [row[:] for row in sol]
            target_blanks = 28 + level * 4  # ~32 … 68
            cells = [(r, c) for r in range(9) for c in range(9)]
            _rnd.shuffle(cells)
            removed = 0
            for r, c in cells:
                if removed >= target_blanks:
                    break
                backup = g[r][c]
                g[r][c] = 0
                test = [row[:] for row in g]
                if count_solutions(test, limit=2) != 1:
                    g[r][c] = backup  # restore — would not be unique
                else:
                    removed += 1
            return g

        def new_puzzle():
            sol = generate_full()
            level = int(lvl_var.get())
            g = carve(sol, level)
            solution["g"] = sol
            t0["t"] = time.time()
            for r in range(9):
                for c in range(9):
                    e = entries[r][c]
                    e.configure(state="normal")
                    e.delete(0, "end")
                    if g[r][c]:
                        e.insert(0, str(g[r][c]))
                        e.configure(state="disabled")
            blanks = sum(1 for r in range(9) for c in range(9) if g[r][c] == 0)
            status.configure(
                text=f"Unique puzzle · {blanks} blanks · Level {level}", text_color="#A6ADC8")

        for r in range(9):
            for c in range(9):
                e = ctk.CTkEntry(grid_frame, width=32, height=28, justify="center",
                                 fg_color="#11111B", border_color="#313244")
                e.grid(row=r, column=c, padx=1, pady=1)
                entries[r][c] = e

        def check():
            if solution["g"] is None:
                return
            ok = total = 0
            for r in range(9):
                for c in range(9):
                    if entries[r][c].cget("state") == "disabled":
                        continue
                    total += 1
                    val = entries[r][c].get().strip()
                    try:
                        if int(val) == solution["g"][r][c]:
                            ok += 1
                    except Exception:
                        pass
            elapsed = time.time() - (t0["t"] or time.time())
            acc = 100.0 * ok / max(1, total)
            time_pen = min(40, elapsed / 6)
            score = max(0, min(100, acc - time_pen + int(lvl_var.get()) * 2))
            status.configure(
                text=f"Correct {ok}/{total} · {elapsed:.0f}s · Score {score:.0f}%",
                text_color="#A6E3A1")
            self._save_brain_score("Sudoku", int(lvl_var.get()), score,
                                   f"{ok}/{total} in {elapsed:.0f}s")

        def hint_solve_one():
            """Fill one empty cell from solver solution (learning aid)."""
            if solution["g"] is None:
                return
            empties = [(r, c) for r in range(9) for c in range(9)
                       if entries[r][c].cget("state") != "disabled" and not entries[r][c].get().strip()]
            if not empties:
                return
            r, c = empties[0]
            entries[r][c].delete(0, "end")
            entries[r][c].insert(0, str(solution["g"][r][c]))

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(pady=8)
        ctk.CTkButton(btn_row, text="New Puzzle", fg_color="#89B4FA", text_color="#11111B",
                      command=new_puzzle).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Hint", fg_color="#313244", text_color="#CDD6F4",
                      command=hint_solve_one).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Check", fg_color="#A6E3A1", text_color="#11111B",
                      command=check).pack(side="left", padx=4)
        new_puzzle()

    def _open_chess(self):
        """Chess UI — engine: Opening Book + Minimax/Alpha-Beta + NN-style eval."""
        try:
            import chess
        except ImportError:
            win = ctk.CTkToplevel(self)
            win.title("Install chess package")
            win.geometry("460x220")
            win.configure(fg_color="#11111B")
            ctk.CTkLabel(win, text="python-chess package missing",
                         font=ctk.CTkFont(size=16, weight="bold"), text_color="#F38BA8").pack(pady=(18, 6))
            ctk.CTkLabel(win, text="Run this in PowerShell / terminal, then restart Acadexa:",
                         text_color="#CDD6F4").pack(pady=4)
            ctk.CTkLabel(win, text="python -m pip install chess",
                         font=ctk.CTkFont(size=14, weight="bold"), text_color="#89B4FA").pack(pady=6)

            def _try_pip():
                import subprocess
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "chess"])
                    win.destroy()
                    self._open_chess()
                except Exception as e:
                    ctk.CTkLabel(win, text=f"Auto-install failed: {e}", text_color="#F38BA8").pack()

            ctk.CTkButton(win, text="Auto Install Now", fg_color="#A6E3A1", text_color="#11111B",
                          command=_try_pip).pack(pady=10)
            return

        import random as _rnd
        win = ctk.CTkToplevel(self)
        win.title("Chess — Graphical Board")
        win.geometry("620x760")
        win.configure(fg_color="#11111B")
        win.transient(self)
        win.lift()
        win.focus_force()
        # keep strong refs so window is not GC'd / auto-closed
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        self._brain_win = win

        board = chess.Board()
        state = {
            "board": board, "moves_user": 0, "moves_cpu": 0,
            "t0": time.time(), "over": False, "selected": None,
        }
        lvl_var = ctk.StringVar(value="3")
        ctk.CTkLabel(win, text="Level 1–10 · Book → Alpha-Beta · NN-style eval", text_color="#CDD6F4").pack(pady=4)
        ctk.CTkOptionMenu(win, variable=lvl_var, values=[str(i) for i in range(1, 11)],
                          fg_color="#313244").pack()
        info = ctk.CTkLabel(win, text="You = White. Click a piece, then destination square (or type e2e4)",
                            text_color="#A6ADC8")
        info.pack(pady=4)

        # --- Graphical chess board (8x8) ---
        SQ = 56
        BOARD_PX = SQ * 8
        canvas = ctk.CTkCanvas(win, width=BOARD_PX + 36, height=BOARD_PX + 36,
                               bg="#1E1E2E", highlightthickness=0)
        canvas.pack(pady=8)
        LIGHT, DARK = "#F0D9B5", "#B58863"
        PIECE_GLYPH = {
            "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
            "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
        }

        def sq_to_xy(sq):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            x = 18 + file * SQ
            y = 18 + (7 - rank) * SQ
            return x, y

        def xy_to_sq(x, y):
            file = int((x - 18) // SQ)
            rank_from_top = int((y - 18) // SQ)
            if not (0 <= file <= 7 and 0 <= rank_from_top <= 7):
                return None
            rank = 7 - rank_from_top
            return chess.square(file, rank)

        def draw_board():
            canvas.delete("all")
            b = state["board"]
            for rank in range(8):
                for file in range(8):
                    x0 = 18 + file * SQ
                    y0 = 18 + (7 - rank) * SQ
                    color = LIGHT if (file + rank) % 2 == 0 else DARK
                    sq = chess.square(file, rank)
                    if state["selected"] == sq:
                        color = "#829769"
                    canvas.create_rectangle(x0, y0, x0 + SQ, y0 + SQ, fill=color, outline="")
                    # file/rank labels
                    if file == 0:
                        canvas.create_text(10, y0 + SQ / 2, text=str(rank + 1),
                                           fill="#A6ADC8", font=("Segoe UI", 9))
                    if rank == 0:
                        canvas.create_text(x0 + SQ / 2, BOARD_PX + 28,
                                           text="abcdefgh"[file], fill="#A6ADC8",
                                           font=("Segoe UI", 9))
            for sq, piece in b.piece_map().items():
                x0, y0 = sq_to_xy(sq)
                glyph = PIECE_GLYPH.get(piece.symbol(), piece.symbol())
                # white pieces slightly brighter
                fill = "#111111" if piece.color == chess.BLACK else "#FAFAFA"
                canvas.create_text(x0 + SQ / 2, y0 + SQ / 2, text=glyph,
                                   font=("Segoe UI Symbol", 32), fill=fill)
            # last move highlight optional
            if b.move_stack:
                mv = b.peek()
                for s in (mv.from_square, mv.to_square):
                    x0, y0 = sq_to_xy(s)
                    canvas.create_rectangle(x0 + 2, y0 + 2, x0 + SQ - 2, y0 + SQ - 2,
                                            outline="#89B4FA", width=2)

        def on_board_click(event):
            if state["over"]:
                return
            sq = xy_to_sq(event.x, event.y)
            if sq is None:
                return
            b = state["board"]
            if state["selected"] is None:
                piece = b.piece_at(sq)
                if piece and piece.color == chess.WHITE and b.turn == chess.WHITE:
                    state["selected"] = sq
                    draw_board()
                return
            # second click = try move
            frm = state["selected"]
            state["selected"] = None
            mv = chess.Move(frm, sq)
            # promotion auto-queen
            if (b.piece_at(frm) and b.piece_at(frm).piece_type == chess.PAWN
                    and chess.square_rank(sq) == 7):
                mv = chess.Move(frm, sq, promotion=chess.QUEEN)
            if mv not in b.legal_moves:
                # try without promotion already handled
                info.configure(text="Illegal move", text_color="#F38BA8")
                draw_board()
                return
            b.push(mv)
            state["moves_user"] += 1
            draw_board()
            finish_if_needed()
            if not state["over"]:
                win.after(150, cpu_move)

        canvas.bind("<Button-1>", on_board_click)
        draw_board()

        move_e = ctk.CTkEntry(win, placeholder_text="or type e2e4", width=140, fg_color="#181825")
        move_e.pack(pady=4)
        engine_note = ctk.CTkLabel(
            win,
            text="Click squares OR type UCI · Pipeline: Book → αβ → NN features",
            font=ctk.CTkFont(size=10), text_color="#585B70")
        engine_note.pack(pady=2)

        def cpu_move():
            b = state["board"]
            if b.is_game_over():
                return
            level = int(lvl_var.get())
            info.configure(text=f"CPU thinking (Lv{level})…", text_color="#FAB387")
            win.update_idletasks()
            mv, src = ChessAIEngine.choose_move(b, level, chess, _rnd)
            if mv is None:
                return
            b.push(mv)
            state["moves_cpu"] += 1
            draw_board()
            info.configure(text=f"CPU {mv.uci()} via {src}", text_color="#A6ADC8")
            finish_if_needed()

        def finish_if_needed():
            b = state["board"]
            if not b.is_game_over() or state["over"]:
                return
            state["over"] = True
            elapsed = time.time() - state["t0"]
            if b.is_checkmate():
                result = "You Win" if b.turn == chess.BLACK else "CPU Wins"
            else:
                result = "Draw"
            level = int(lvl_var.get())
            if result == "You Win":
                score = min(100, 50 + level * 5)
            elif result == "Draw":
                score = 40 + level * 2
            else:
                score = max(5, 35 - level * 2)
            detail = (f"{result} · you {state['moves_user']} moves · "
                      f"CPU {state['moves_cpu']} moves · {elapsed:.0f}s")
            info.configure(text=detail, text_color="#A6E3A1")
            self._save_brain_score("Chess", level, score, detail)

        def do_move():
            if state["over"]:
                return
            txt = move_e.get().strip().lower()
            try:
                mv = chess.Move.from_uci(txt)
            except Exception:
                info.configure(text="Invalid format. Use e2e4", text_color="#F38BA8")
                return
            b = state["board"]
            if mv not in b.legal_moves:
                info.configure(text="Illegal move", text_color="#F38BA8")
                return
            b.push(mv)
            state["moves_user"] += 1
            move_e.delete(0, "end")
            draw_board()
            finish_if_needed()
            if not state["over"]:
                win.after(120, cpu_move)

        ctk.CTkButton(win, text="Play Move (typed)", fg_color="#A6E3A1", text_color="#11111B",
                      command=do_move).pack(pady=6)
        ctk.CTkLabel(
            win,
            text="Lv1: no book · Lv2+: opening book · deeper αβ · click board to move",
            font=ctk.CTkFont(size=10), text_color="#585B70").pack(pady=4)

    # ---------- ATS ----------
    def _build_ats(self):
        frame = ctk.CTkScrollableFrame(self.tab_ats, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        ctk.CTkLabel(frame, text="ATS Resume & Job Description Matcher",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#CDD6F4").pack(anchor="w", pady=(0, 4))
        ctk.CTkLabel(
            frame,
            text="Match resume text to a job description, and get career suggestions from your Student Intelligence Profile.",
            text_color="#A6ADC8", font=ctk.CTkFont(size=11), wraplength=700, justify="left",
        ).pack(anchor="w", pady=(0, 10))

        # ---- Career suggestions from profile (dynamic) ----
        sug_card = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=12, border_width=1, border_color="#313244")
        sug_card.pack(fill="x", pady=(0, 12))
        head = ctk.CTkFrame(sug_card, fg_color="transparent")
        head.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(head, text="Career / Job Suggestions (from your profile)",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color="#89B4FA").pack(side="left")
        ctk.CTkButton(head, text="Refresh from Profile", width=150, fg_color="#313244",
                      command=self._refresh_career_suggestions).pack(side="right")
        self.lbl_career_basis = ctk.CTkLabel(
            sug_card, text="", text_color="#A6ADC8", font=ctk.CTkFont(size=11),
            wraplength=680, justify="left",
        )
        self.lbl_career_basis.pack(anchor="w", padx=12, pady=(0, 6))
        self.career_sug_box = ctk.CTkFrame(sug_card, fg_color="transparent")
        self.career_sug_box.pack(fill="x", padx=8, pady=(0, 10))
        self.after(200, self._refresh_career_suggestions)

        ctk.CTkLabel(frame, text="Paste Resume Text", text_color="#89B4FA").pack(anchor="w")
        self.txt_resume = ctk.CTkTextbox(frame, height=130, fg_color="#181825", border_color="#313244")
        self.txt_resume.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(frame, text="Load last generated resume text", fg_color="#313244",
                      command=self._ats_load_resume_from_history).pack(anchor="w", pady=(0, 8))

        ctk.CTkLabel(frame, text="Paste Target Job Description", text_color="#89B4FA").pack(anchor="w")
        self.txt_jd = ctk.CTkTextbox(frame, height=130, fg_color="#181825", border_color="#313244")
        self.txt_jd.pack(fill="x", pady=(0, 10))
        ctk.CTkButton(frame, text="Evaluate Match Score", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._analyze_ats).pack(fill="x", pady=(0, 10))
        self.lbl_ats = ctk.CTkLabel(frame, text="Match Score: --%", font=ctk.CTkFont(size=18, weight="bold"),
                                    text_color="#89B4FA")
        self.lbl_ats.pack(anchor="w")
        self.txt_ats_details = ctk.CTkTextbox(frame, height=110, fg_color="#181825", border_color="#313244")
        self.txt_ats_details.pack(fill="x", pady=6)

    def _ats_load_resume_from_history(self):
        hist = self.user_data.get("resume_history") or []
        if not hist:
            self.txt_ats_details.delete("1.0", "end")
            self.txt_ats_details.insert("1.0", "No saved resume yet. Build one in CV & Resume Builder first.")
            return
        text = hist[-1].get("text") or ""
        self.txt_resume.delete("1.0", "end")
        self.txt_resume.insert("1.0", text)

    def _career_suggestion_engine(self):
        """Build ranked career/job ideas ONLY from profile fields — no invented credentials."""
        sp = self.user_data.get("student_profile", {}) or {}
        courses = self.user_data.get("courses", []) or []
        skills = self.user_data.get("skills", []) or []
        projects = self.user_data.get("projects", []) or []
        mastery = self.user_data.get("topic_mastery", {}) or {}
        goal = (sp.get("career_goal") or "").strip()
        dept = (sp.get("department") or sp.get("major") or "").strip().lower()
        degree = (sp.get("degree") or "").strip().lower()
        faculty = (sp.get("faculty") or "").strip().lower()

        skill_names = []
        for s in skills:
            skill_names.append((s.get("name") if isinstance(s, dict) else str(s)).strip().lower())
        skill_names = [s for s in skill_names if s]
        course_titles = " ".join((c.get("title") or "") + " " + (c.get("code") or "") for c in courses).lower()
        proj_blob = " ".join(
            str(p.get("name", "") if isinstance(p, dict) else p) + " " +
            str(p.get("tech", "") if isinstance(p, dict) else "")
            for p in projects
        ).lower()
        mastery_topics = " ".join(mastery.keys()).lower()
        blob = " ".join([dept, degree, faculty, goal.lower(), course_titles, " ".join(skill_names), proj_blob, mastery_topics])

        # Role templates: keywords that must appear in profile blob to suggest
        catalog = [
            (["cse", "computer", "software", "cs", "programming", "coding"], [
                ("Software Engineer Intern", "Junior software development roles; emphasize projects & GitHub."),
                ("Backend Developer (Junior)", "APIs, databases, server-side — if you have related coursework/projects."),
                ("Frontend Developer (Junior)", "UI apps — if you have web/mobile project evidence."),
                ("QA / Test Engineer Intern", "Testing mindset; good if you document projects carefully."),
                ("DevOps Intern", "Only if you have Linux/cloud/CI exposure in skills or projects."),
            ]),
            (["eee", "electrical", "electronics", "power"], [
                ("Electrical Engineering Intern", "Power/electronics labs and coursework as proof."),
                ("Embedded Systems Intern", "If microcontrollers/IoT appear in courses or projects."),
                ("Hardware Test Intern", "Lab measurement and documentation skills."),
            ]),
            (["data", "ml", "machine learning", "ai", "analytics", "statistics", "python"], [
                ("Data Analyst Intern", "Spreadsheets, SQL, Python analysis if present in skills/projects."),
                ("ML Engineering Intern", "Only if ML/AI projects or courses exist in your profile."),
                ("Business Intelligence Intern", "Reporting/dashboard interest + analytical coursework."),
            ]),
            (["network", "cyber", "security", "cloud"], [
                ("Network Support Intern", "Networking courses as foundation."),
                ("Cybersecurity Intern", "Only if security topics appear in mastery/courses."),
                ("Cloud Support Associate", "If cloud tools appear in skills."),
            ]),
            (["mechanical", "me ", "thermal", "design"], [
                ("Mechanical Design Intern", "CAD/design coursework."),
                ("Manufacturing Intern", "Process/production related courses."),
            ]),
            (["civil", "structure", "construction"], [
                ("Site Engineer Intern", "Civil coursework and site exposure."),
                ("Structural Design Intern", "If structural courses listed."),
            ]),
            (["bba", "business", "marketing", "finance", "accounting", "management"], [
                ("Business Analyst Intern", "Problem structuring + tools."),
                ("Marketing Intern", "If marketing/comms related goal or courses."),
                ("Finance Intern", "If accounting/finance coursework exists."),
            ]),
        ]

        suggestions = []
        seen = set()
        # Always respect explicit career goal first
        if goal:
            key = goal.lower()
            if key not in seen:
                seen.add(key)
                suggestions.append({
                    "title": goal if "intern" in goal.lower() or "engineer" in goal.lower() or "analyst" in goal.lower()
                    else f"{goal} (entry-level / intern track)",
                    "why": "Taken from your Career Goal in Student Intelligence Profile.",
                    "fit": "High (your stated goal)",
                    "sample_jd": (
                        f"We are looking for a motivated candidate interested in {goal}. "
                        f"Responsibilities include learning the stack, contributing to team tasks, "
                        f"and documenting work. Preferred: relevant coursework, projects, and communication skills."
                    ),
                })

        for keys, roles in catalog:
            if any(k in blob for k in keys):
                for title, why in roles:
                    if title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    suggestions.append({
                        "title": title,
                        "why": why,
                        "fit": "Profile-matched",
                        "sample_jd": (
                            f"Internship/junior role: {title}. Responsibilities: assist the team, "
                            f"complete assigned tasks, write clear documentation, and learn industry tools. "
                            f"Requirements: foundational knowledge from your degree/courses, willingness to learn, "
                            f"and evidence from academic projects where available."
                        ),
                    })
                    if len(suggestions) >= 10:
                        break
            if len(suggestions) >= 10:
                break

        if not suggestions:
            suggestions.append({
                "title": "General Intern / Trainee",
                "why": "Profile is light — add department, skills, projects, and career goal for better matches.",
                "fit": "Generic",
                "sample_jd": (
                    "Entry-level intern: support day-to-day team work, complete training modules, "
                    "and contribute to documentation. Requirements: enrolled student, basic computer literacy."
                ),
            })

        basis_parts = []
        if sp.get("name"):
            basis_parts.append(f"Name: {sp.get('name')}")
        if sp.get("department") or sp.get("major"):
            basis_parts.append(f"Dept/Major: {sp.get('department') or sp.get('major')}")
        if sp.get("degree"):
            basis_parts.append(f"Degree: {sp.get('degree')}")
        if goal:
            basis_parts.append(f"Goal: {goal}")
        if courses:
            basis_parts.append(f"Courses: {len(courses)}")
        if skills:
            basis_parts.append(f"Skills: {len(skills)}")
        if projects:
            basis_parts.append(f"Projects: {len(projects)}")
        if mastery:
            basis_parts.append(f"Mastered topics: {len(mastery)}")
        return suggestions, " · ".join(basis_parts) if basis_parts else "Complete your Student Intelligence Profile for sharper suggestions."

    def _refresh_career_suggestions(self):
        if not hasattr(self, "career_sug_box"):
            return
        for w in self.career_sug_box.winfo_children():
            w.destroy()
        suggestions, basis = self._career_suggestion_engine()
        if hasattr(self, "lbl_career_basis"):
            self.lbl_career_basis.configure(text=f"Based on: {basis}")

        for i, s in enumerate(suggestions[:8]):
            card = ctk.CTkFrame(self.career_sug_box, fg_color="#11111B", corner_radius=8,
                                border_width=1, border_color="#313244")
            card.pack(fill="x", pady=4, padx=4)
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(8, 2))
            ctk.CTkLabel(top, text=f"{i+1}. {s['title']}", font=ctk.CTkFont(size=13, weight="bold"),
                         text_color="#CDD6F4").pack(side="left")
            ctk.CTkLabel(top, text=s.get("fit", ""), text_color="#A6E3A1",
                         font=ctk.CTkFont(size=11)).pack(side="right")
            ctk.CTkLabel(card, text=s.get("why", ""), text_color="#A6ADC8", font=ctk.CTkFont(size=11),
                         wraplength=640, justify="left").pack(anchor="w", padx=10, pady=(0, 6))

            def use_jd(sample=s.get("sample_jd", ""), title=s["title"]):
                self.txt_jd.delete("1.0", "end")
                self.txt_jd.insert("1.0", sample)
                self.lbl_ats.configure(text=f"JD loaded: {title}", text_color="#A6E3A1")

            ctk.CTkButton(card, text="Use as Job Description", width=160, height=28,
                          fg_color="#89B4FA", text_color="#11111B",
                          command=use_jd).pack(anchor="e", padx=10, pady=(0, 8))

    def _analyze_ats(self):
        resume = self.txt_resume.get("1.0", "end").strip()
        jd = self.txt_jd.get("1.0", "end").strip()
        if not resume or not jd:
            self.lbl_ats.configure(text="Match Score: --%", text_color="#F38BA8")
            self.txt_ats_details.delete("1.0", "end")
            self.txt_ats_details.insert(
                "1.0",
                "Paste both Resume text and Job Description.\n"
                "Tip: use Load last generated resume + a Career Suggestion JD.",
            )
            return
        score, matched, missing = job_matcher.match(resume, jd)
        # Filter trivial matched tokens (years, single letters)
        matched_f = [m for m in matched if len(m) > 2 and not m.isdigit()]
        missing_f = [m for m in missing if len(m) > 2 and not m.isdigit()]
        color = "#A6E3A1" if score >= 60 else ("#FAB387" if score >= 35 else "#F38BA8")
        self.lbl_ats.configure(text=f"Match Score: {score}%", text_color=color)
        self.txt_ats_details.delete("1.0", "end")
        self.txt_ats_details.insert(
            "1.0",
            f"Matched keywords: {', '.join(matched_f[:15]) if matched_f else 'None'}\n\n"
            f"Missing keywords (add only if true for you): {', '.join(missing_f[:15]) if missing_f else 'None'}\n\n"
            f"Note: Score is a TF-IDF text similarity estimate — not a guarantee of ATS success. "
            f"Avoid keyword stuffing.",
        )


    # ---------- PROFILE ----------
    def _build_profile(self):
        frame = ctk.CTkScrollableFrame(self.tab_profile, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=12, pady=8)
        ctk.CTkLabel(frame, text="Student Intelligence Profile",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#CDD6F4").pack(anchor="w", pady=(0, 10))

        pic_box = ctk.CTkFrame(frame, fg_color="#181825", corner_radius=10)
        pic_box.pack(fill="x", pady=6)
        self.profile_avatar = ctk.CTkLabel(pic_box, text="", image=make_circular_image(None, size=(60, 60)))
        self.profile_avatar.pack(side="left", padx=12, pady=10)
        ctk.CTkButton(pic_box, text="Change Profile Picture", fg_color="#89B4FA", text_color="#11111B",
                      command=self._change_pic).pack(side="left", padx=8)

        self.profile_entries = {}
        fields = [
            ("Student ID", "student_id"), ("University", "university"),
            ("Faculty", "faculty"), ("Department", "department"),
            ("Degree", "degree"), ("Major", "major"), ("Minor", "minor"),
            ("Semester / Level", "semester_year"), ("GPA (out of 4.0)", "gpa"),
            ("Career Goal", "career_goal"),
        ]
        for label, key in fields:
            ctk.CTkLabel(frame, text=label, text_color="#CDD6F4").pack(anchor="w", pady=(4, 1))
            e = ctk.CTkEntry(frame, fg_color="#181825", border_color="#313244")
            e.pack(fill="x", pady=(0, 4))
            self.profile_entries[key] = e
        ctk.CTkButton(frame, text="Save Profile Changes", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=self._save_profile).pack(fill="x", pady=12)

    def _change_pic(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp")])
        if path:
            self.user_data.setdefault("student_profile", {})["profile_pic"] = path
            db.save_user_profile(self.current_user_id, self.user_data)
            self.profile_avatar.configure(image=make_circular_image(path, size=(60, 60)))
            self._render_sidebar_profile()

    def _save_profile(self):
        sp = self.user_data.setdefault("student_profile", {})
        for k, e in self.profile_entries.items():
            sp[k] = e.get().strip()
        if not sp.get("gpa"):
            sp["gpa"] = "0.0"
        db.save_user_profile(self.current_user_id, self.user_data)
        self.refresh_ui()

    # ---------- REFRESH ----------
    def refresh_ui(self):
        if not self.current_user_id:
            return
        self.user_data = db.get_user_profile(self.current_user_id)
        sp = self.user_data.get("student_profile", {})
        self._render_sidebar_profile()

        if hasattr(self, "profile_entries"):
            for k, e in self.profile_entries.items():
                e.delete(0, "end")
                e.insert(0, str(sp.get(k, "")))
        if hasattr(self, "profile_avatar"):
            self.profile_avatar.configure(image=make_circular_image(sp.get("profile_pic", ""), size=(60, 60)))

        # courses list
        if hasattr(self, "course_scroll"):
            for w in self.course_scroll.winfo_children():
                w.destroy()
            courses = self.user_data.get("courses", [])
            self.lbl_course_count.configure(text=f"Subjects: {len(courses)} / 10")
            for i, c in enumerate(courses):
                card = ctk.CTkFrame(self.course_scroll, fg_color="#11111B", corner_radius=8)
                card.pack(fill="x", pady=4, padx=4)
                top = ctk.CTkFrame(card, fg_color="transparent")
                top.pack(fill="x", padx=8, pady=4)
                title = f"{c.get('code', '')}  {c.get('title', '')}".strip()
                ctk.CTkLabel(top, text=title, font=ctk.CTkFont(size=13, weight="bold"),
                             text_color="#89B4FA").pack(side="left")
                ctk.CTkButton(top, text="✕", width=28, height=24, fg_color="#F38BA8", text_color="#11111B",
                              command=lambda idx=i: self._remove_course(idx)).pack(side="right")
                topics = ", ".join(c.get("topics", [])) or "No topics"
                ctk.CTkLabel(card, text=f"Domain: {c.get('discipline', '')} | Topics: {topics}",
                             font=ctk.CTkFont(size=11), text_color="#A6ADC8", wraplength=520,
                             justify="left").pack(anchor="w", padx=10, pady=(0, 6))

        # timer subjects
        if hasattr(self, "cmb_timer_subject"):
            titles = []
            for c in self.user_data.get("courses", []):
                t = f"{c.get('code', '')} {c.get('title', '')}".strip()
                if t:
                    titles.append(t)
                for tp in c.get("topics", [])[:5]:
                    titles.append(f"{t} | {tp}")
            for r in self.user_data.get("study_routine", []):
                if r.get("type") == "study" and r.get("subject") and r["subject"] not in titles:
                    titles.append(r["subject"])
            if titles:
                self.cmb_timer_subject.configure(values=titles[:30])
                self.cmb_timer_subject.set(titles[0])
            else:
                self.cmb_timer_subject.configure(values=["Custom Subject Entry"])
                self.cmb_timer_subject.set("Custom Subject Entry")

        # metrics
        ev = decision_agent.evaluate_student(self.user_data)
        study_ev = study_intelligence.evaluate(self.user_data.get("study_sessions", []))
        self.card_academic.configure(text=f"{ev.get('task_completion', ev.get('academic_health', 0))}%")
        self.card_mastery.configure(text=f"{ev['knowledge_mastery']}%")
        self.card_readiness.configure(text=f"{ev['job_readiness']}%")
        self.card_routine.configure(text=f"{ev['routine_efficiency']}%")
        self.card_hours.configure(text=f"{study_ev['total_hours']} hrs")
        self.card_consistency.configure(text=f"{ev['learning_intelligence']:.0f}%")
        self.card_quizzes.configure(text=str(len(self.user_data.get("quiz_history", []))))
        self.card_status.configure(text=f"{ev.get('habit_health', 0)}%")
        self.lbl_recommendations.configure(text="\n".join(f"• {r}" for r in ev["recommendations"]))

        # assessments list
        if hasattr(self, "assess_box"):
            for w in self.assess_box.winfo_children():
                w.destroy()
            hist = self.user_data.get("quiz_history", [])[-8:]
            if not hist:
                ctk.CTkLabel(self.assess_box, text="No quizzes yet.", text_color="#A6ADC8").pack(padx=10, pady=10)
            else:
                for h in reversed(hist):
                    ctk.CTkLabel(self.assess_box,
                                 text=f"{h.get('date', '')}  |  {h.get('topic', '')}  →  {h.get('score', 0)}% ({h.get('status', '')})",
                                 text_color="#CDD6F4", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=12, pady=3)

        # dashboard focus graph
        try:
            self._render_dash_focus_chart()
        except Exception:
            pass


    # ==================================================================
    # POST-FOCUS-SESSION INSTANT QUIZ (5 MCQ authenticity check)
    # ==================================================================
    def _launch_post_session_quiz(self, subject, duration_mins, final_conc, timeline, leaves):
        """After a focus block ends, open a 5-mark MCQ on that subject."""
        win = ctk.CTkToplevel(self)
        win.title(f"Post-Session Quiz — {subject[:40]}")
        win.geometry("720x640")
        win.configure(fg_color="#11111B")
        win.transient(self)
        win.lift()
        win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)

        ctk.CTkLabel(win, text="Instant Authenticity Quiz", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#89B4FA").pack(pady=(12, 2))
        ctk.CTkLabel(win, text=f"You just studied: {subject} ({duration_mins} min) · Conc {final_conc:.0f}% · Leaves {leaves}",
                     text_color="#A6ADC8", font=ctk.CTkFont(size=12)).pack(pady=(0, 8))

        # Build 5 MCQs from subject keywords + materials
        topic = subject.split("|")[0].strip() if "|" in subject else subject
        mats = self.user_data.get("study_materials", []) or []
        mat_text = " ".join((m.get("text") or m.get("name") or "")[:800] for m in mats
                            if topic.lower()[:8] in (m.get("name") or "").lower()
                            or topic.lower()[:8] in (m.get("text") or "").lower())
        keywords = [w for w in topic.replace("-", " ").split() if len(w) > 2][:6] or ["concept"]
        stems = [
            f"Which statement best describes a core idea in {topic}?",
            f"In {topic}, what is the most accurate application?",
            f"A common mistake when learning {topic} is:",
            f"Which option strengthens mastery of {topic}?",
            f"For career use of {topic}, which skill matters most?",
        ]
        import random as _rnd
        _rnd.seed(hash(topic + str(duration_mins) + str(int(final_conc))) % 10**8)
        questions = []
        for i, stem in enumerate(stems):
            correct = f"Correct principle related to {keywords[i % len(keywords)]}"
            opts = [
                correct,
                f"Unrelated trivia about {keywords[(i+1) % len(keywords)]}",
                "Memorizing definitions without practice",
                "Ignoring fundamentals and jumping to advanced tools",
            ]
            _rnd.shuffle(opts)
            ans = opts.index(correct)
            questions.append({"q": stem, "options": opts, "answer": ans})

        vars_ = []
        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent", height=380)
        scroll.pack(fill="both", expand=True, padx=12, pady=4)
        for qi, qq in enumerate(questions):
            box = ctk.CTkFrame(scroll, fg_color="#181825", corner_radius=8)
            box.pack(fill="x", pady=6)
            ctk.CTkLabel(box, text=f"{qi+1}. {qq['q']}", wraplength=640, justify="left",
                         text_color="#CDD6F4").pack(anchor="w", padx=10, pady=(8, 4))
            v = ctk.StringVar(value="")
            vars_.append(v)
            for oi, opt in enumerate(qq["options"]):
                ctk.CTkRadioButton(box, text=opt, variable=v, value=str(oi),
                                   text_color="#A6ADC8").pack(anchor="w", padx=18, pady=2)

        result_lbl = ctk.CTkLabel(win, text="", text_color="#A6E3A1", justify="left")
        result_lbl.pack(pady=6)

        def submit():
            correct = 0
            for qi, qq in enumerate(questions):
                try:
                    if int(vars_[qi].get()) == qq["answer"]:
                        correct += 1
                except Exception:
                    pass
            quiz_score = round(100.0 * correct / 5.0, 1)
            # Judgment: real study vs acting
            if final_conc >= 75 and leaves <= 1 and quiz_score >= 60:
                verdict = "STRONG — camera focus + quiz aligned (likely genuine study)"
                color = "#A6E3A1"
            elif final_conc >= 60 and quiz_score >= 40:
                verdict = "DEVELOPING — partial focus; review weak points"
                color = "#FAB387"
            else:
                verdict = "WEAK / POSSIBLE ACTING — low focus or quiz mismatch"
                color = "#F38BA8"
            # Save mastery + quiz history
            status = "Excellent" if quiz_score >= 90 else ("Strong" if quiz_score >= 75 else (
                "Developing" if quiz_score >= 60 else ("Weak" if quiz_score >= 40 else "Critical")))
            self.user_data.setdefault("topic_mastery", {})[topic] = quiz_score
            self.user_data.setdefault("quiz_history", []).append({
                "topic": topic, "score": quiz_score, "status": status,
                "type": "PostSessionMCQ", "correct": correct, "total": 5,
                "session_conc": round(final_conc, 1), "leaves": leaves,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            if quiz_score < 60:
                wt = self.user_data.setdefault("weak_topics", [])
                if topic not in wt:
                    wt.append(topic)
            db.save_user_profile(self.current_user_id, self.user_data)
            result_lbl.configure(
                text=f"Quiz: {correct}/5 = {quiz_score}% ({status})\n"
                     f"Session concentration: {final_conc:.0f}% · Leaves: {leaves}\n"
                     f"Judgment: {verdict}",
                text_color=color)
            self.refresh_ui()

        ctk.CTkButton(win, text="Submit 5-Mark Quiz", fg_color="#89B4FA", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), command=submit).pack(pady=10)

    # ==================================================================
    # ADDITIONAL BRAIN GAMES
    # ==================================================================
    def _open_minesweeper(self):
        import random as _rnd
        win = ctk.CTkToplevel(self)
        win.title("Minesweeper")
        win.geometry("520x600")
        win.configure(fg_color="#11111B")
        win.lift(); win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        lvl_var = ctk.StringVar(value="3")
        ctk.CTkLabel(win, text="Minesweeper — Level sets grid & mine density", text_color="#CDD6F4").pack(pady=6)
        ctk.CTkOptionMenu(win, variable=lvl_var, values=[str(i) for i in range(1, 11)], fg_color="#313244").pack()
        info = ctk.CTkLabel(win, text="Left-click open · avoid mines", text_color="#A6ADC8")
        info.pack()
        board_frame = ctk.CTkFrame(win, fg_color="#181825")
        board_frame.pack(pady=8)
        state = {"grid": None, "mines": set(), "opened": set(), "flags": set(),
                 "t0": None, "over": False, "mistakes": 0}

        def build():
            for w in board_frame.winfo_children():
                w.destroy()
            level = int(lvl_var.get())
            n = 6 + level // 2  # 6..10
            mines_n = max(3, int(n * n * (0.08 + level * 0.02)))
            state["n"] = n
            state["mines"] = set()
            while len(state["mines"]) < mines_n:
                state["mines"].add((_rnd.randrange(n), _rnd.randrange(n)))
            state["opened"] = set()
            state["flags"] = set()
            state["over"] = False
            state["mistakes"] = 0
            state["t0"] = time.time()
            state["btns"] = {}
            for r in range(n):
                for c in range(n):
                    b = ctk.CTkButton(board_frame, text="", width=32, height=32,
                                      fg_color="#313244", command=lambda rr=r, cc=c: open_cell(rr, cc))
                    b.grid(row=r, column=c, padx=1, pady=1)
                    state["btns"][(r, c)] = b
            info.configure(text=f"{n}×{n} · {mines_n} mines · Level {level}")

        def count_adj(r, c):
            n = state["n"]
            cnt = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (r+dr, c+dc) in state["mines"]:
                        cnt += 1
            return cnt

        def open_cell(r, c):
            if state["over"] or (r, c) in state["opened"]:
                return
            if (r, c) in state["mines"]:
                state["over"] = True
                state["mistakes"] += 1
                state["btns"][(r, c)].configure(text="*", fg_color="#F38BA8")
                elapsed = time.time() - state["t0"]
                level = int(lvl_var.get())
                score = max(5, 30 - level * 2)
                self._save_brain_score("Minesweeper", level, score,
                    f"Hit mine · {elapsed:.0f}s · mistakes {state['mistakes']}",
                    metrics={"time_sec": elapsed, "mistakes": state["mistakes"], "success": False})
                info.configure(text="Boom! Mine hit — score saved", text_color="#F38BA8")
                return
            # flood open
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if (rr, cc) in state["opened"] or not (0 <= rr < state["n"] and 0 <= cc < state["n"]):
                    continue
                if (rr, cc) in state["mines"]:
                    continue
                state["opened"].add((rr, cc))
                adj = count_adj(rr, cc)
                state["btns"][(rr, cc)].configure(
                    text=str(adj) if adj else " ", fg_color="#45475A", state="disabled")
                if adj == 0:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            stack.append((rr+dr, cc+dc))
            safe = state["n"] * state["n"] - len(state["mines"])
            if len(state["opened"]) >= safe:
                state["over"] = True
                elapsed = time.time() - state["t0"]
                level = int(lvl_var.get())
                score = min(100, 50 + level * 4 + max(0, 30 - int(elapsed)))
                self._save_brain_score("Minesweeper", level, score,
                    f"Cleared · {elapsed:.0f}s · opened {len(state['opened'])}",
                    metrics={"time_sec": elapsed, "mistakes": 0, "success": True,
                             "opened": len(state["opened"])})
                info.configure(text=f"Cleared! Score {score}%", text_color="#A6E3A1")

        ctk.CTkButton(win, text="New Game", fg_color="#89B4FA", text_color="#11111B", command=build).pack(pady=8)
        build()

    def _open_2048(self):
        import random as _rnd
        win = ctk.CTkToplevel(self)
        win.title("2048")
        win.geometry("420x520")
        win.configure(fg_color="#11111B")
        win.lift(); win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        ctk.CTkLabel(win, text="2048 — Arrow keys / WASD · merge tiles", text_color="#CDD6F4").pack(pady=8)
        grid_f = ctk.CTkFrame(win, fg_color="#181825")
        grid_f.pack(pady=6)
        state = {"board": [[0]*4 for _ in range(4)], "score": 0, "moves": 0, "t0": time.time(), "over": False}
        labels = [[None]*4 for _ in range(4)]
        colors = {0: "#313244", 2: "#45475A", 4: "#585B70", 8: "#FAB387", 16: "#F38BA8",
                  32: "#EBA0AC", 64: "#A6E3A1", 128: "#94E2D5", 256: "#89B4FA", 512: "#B4BEFE",
                  1024: "#CBA6F7", 2048: "#F9E2AF"}

        def spawn():
            empty = [(r, c) for r in range(4) for c in range(4) if state["board"][r][c] == 0]
            if not empty:
                return
            r, c = _rnd.choice(empty)
            state["board"][r][c] = 4 if _rnd.random() < 0.1 else 2

        def draw():
            for r in range(4):
                for c in range(4):
                    v = state["board"][r][c]
                    labels[r][c].configure(text=str(v) if v else "", fg_color=colors.get(v, "#CBA6F7"))

        for r in range(4):
            for c in range(4):
                lb = ctk.CTkLabel(grid_f, text="", width=80, height=80, corner_radius=8,
                                  fg_color="#313244", font=ctk.CTkFont(size=22, weight="bold"))
                lb.grid(row=r, column=c, padx=3, pady=3)
                labels[r][c] = lb

        info = ctk.CTkLabel(win, text="Score: 0", text_color="#A6E3A1")
        info.pack(pady=4)

        def compress_line(line):
            tiles = [x for x in line if x]
            out, sc = [], 0
            i = 0
            while i < len(tiles):
                if i + 1 < len(tiles) and tiles[i] == tiles[i+1]:
                    out.append(tiles[i]*2)
                    sc += tiles[i]*2
                    i += 2
                else:
                    out.append(tiles[i])
                    i += 1
            out += [0] * (4 - len(out))
            return out, sc

        def move(dir_):
            if state["over"]:
                return
            b = state["board"]
            old = [row[:] for row in b]
            sc = 0
            if dir_ in ("Left", "a"):
                for r in range(4):
                    b[r], s = compress_line(b[r]); sc += s
            elif dir_ in ("Right", "d"):
                for r in range(4):
                    rev, s = compress_line(list(reversed(b[r]))); sc += s
                    b[r] = list(reversed(rev))
            elif dir_ in ("Up", "w"):
                for c in range(4):
                    col = [b[r][c] for r in range(4)]
                    col, s = compress_line(col); sc += s
                    for r in range(4):
                        b[r][c] = col[r]
            elif dir_ in ("Down", "s"):
                for c in range(4):
                    col = [b[r][c] for r in range(4)]
                    rev, s = compress_line(list(reversed(col))); sc += s
                    col = list(reversed(rev))
                    for r in range(4):
                        b[r][c] = col[r]
            if b != old:
                state["score"] += sc
                state["moves"] += 1
                spawn()
                draw()
                info.configure(text=f"Score: {state['score']} · Moves: {state['moves']}")
                if not any(0 in row for row in b) and not any(
                    b[r][c] == b[r][c+1] for r in range(4) for c in range(3)) and not any(
                    b[r][c] == b[r+1][c] for r in range(3) for c in range(4)):
                    state["over"] = True
                    elapsed = time.time() - state["t0"]
                    mx = max(max(row) for row in b)
                    score = min(100, int(20 + (mx ** 0.5) * 8 + min(30, state["score"] / 50)))
                    self._save_brain_score("2048", 5, score,
                        f"max {mx} · score {state['score']} · moves {state['moves']} · {elapsed:.0f}s",
                        metrics={"max_tile": mx, "game_score": state["score"],
                                 "moves": state["moves"], "time_sec": elapsed})
                    info.configure(text=f"Game over · cognitive score {score}%", text_color="#FAB387")

        def on_key(e):
            key = e.keysym
            mapping = {"Left": "Left", "Right": "Right", "Up": "Up", "Down": "Down",
                       "a": "a", "d": "d", "w": "w", "s": "s"}
            if key in mapping:
                move(mapping[key])
        win.bind("<Key>", on_key)
        win.focus_set()
        spawn(); spawn(); draw()

        def end_manual():
            if state["over"]:
                return
            state["over"] = True
            elapsed = time.time() - state["t0"]
            mx = max(max(row) for row in state["board"])
            score = min(100, int(20 + (mx ** 0.5) * 8 + min(30, state["score"] / 50)))
            self._save_brain_score("2048", 5, score,
                f"max {mx} · score {state['score']} · moves {state['moves']} · {elapsed:.0f}s",
                metrics={"max_tile": mx, "game_score": state["score"],
                         "moves": state["moves"], "time_sec": elapsed})
            info.configure(text=f"Saved · cognitive score {score}%", text_color="#A6E3A1")
        ctk.CTkButton(win, text="End & Save Score", fg_color="#A6E3A1", text_color="#11111B",
                      command=end_manual).pack(pady=8)

    def _open_nqueens(self):
        win = ctk.CTkToplevel(self)
        win.title("N-Queens")
        win.geometry("480x560")
        win.configure(fg_color="#11111B")
        win.lift(); win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        n_var = ctk.StringVar(value="8")
        ctk.CTkLabel(win, text="N-Queens — place N queens, no attacks", text_color="#CDD6F4").pack(pady=6)
        ctk.CTkOptionMenu(win, variable=n_var, values=["4", "5", "6", "7", "8"], fg_color="#313244").pack()
        board_f = ctk.CTkFrame(win, fg_color="#181825")
        board_f.pack(pady=8)
        state = {"queens": set(), "n": 8, "t0": time.time(), "moves": 0}

        def rebuild():
            for w in board_f.winfo_children():
                w.destroy()
            n = int(n_var.get())
            state["n"] = n
            state["queens"] = set()
            state["t0"] = time.time()
            state["moves"] = 0
            state["btns"] = {}
            for r in range(n):
                for c in range(n):
                    col = "#F0D9B5" if (r+c) % 2 == 0 else "#B58863"
                    b = ctk.CTkButton(board_f, text="", width=40, height=40, fg_color=col,
                                      text_color="#11111B",
                                      command=lambda rr=r, cc=c: toggle(rr, cc))
                    b.grid(row=r, column=c, padx=1, pady=1)
                    state["btns"][(r, c)] = b

        def attacks(r, c):
            for qr, qc in state["queens"]:
                if qr == r or qc == c or abs(qr-r) == abs(qc-c):
                    return True
            return False

        info = ctk.CTkLabel(win, text="", text_color="#A6ADC8")
        info.pack()

        def toggle(r, c):
            state["moves"] += 1
            if (r, c) in state["queens"]:
                state["queens"].discard((r, c))
                n = state["n"]
                col = "#F0D9B5" if (r+c) % 2 == 0 else "#B58863"
                state["btns"][(r, c)].configure(text="", fg_color=col)
            else:
                if attacks(r, c):
                    info.configure(text="Attack! Invalid", text_color="#F38BA8")
                    return
                state["queens"].add((r, c))
                state["btns"][(r, c)].configure(text="♛", fg_color="#89B4FA")
            info.configure(text=f"Queens: {len(state['queens'])}/{state['n']}", text_color="#CDD6F4")
            if len(state["queens"]) == state["n"]:
                elapsed = time.time() - state["t0"]
                level = state["n"]
                score = min(100, 40 + level * 6 + max(0, 25 - int(elapsed / 5)))
                self._save_brain_score("NQueens", level, score,
                    f"solved N={level} · moves {state['moves']} · {elapsed:.0f}s",
                    metrics={"n": level, "moves": state["moves"], "time_sec": elapsed, "success": True})
                info.configure(text=f"Solved! Score {score}%", text_color="#A6E3A1")

        ctk.CTkButton(win, text="New Board", fg_color="#89B4FA", text_color="#11111B", command=rebuild).pack(pady=8)
        rebuild()

    def _open_hanoi(self):
        win = ctk.CTkToplevel(self)
        win.title("Tower of Hanoi")
        win.geometry("520x420")
        win.configure(fg_color="#11111B")
        win.lift(); win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        n_var = ctk.StringVar(value="3")
        ctk.CTkLabel(win, text="Tower of Hanoi — move stack to right peg", text_color="#CDD6F4").pack(pady=6)
        ctk.CTkOptionMenu(win, variable=n_var, values=["3", "4", "5", "6"], fg_color="#313244").pack()
        info = ctk.CTkLabel(win, text="", text_color="#A6ADC8")
        info.pack()
        pegs_f = ctk.CTkFrame(win, fg_color="#181825")
        pegs_f.pack(pady=10, fill="x", padx=20)
        state = {"pegs": [[], [], []], "selected": None, "moves": 0, "t0": time.time(), "n": 3}

        def reset():
            n = int(n_var.get())
            state["n"] = n
            state["pegs"] = [list(range(n, 0, -1)), [], []]
            state["selected"] = None
            state["moves"] = 0
            state["t0"] = time.time()
            draw()

        def draw():
            for w in pegs_f.winfo_children():
                w.destroy()
            for i in range(3):
                col = ctk.CTkFrame(pegs_f, fg_color="#313244", width=140, height=200)
                col.pack(side="left", padx=8, expand=True, fill="y")
                col.pack_propagate(False)
                ctk.CTkLabel(col, text=f"Peg {i+1}", text_color="#89B4FA").pack(pady=4)
                for disk in reversed(state["pegs"][i]):
                    ctk.CTkLabel(col, text="█" * (disk * 2 + 2), text_color="#A6E3A1",
                                 font=ctk.CTkFont(size=10)).pack()
                ctk.CTkButton(col, text="Select / Drop", width=100,
                              command=lambda p=i: click_peg(p)).pack(side="bottom", pady=6)
            info.configure(text=f"Moves: {state['moves']} · optimal {2**state['n']-1}")

        def click_peg(p):
            if state["selected"] is None:
                if state["pegs"][p]:
                    state["selected"] = p
                    info.configure(text=f"Selected peg {p+1}", text_color="#FAB387")
                return
            src = state["selected"]
            state["selected"] = None
            if not state["pegs"][src]:
                return
            disk = state["pegs"][src][-1]
            if state["pegs"][p] and state["pegs"][p][-1] < disk:
                info.configure(text="Invalid — larger on smaller", text_color="#F38BA8")
                return
            state["pegs"][src].pop()
            state["pegs"][p].append(disk)
            state["moves"] += 1
            draw()
            if len(state["pegs"][2]) == state["n"]:
                elapsed = time.time() - state["t0"]
                opt = 2 ** state["n"] - 1
                efficiency = max(0, 100 - (state["moves"] - opt) * 5)
                score = min(100, 40 + state["n"] * 8 + efficiency * 0.3)
                self._save_brain_score("Hanoi", state["n"], score,
                    f"moves {state['moves']}/{opt} · {elapsed:.0f}s",
                    metrics={"moves": state["moves"], "optimal": opt, "time_sec": elapsed, "success": True})
                info.configure(text=f"Solved! Score {score:.0f}%", text_color="#A6E3A1")

        ctk.CTkButton(win, text="Reset", fg_color="#89B4FA", text_color="#11111B", command=reset).pack(pady=8)
        reset()

    def _open_mastermind(self):
        import random as _rnd
        win = ctk.CTkToplevel(self)
        win.title("Mastermind")
        win.geometry("480x520")
        win.configure(fg_color="#11111B")
        win.lift(); win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        colors = ["R", "G", "B", "Y", "P", "O"]
        secret = [_rnd.choice(colors) for _ in range(4)]
        state = {"tries": 0, "t0": time.time(), "max": 10}
        ctk.CTkLabel(win, text="Mastermind — guess 4 colors (R G B Y P O)", text_color="#CDD6F4").pack(pady=8)
        entry = ctk.CTkEntry(win, placeholder_text="e.g. R G B Y", fg_color="#181825")
        entry.pack(pady=4)
        log = ctk.CTkTextbox(win, height=280, fg_color="#181825")
        log.pack(fill="x", padx=12, pady=6)

        def guess():
            parts = entry.get().strip().upper().split()
            if len(parts) != 4 or any(p not in colors for p in parts):
                log.insert("end", "Use 4 colors from R G B Y P O\\n")
                return
            state["tries"] += 1
            exact = sum(a == b for a, b in zip(parts, secret))
            secret_left = [b for a, b in zip(parts, secret) if a != b]
            guess_left = [a for a, b in zip(parts, secret) if a != b]
            color_only = 0
            for g in guess_left:
                if g in secret_left:
                    color_only += 1
                    secret_left.remove(g)
            log.insert("end", f"Try {state['tries']}: {' '.join(parts)} → black {exact}, white {color_only}\\n")
            entry.delete(0, "end")
            if exact == 4:
                elapsed = time.time() - state["t0"]
                score = min(100, 30 + (state["max"] - state["tries"]) * 8 + max(0, 20 - int(elapsed / 3)))
                self._save_brain_score("Mastermind", 5, score,
                    f"cracked in {state['tries']} tries · {elapsed:.0f}s",
                    metrics={"tries": state["tries"], "time_sec": elapsed, "success": True})
                log.insert("end", f"Solved! Score {score}%\\n")
            elif state["tries"] >= state["max"]:
                score = 15
                self._save_brain_score("Mastermind", 5, score,
                    f"failed · secret was {' '.join(secret)}",
                    metrics={"tries": state["tries"], "success": False})
                log.insert("end", f"Out of tries. Secret: {' '.join(secret)}\\n")

        ctk.CTkButton(win, text="Guess", fg_color="#A6E3A1", text_color="#11111B", command=guess).pack(pady=8)

    def _open_connect4(self):
        import random as _rnd
        win = ctk.CTkToplevel(self)
        win.title("Connect Four")
        win.geometry("500x560")
        win.configure(fg_color="#11111B")
        win.lift(); win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        ROWS, COLS = 6, 7
        board = [[0]*COLS for _ in range(ROWS)]
        state = {"board": board, "moves": 0, "t0": time.time(), "over": False}
        ctk.CTkLabel(win, text="Connect Four — you are Red (click column)", text_color="#CDD6F4").pack(pady=6)
        grid_f = ctk.CTkFrame(win, fg_color="#181825")
        grid_f.pack(pady=6)
        cells = [[None]*COLS for _ in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                lb = ctk.CTkLabel(grid_f, text="●", width=48, height=48, fg_color="#313244",
                                  text_color="#313244", font=ctk.CTkFont(size=28))
                lb.grid(row=r, column=c, padx=2, pady=2)
                cells[r][c] = lb
        info = ctk.CTkLabel(win, text="", text_color="#A6ADC8")
        info.pack()

        def draw():
            for r in range(ROWS):
                for c in range(COLS):
                    v = state["board"][r][c]
                    if v == 1:
                        cells[r][c].configure(text_color="#F38BA8")
                    elif v == 2:
                        cells[r][c].configure(text_color="#89B4FA")
                    else:
                        cells[r][c].configure(text_color="#313244")

        def winner(p):
            b = state["board"]
            for r in range(ROWS):
                for c in range(COLS):
                    if b[r][c] != p:
                        continue
                    for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
                        if all(0 <= r+dr*i < ROWS and 0 <= c+dc*i < COLS and b[r+dr*i][c+dc*i] == p for i in range(4)):
                            return True
            return False

        def drop(col, player):
            for r in range(ROWS-1, -1, -1):
                if state["board"][r][col] == 0:
                    state["board"][r][col] = player
                    return True
            return False

        def cpu():
            opts = [c for c in range(COLS) if state["board"][0][c] == 0]
            if not opts:
                return
            # prefer winning / blocking
            for p in (2, 1):
                for c in opts:
                    # simulate
                    for r in range(ROWS-1, -1, -1):
                        if state["board"][r][c] == 0:
                            state["board"][r][c] = p
                            win_p = winner(p)
                            state["board"][r][c] = 0
                            if win_p:
                                drop(c, 2)
                                return
                    break
            drop(_rnd.choice(opts), 2)

        def play_col(c):
            if state["over"]:
                return
            if not drop(c, 1):
                return
            state["moves"] += 1
            draw()
            if winner(1):
                state["over"] = True
                elapsed = time.time() - state["t0"]
                score = min(100, 55 + max(0, 30 - state["moves"]))
                self._save_brain_score("ConnectFour", 5, score,
                    f"You win · moves {state['moves']} · {elapsed:.0f}s",
                    metrics={"moves": state["moves"], "time_sec": elapsed, "result": "win"})
                info.configure(text=f"You win! Score {score}%", text_color="#A6E3A1")
                return
            cpu()
            draw()
            if winner(2):
                state["over"] = True
                elapsed = time.time() - state["t0"]
                score = max(10, 40 - state["moves"])
                self._save_brain_score("ConnectFour", 5, score,
                    f"CPU wins · moves {state['moves']} · {elapsed:.0f}s",
                    metrics={"moves": state["moves"], "time_sec": elapsed, "result": "loss"})
                info.configure(text="CPU wins", text_color="#F38BA8")

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(pady=8)
        for c in range(COLS):
            ctk.CTkButton(btn_row, text=str(c+1), width=48, fg_color="#45475A",
                          command=lambda cc=c: play_col(cc)).pack(side="left", padx=2)

    def _open_nonogram(self):
        """Simple 5x5 nonogram."""
        import random as _rnd
        win = ctk.CTkToplevel(self)
        win.title("Nonogram 5×5")
        win.geometry("420x500")
        win.configure(fg_color="#11111B")
        win.lift(); win.focus_force()
        if not hasattr(self, "_open_game_windows"):
            self._open_game_windows = []
        self._open_game_windows.append(win)
        # generate solution pattern
        sol = [[_rnd.randint(0, 1) for _ in range(5)] for _ in range(5)]
        def runs(line):
            out, cnt = [], 0
            for v in line + [0]:
                if v:
                    cnt += 1
                elif cnt:
                    out.append(cnt); cnt = 0
            return out or [0]
        row_clues = [runs(r) for r in sol]
        col_clues = [runs([sol[r][c] for r in range(5)]) for c in range(5)]
        grid = [[0]*5 for _ in range(5)]
        state = {"t0": time.time(), "moves": 0}
        ctk.CTkLabel(win, text="Nonogram — click cells to match clues", text_color="#CDD6F4").pack(pady=6)
        outer = ctk.CTkFrame(win, fg_color="#181825")
        outer.pack(pady=8)
        btns = [[None]*5 for _ in range(5)]
        # col clues
        for c in range(5):
            ctk.CTkLabel(outer, text=" ".join(map(str, col_clues[c])), width=40,
                         text_color="#89B4FA", font=ctk.CTkFont(size=10)).grid(row=0, column=c+1)
        for r in range(5):
            ctk.CTkLabel(outer, text=" ".join(map(str, row_clues[r])), width=40,
                         text_color="#89B4FA", font=ctk.CTkFont(size=10)).grid(row=r+1, column=0)
            for c in range(5):
                b = ctk.CTkButton(outer, text="", width=40, height=40, fg_color="#313244",
                                  command=lambda rr=r, cc=c: flip(rr, cc))
                b.grid(row=r+1, column=c+1, padx=1, pady=1)
                btns[r][c] = b
        info = ctk.CTkLabel(win, text="", text_color="#A6ADC8")
        info.pack()

        def flip(r, c):
            state["moves"] += 1
            grid[r][c] = 1 - grid[r][c]
            btns[r][c].configure(fg_color="#A6E3A1" if grid[r][c] else "#313244")
            if grid == sol:
                elapsed = time.time() - state["t0"]
                score = min(100, 55 + max(0, 30 - state["moves"]))
                self._save_brain_score("Nonogram", 5, score,
                    f"solved · moves {state['moves']} · {elapsed:.0f}s",
                    metrics={"moves": state["moves"], "time_sec": elapsed, "success": True})
                info.configure(text=f"Solved! Score {score}%", text_color="#A6E3A1")

        ctk.CTkButton(win, text="Show Solution", fg_color="#313244", command=lambda: info.configure(
            text="Solution revealed (no score)", text_color="#FAB387")).pack(pady=8)


    # ==========================================================================
    # CV & RESUME BUILDER (ATS-Friendly + Professional CV)
    # Does not fabricate data — profile or manual only
    # ==========================================================================
    def _build_resume(self):
        """Root container for Resume Builder — method selection home."""
        self.resume_root = ctk.CTkFrame(self.tab_resume, fg_color="transparent")
        self.resume_root.pack(fill="both", expand=True)
        self._resume_show_home()

    def _resume_clear_root(self):
        for w in self.resume_root.winfo_children():
            w.destroy()

    def _resume_show_home(self):
        self._resume_clear_root()
        f = ctk.CTkScrollableFrame(self.resume_root, fg_color="transparent")
        f.pack(fill="both", expand=True, padx=12, pady=8)

        ctk.CTkLabel(f, text="CV & Resume Builder", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#CDD6F4").pack(anchor="w")
        ctk.CTkLabel(
            f,
            text="Create a professional CV or ATS-friendly resume from your Acadexa profile or your own information.",
            text_color="#A6ADC8", font=ctk.CTkFont(size=12), wraplength=700, justify="left",
        ).pack(anchor="w", pady=(2, 14))

        # Document type
        type_row = ctk.CTkFrame(f, fg_color="#181825", corner_radius=12)
        type_row.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(type_row, text="Resume / CV Type", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", padx=14, pady=(12, 4))
        self.resume_doc_type = ctk.StringVar(value="ATS-Friendly Resume")
        ctk.CTkOptionMenu(
            type_row,
            variable=self.resume_doc_type,
            values=["ATS-Friendly Resume", "Professional CV"],
            fg_color="#313244", width=280,
        ).pack(anchor="w", padx=14, pady=(0, 12))

        cards = ctk.CTkFrame(f, fg_color="transparent")
        cards.pack(fill="x")
        cards.grid_columnconfigure((0, 1), weight=1)

        # Card 1 — Profile
        c1 = ctk.CTkFrame(cards, fg_color="#181825", corner_radius=14, border_width=1, border_color="#313244")
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)
        ctk.CTkLabel(c1, text="Create from Acadexa Profile", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", padx=16, pady=(16, 6))
        ctk.CTkLabel(
            c1,
            text="Use your saved academic, skill, project, and career information to automatically build your CV or resume.",
            text_color="#A6ADC8", font=ctk.CTkFont(size=12), wraplength=320, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 12))
        ctk.CTkButton(
            c1, text="Use My Profile", fg_color="#A6E3A1", text_color="#11111B",
            font=ctk.CTkFont(weight="bold"), height=36,
            command=self._resume_from_profile,
        ).pack(anchor="w", padx=16, pady=(0, 16))

        # Card 2 — Manual
        c2 = ctk.CTkFrame(cards, fg_color="#181825", corner_radius=14, border_width=1, border_color="#313244")
        c2.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=4)
        ctk.CTkLabel(c2, text="Enter Information Manually", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#89B4FA").pack(anchor="w", padx=16, pady=(16, 6))
        ctk.CTkLabel(
            c2,
            text="Provide your information through a guided form and generate a new professional CV or resume.",
            text_color="#A6ADC8", font=ctk.CTkFont(size=12), wraplength=320, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 12))
        ctk.CTkButton(
            c2, text="Enter Information", fg_color="#89B4FA", text_color="#11111B",
            font=ctk.CTkFont(weight="bold"), height=36,
            command=self._resume_manual_form,
        ).pack(anchor="w", padx=16, pady=(0, 16))

        # History
        hist = self.user_data.get("resume_history", []) or []
        ctk.CTkLabel(f, text="Resume History", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#CDD6F4").pack(anchor="w", pady=(20, 8))
        if not hist:
            ctk.CTkLabel(f, text="No saved resumes yet. Generate one to see it here.",
                         text_color="#A6ADC8").pack(anchor="w")
        else:
            for i, r in enumerate(reversed(hist[-12:])):
                card = ctk.CTkFrame(f, fg_color="#11111B", corner_radius=8, border_width=1, border_color="#313244")
                card.pack(fill="x", pady=4)
                left = ctk.CTkFrame(card, fg_color="transparent")
                left.pack(side="left", fill="x", expand=True, padx=12, pady=8)
                ctk.CTkLabel(left, text=r.get("name", "Resume"), font=ctk.CTkFont(size=13, weight="bold"),
                             text_color="#CDD6F4").pack(anchor="w")
                ctk.CTkLabel(
                    left,
                    text=f"{r.get('doc_type', '')} · Target: {r.get('target_role', '—')} · {r.get('created', '')}",
                    text_color="#A6ADC8", font=ctk.CTkFont(size=11),
                ).pack(anchor="w")
                btns = ctk.CTkFrame(card, fg_color="transparent")
                btns.pack(side="right", padx=8)
                idx = len(hist) - 1 - i
                ctk.CTkButton(btns, text="View", width=60, fg_color="#313244",
                              command=lambda j=idx: self._resume_view_saved(j)).pack(side="left", padx=2)
                ctk.CTkButton(btns, text="PDF", width=50, fg_color="#89B4FA", text_color="#11111B",
                              command=lambda j=idx: self._resume_download_saved(j)).pack(side="left", padx=2)
                ctk.CTkButton(btns, text="✕", width=32, fg_color="#F38BA8", text_color="#11111B",
                              command=lambda j=idx: self._resume_delete_saved(j)).pack(side="left", padx=2)

    def _resume_collect_from_profile(self):
        """Map Student Intelligence Profile → resume dict. Never invent fields."""
        sp = self.user_data.get("student_profile", {}) or {}
        courses = self.user_data.get("courses", []) or []
        skills = self.user_data.get("skills", []) or []
        projects = self.user_data.get("projects", []) or []
        name = (sp.get("name") or "").strip()
        data = {
            "full_name": name,
            "title": (sp.get("career_goal") or "").strip(),
            "email": (sp.get("email") or "").strip(),
            "phone": (sp.get("phone") or "").strip(),
            "location": (sp.get("location") or "").strip(),
            "linkedin": (sp.get("linkedin") or "").strip(),
            "github": (sp.get("github") or "").strip(),
            "portfolio": (sp.get("portfolio") or "").strip(),
            "photo": (sp.get("profile_pic") or "").strip(),
            "summary": "",
            "objective": (sp.get("career_goal") or "").strip(),
            "education": [],
            "experience": [],
            "projects": [],
            "skills_technical": [],
            "skills_soft": [],
            "certifications": list(self.user_data.get("certifications", []) or []),
            "achievements": list(self.user_data.get("achievements", []) or []),
            "research": list(self.user_data.get("research", []) or []),
            "languages": list(self.user_data.get("languages", []) or []),
            "target_role": (sp.get("career_goal") or "").strip(),
            "target_industry": "",
            "job_description": "",
        }
        # Education from profile
        uni = (sp.get("university") or "").strip()
        deg = (sp.get("degree") or "").strip()
        gpa_raw = str(sp.get("gpa") or "").strip()
        try:
            gpa_ok = gpa_raw and float(gpa_raw) > 0
        except Exception:
            gpa_ok = bool(gpa_raw) and gpa_raw not in ("0", "0.0", "0.00")
        if uni or deg or gpa_ok or sp.get("graduation_year"):
            data["education"].append({
                "degree": deg,
                "university": uni,
                "department": (sp.get("department") or sp.get("major") or "").strip(),
                "start_year": "",
                "grad_year": (sp.get("graduation_year") or "").strip(),
                "gpa": gpa_raw if gpa_ok else "",
                "coursework": ", ".join(
                    c.get("title", "") for c in courses[:8] if c.get("title")
                ),
            })
        # Build honest summary only from known fields (never invent jobs)
        bits = []
        if name and (deg or uni):
            role = (sp.get("career_goal") or "student").strip()
            bits.append(f"{name} is pursuing {deg or 'a degree'}" + (f" at {uni}" if uni else "") + ".")
            if role:
                bits.append(f"Career interest: {role}.")
            dept = (sp.get("department") or sp.get("major") or "").strip()
            if dept:
                bits.append(f"Department: {dept}.")
            if courses:
                titles = [c.get("title") for c in courses[:5] if c.get("title")]
                if titles:
                    bits.append("Relevant coursework includes " + ", ".join(titles) + ".")
            if skills:
                sk = []
                for s in skills[:8]:
                    sk.append(s.get("name") if isinstance(s, dict) else str(s))
                sk = [x for x in sk if x]
                if sk:
                    bits.append("Skills: " + ", ".join(sk) + ".")
        data["summary"] = " ".join(bits)
        # Skills list may be strings or dicts
        for s in skills:
            if isinstance(s, dict):
                label = s.get("name") or s.get("skill") or ""
                kind = (s.get("type") or "technical").lower()
            else:
                label = str(s)
                kind = "technical"
            if not label:
                continue
            if "soft" in kind:
                data["skills_soft"].append(label)
            else:
                data["skills_technical"].append(label)
        # Projects
        for p in projects:
            if isinstance(p, dict):
                data["projects"].append({
                    "name": p.get("name") or p.get("title") or "",
                    "description": p.get("description") or "",
                    "role": p.get("role") or "",
                    "tech": p.get("technologies") or p.get("tech") or "",
                    "duration": p.get("duration") or "",
                    "github": p.get("github") or "",
                    "demo": p.get("demo") or p.get("live") or "",
                    "contributions": p.get("contributions") or p.get("outcome") or "",
                })
            elif str(p).strip():
                data["projects"].append({
                    "name": str(p), "description": "", "role": "", "tech": "",
                    "duration": "", "github": "", "demo": "", "contributions": "",
                })
        return data

    def _resume_from_profile(self):
        data = self._resume_collect_from_profile()
        has_core = bool(data.get("full_name")) and (
            data.get("education") or data.get("skills_technical") or data.get("projects")
            or data.get("title")
        )
        if not has_core:
            self._resume_clear_root()
            f = ctk.CTkFrame(self.resume_root, fg_color="transparent")
            f.pack(fill="both", expand=True, padx=20, pady=30)
            ctk.CTkLabel(
                f, text="Your Acadexa profile does not contain enough information to create a complete resume.",
                text_color="#FAB387", font=ctk.CTkFont(size=14), wraplength=520, justify="left",
            ).pack(anchor="w", pady=8)
            ctk.CTkButton(f, text="Complete My Profile", fg_color="#89B4FA", text_color="#11111B",
                          command=lambda: self.tabview.set("Student Intelligence Profile")).pack(anchor="w", pady=6)
            ctk.CTkButton(f, text="Enter Information Manually", fg_color="#A6E3A1", text_color="#11111B",
                          command=self._resume_manual_form).pack(anchor="w", pady=6)
            ctk.CTkButton(f, text="← Back", fg_color="#313244", command=self._resume_show_home).pack(anchor="w", pady=12)
            return
        self._resume_show_review(data, source="profile")

    def _resume_manual_form(self, prefill=None):
        self._resume_clear_root()
        data = prefill or {
            "full_name": "", "title": "", "email": "", "phone": "", "location": "",
            "linkedin": "", "github": "", "portfolio": "", "photo": "",
            "summary": "", "objective": "",
            "education": [{"degree": "", "university": "", "department": "", "start_year": "",
                           "grad_year": "", "gpa": "", "coursework": ""}],
            "experience": [],
            "projects": [{"name": "", "description": "", "role": "", "tech": "",
                          "duration": "", "github": "", "demo": "", "contributions": ""}],
            "skills_technical": [], "skills_soft": [],
            "certifications": [], "achievements": [], "research": [], "languages": [],
            "target_role": "", "target_industry": "", "job_description": "",
        }
        # Prefill name from profile if empty
        sp = self.user_data.get("student_profile", {})
        if not data.get("full_name") and sp.get("name"):
            data["full_name"] = sp.get("name", "")

        scroll = ctk.CTkScrollableFrame(self.resume_root, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=6)

        ctk.CTkLabel(scroll, text="Manual Resume Information", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#CDD6F4").pack(anchor="w", pady=(0, 4))
        ctk.CTkButton(scroll, text="← Back", width=80, fg_color="#313244",
                      command=self._resume_show_home).pack(anchor="w", pady=(0, 10))

        self._resume_form_vars = {}
        self._resume_list_frames = {}

        def section(title):
            ctk.CTkLabel(scroll, text=title, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color="#89B4FA").pack(anchor="w", pady=(14, 4))

        def field(label, key, placeholder="", height=None):
            ctk.CTkLabel(scroll, text=label, text_color="#CDD6F4", font=ctk.CTkFont(size=12)).pack(anchor="w")
            if height:
                w = ctk.CTkTextbox(scroll, height=height, fg_color="#181825", border_color="#313244")
                w.insert("1.0", data.get(key, "") or "")
                w.pack(fill="x", pady=(0, 6))
            else:
                w = ctk.CTkEntry(scroll, placeholder_text=placeholder, fg_color="#181825", border_color="#313244")
                w.insert(0, data.get(key, "") or "")
                w.pack(fill="x", pady=(0, 6))
            self._resume_form_vars[key] = w
            return w

        section("A. Personal Information")
        field("Full Name *", "full_name", "Your full name")
        field("Professional Title / Target Role", "title", "e.g. Software Engineering Student")
        field("Email *", "email", "you@email.com")
        field("Phone", "phone")
        field("Location", "location", "City, Country")
        field("LinkedIn URL", "linkedin", "https://linkedin.com/in/...")
        field("GitHub URL", "github", "https://github.com/...")
        field("Portfolio URL", "portfolio", "https://...")

        # Photo
        ctk.CTkLabel(scroll, text="Profile Photo (Optional — Professional CV)", text_color="#CDD6F4").pack(anchor="w")
        photo_row = ctk.CTkFrame(scroll, fg_color="#181825", corner_radius=8)
        photo_row.pack(fill="x", pady=(0, 8))
        self._resume_photo_path = data.get("photo") or ""
        self._resume_photo_lbl = ctk.CTkLabel(photo_row, text="No photo", text_color="#A6ADC8")
        self._resume_photo_lbl.pack(side="left", padx=12, pady=10)
        if self._resume_photo_path and os.path.exists(self._resume_photo_path):
            try:
                self._resume_photo_lbl.configure(image=make_circular_image(self._resume_photo_path, size=(48, 48)), text="")
            except Exception:
                pass

        def pick_photo():
            from tkinter import filedialog, messagebox
            p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
            if not p:
                return
            if os.path.getsize(p) > 5 * 1024 * 1024:
                messagebox.showwarning("Photo", "Please use an image under 5 MB.")
                return
            self._resume_photo_path = p
            self._resume_photo_lbl.configure(image=make_circular_image(p, size=(48, 48)), text="")

        def clear_photo():
            self._resume_photo_path = ""
            self._resume_photo_lbl.configure(image=None, text="No photo")

        ctk.CTkButton(photo_row, text="Upload Photo", fg_color="#89B4FA", text_color="#11111B",
                      command=pick_photo).pack(side="left", padx=4)
        ctk.CTkButton(photo_row, text="Remove", fg_color="#313244", command=clear_photo).pack(side="left", padx=4)

        section("B. Professional Summary")
        field("Professional Summary", "summary", height=80)
        field("Career Objective", "objective", height=60)

        def gen_summary():
            name = self._resume_form_vars["full_name"].get().strip() or "Candidate"
            title = self._resume_form_vars["title"].get().strip() or "student"
            obj = self._resume_form_vars["objective"].get("1.0", "end").strip() if hasattr(
                self._resume_form_vars["objective"], "get") else ""
            # Only use what user typed — no invention
            text = (
                f"{name} is a motivated {title} seeking opportunities to apply academic knowledge "
                f"and grow professionally."
            )
            if obj:
                text = obj if len(obj) > 40 else f"{text} {obj}"
            self._resume_form_vars["summary"].delete("1.0", "end")
            self._resume_form_vars["summary"].insert("1.0", text)

        ctk.CTkButton(scroll, text="Generate Summary (from your fields only)", fg_color="#313244",
                      command=gen_summary).pack(anchor="w", pady=4)

        section("C. Education")
        edu_box = ctk.CTkFrame(scroll, fg_color="#11111B", corner_radius=8)
        edu_box.pack(fill="x", pady=4)
        self._resume_list_frames["education"] = edu_box
        self._resume_edu_entries = []

        def add_edu(pre=None):
            pre = pre or {}
            row = ctk.CTkFrame(edu_box, fg_color="#181825", corner_radius=6)
            row.pack(fill="x", padx=6, pady=4)
            ents = {}
            for lab, k, ph in [
                ("Degree", "degree", "B.Sc. in CSE"),
                ("University", "university", "University name"),
                ("Department", "department", "CSE"),
                ("Grad Year", "grad_year", "2026"),
                ("GPA", "gpa", "3.5"),
            ]:
                ctk.CTkLabel(row, text=lab, text_color="#A6ADC8", width=90, anchor="w").pack(side="left", padx=2)
                e = ctk.CTkEntry(row, width=110, placeholder_text=ph, fg_color="#11111B")
                e.insert(0, pre.get(k, ""))
                e.pack(side="left", padx=2, pady=6)
                ents[k] = e
            ents["coursework"] = ctk.CTkEntry(row, width=140, placeholder_text="Coursework", fg_color="#11111B")
            ents["coursework"].insert(0, pre.get("coursework", ""))
            ents["coursework"].pack(side="left", padx=2)
            ents["start_year"] = ctk.CTkEntry(row, width=60, placeholder_text="Start", fg_color="#11111B")
            ents["start_year"].insert(0, pre.get("start_year", ""))
            ents["start_year"].pack(side="left", padx=2)

            def rm():
                row.destroy()
                if row in [x[0] for x in self._resume_edu_entries]:
                    self._resume_edu_entries = [x for x in self._resume_edu_entries if x[0] is not row]

            ctk.CTkButton(row, text="✕", width=28, fg_color="#F38BA8", text_color="#11111B", command=rm).pack(side="right", padx=4)
            self._resume_edu_entries.append((row, ents))

        for ed in data.get("education") or [{}]:
            add_edu(ed)
        ctk.CTkButton(scroll, text="+ Add Education", fg_color="#313244", command=lambda: add_edu()).pack(anchor="w", pady=4)

        section("D. Work Experience (optional)")
        ctk.CTkLabel(scroll, text="Leave empty if you have no professional experience yet.",
                     text_color="#A6ADC8", font=ctk.CTkFont(size=11)).pack(anchor="w")
        exp_box = ctk.CTkFrame(scroll, fg_color="#11111B", corner_radius=8)
        exp_box.pack(fill="x", pady=4)
        self._resume_exp_entries = []

        def add_exp(pre=None):
            pre = pre or {}
            row = ctk.CTkFrame(exp_box, fg_color="#181825", corner_radius=6)
            row.pack(fill="x", padx=6, pady=4)
            ents = {}
            grid = ctk.CTkFrame(row, fg_color="transparent")
            grid.pack(fill="x", padx=4, pady=4)
            for lab, k in [("Job Title", "title"), ("Organization", "org"), ("Type", "emp_type"),
                           ("Location", "location"), ("Start", "start"), ("End", "end")]:
                ctk.CTkLabel(grid, text=lab, text_color="#A6ADC8", width=80, anchor="w").pack(side="left")
                e = ctk.CTkEntry(grid, width=100, fg_color="#11111B")
                e.insert(0, pre.get(k, ""))
                e.pack(side="left", padx=2)
                ents[k] = e
            resp = ctk.CTkTextbox(row, height=50, fg_color="#11111B")
            resp.insert("1.0", pre.get("responsibilities", ""))
            resp.pack(fill="x", padx=6, pady=2)
            ents["responsibilities"] = resp
            ach = ctk.CTkTextbox(row, height=40, fg_color="#11111B")
            ach.insert("1.0", pre.get("achievements", ""))
            ach.pack(fill="x", padx=6, pady=2)
            ents["achievements"] = ach

            def rm():
                row.destroy()
                self._resume_exp_entries = [x for x in self._resume_exp_entries if x[0] is not row]
            ctk.CTkButton(row, text="Remove Experience", fg_color="#F38BA8", text_color="#11111B",
                          command=rm).pack(anchor="e", padx=6, pady=4)
            self._resume_exp_entries.append((row, ents))

        for ex in data.get("experience") or []:
            add_exp(ex)
        ctk.CTkButton(scroll, text="+ Add Experience", fg_color="#313244", command=lambda: add_exp()).pack(anchor="w", pady=4)

        section("E. Projects")
        proj_box = ctk.CTkFrame(scroll, fg_color="#11111B", corner_radius=8)
        proj_box.pack(fill="x", pady=4)
        self._resume_proj_entries = []

        def add_proj(pre=None):
            pre = pre or {}
            row = ctk.CTkFrame(proj_box, fg_color="#181825", corner_radius=6)
            row.pack(fill="x", padx=6, pady=4)
            ents = {}
            for lab, k, ph in [("Name *", "name", "Project name"), ("Role", "role", "Developer"),
                               ("Tech", "tech", "Python, React"), ("Duration", "duration", "3 months"),
                               ("GitHub", "github", "https://..."), ("Demo", "demo", "https://...")]:
                ctk.CTkLabel(row, text=lab, text_color="#A6ADC8").pack(anchor="w", padx=6)
                e = ctk.CTkEntry(row, placeholder_text=ph, fg_color="#11111B")
                e.insert(0, pre.get(k, ""))
                e.pack(fill="x", padx=6, pady=2)
                ents[k] = e
            for lab, k in [("Description", "description"), ("Contributions / Outcome", "contributions")]:
                ctk.CTkLabel(row, text=lab, text_color="#A6ADC8").pack(anchor="w", padx=6)
                t = ctk.CTkTextbox(row, height=45, fg_color="#11111B")
                t.insert("1.0", pre.get(k, ""))
                t.pack(fill="x", padx=6, pady=2)
                ents[k] = t

            def rm():
                row.destroy()
                self._resume_proj_entries = [x for x in self._resume_proj_entries if x[0] is not row]
            ctk.CTkButton(row, text="Remove Project", fg_color="#F38BA8", text_color="#11111B",
                          command=rm).pack(anchor="e", padx=6, pady=4)
            self._resume_proj_entries.append((row, ents))

        for pr in data.get("projects") or [{}]:
            add_proj(pr)
        ctk.CTkButton(scroll, text="+ Add Project", fg_color="#313244", command=lambda: add_proj()).pack(anchor="w", pady=4)

        section("F. Skills")
        field("Technical Skills (comma-separated)", "skills_tech_str",
              placeholder="Python, SQL, Machine Learning, Git")
        # prefill skills
        if data.get("skills_technical"):
            self._resume_form_vars["skills_tech_str"].delete(0, "end")
            self._resume_form_vars["skills_tech_str"].insert(0, ", ".join(data["skills_technical"]))
        field("Soft Skills (comma-separated)", "skills_soft_str", placeholder="Communication, Teamwork")
        if data.get("skills_soft"):
            self._resume_form_vars["skills_soft_str"].delete(0, "end")
            self._resume_form_vars["skills_soft_str"].insert(0, ", ".join(data["skills_soft"]))

        def import_skills():
            skills = self.user_data.get("skills", []) or []
            tech = []
            for s in skills:
                tech.append(s.get("name") if isinstance(s, dict) else str(s))
            if tech:
                cur = self._resume_form_vars["skills_tech_str"].get().strip()
                merged = ", ".join(filter(None, [cur] + tech))
                self._resume_form_vars["skills_tech_str"].delete(0, "end")
                self._resume_form_vars["skills_tech_str"].insert(0, merged)
        ctk.CTkButton(scroll, text="Import Skills from Acadexa Profile", fg_color="#313244",
                      command=import_skills).pack(anchor="w", pady=4)

        section("G–K. Certifications, Achievements, Languages (optional)")
        field("Certifications (one per line: Name | Org | Date | URL)", "certs_text", height=60)
        field("Achievements (one per line)", "achievements_text", height=50)
        field("Languages (e.g. English — Professional)", "languages_text", height=40)

        section("Target Job / Career (optional)")
        field("Target Job Title", "target_role", "Software Engineer Intern")
        field("Target Industry", "target_industry", "Technology")
        field("Job Description (paste full JD for tailoring)", "job_description", height=100)

        def analyze_jd():
            jd = self._resume_form_vars["job_description"].get("1.0", "end").strip()
            res_bits = []
            res_bits.append(self._resume_form_vars["skills_tech_str"].get())
            res_bits.append(self._resume_form_vars.get("summary").get("1.0", "end") if hasattr(
                self._resume_form_vars.get("summary"), "get") else "")
            resume_txt = " ".join(str(x) for x in res_bits)
            if not jd:
                return
            score, matched, missing = job_matcher.match(resume_txt, jd)
            from tkinter import messagebox
            messagebox.showinfo(
                "Job Description Analysis",
                f"Keyword alignment (estimate): {score}%\n\n"
                f"Matched: {', '.join(matched[:15]) or 'None'}\n\n"
                f"Missing keywords to consider (do not invent skills): {', '.join(missing[:15]) or 'None'}\n\n"
                f"This is not a guarantee of ATS success.",
            )
        ctk.CTkButton(scroll, text="Analyze Job Description", fg_color="#FAB387", text_color="#11111B",
                      command=analyze_jd).pack(anchor="w", pady=6)

        def collect_and_review():
            d = self._resume_gather_form()
            errs = []
            if not d.get("full_name"):
                errs.append("Please enter your full name.")
            email = d.get("email") or ""
            if not email or "@" not in email:
                errs.append("Please enter a valid email address.")
            if errs:
                from tkinter import messagebox
                messagebox.showwarning("Validation", "\n".join(errs))
                return
            self._resume_show_review(d, source="manual")

        ctk.CTkButton(scroll, text="Review & Continue →", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), height=40, command=collect_and_review).pack(fill="x", pady=16)

    def _resume_gather_form(self):
        def gv(key):
            w = self._resume_form_vars.get(key)
            if w is None:
                return ""
            if isinstance(w, ctk.CTkTextbox):
                return w.get("1.0", "end").strip()
            return w.get().strip()

        def parse_lines(text, seps=True):
            rows = []
            for line in (text or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                if seps and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    rows.append({"name": parts[0] if parts else "", "org": parts[1] if len(parts) > 1 else "",
                                 "date": parts[2] if len(parts) > 2 else "", "url": parts[3] if len(parts) > 3 else ""})
                else:
                    rows.append(line)
            return rows

        edu = []
        for _, ents in getattr(self, "_resume_edu_entries", []):
            try:
                item = {k: ents[k].get().strip() for k in ents}
            except Exception:
                continue
            if any(item.values()):
                edu.append(item)

        exp = []
        for _, ents in getattr(self, "_resume_exp_entries", []):
            item = {}
            for k, w in ents.items():
                item[k] = w.get("1.0", "end").strip() if isinstance(w, ctk.CTkTextbox) else w.get().strip()
            if item.get("title") or item.get("org"):
                exp.append(item)

        projs = []
        for _, ents in getattr(self, "_resume_proj_entries", []):
            item = {}
            for k, w in ents.items():
                item[k] = w.get("1.0", "end").strip() if isinstance(w, ctk.CTkTextbox) else w.get().strip()
            if item.get("name"):
                projs.append(item)

        tech = [s.strip() for s in gv("skills_tech_str").split(",") if s.strip()]
        soft = [s.strip() for s in gv("skills_soft_str").split(",") if s.strip()]
        certs = parse_lines(gv("certs_text"))
        ach = parse_lines(gv("achievements_text"), seps=False)
        langs = parse_lines(gv("languages_text"), seps=False)

        return {
            "full_name": gv("full_name"),
            "title": gv("title"),
            "email": gv("email"),
            "phone": gv("phone"),
            "location": gv("location"),
            "linkedin": gv("linkedin"),
            "github": gv("github"),
            "portfolio": gv("portfolio"),
            "photo": getattr(self, "_resume_photo_path", "") or "",
            "summary": gv("summary"),
            "objective": gv("objective"),
            "education": edu,
            "experience": exp,
            "projects": projs,
            "skills_technical": tech,
            "skills_soft": soft,
            "certifications": certs,
            "achievements": ach,
            "research": [],
            "languages": langs,
            "target_role": gv("target_role"),
            "target_industry": gv("target_industry"),
            "job_description": gv("job_description"),
        }

    def _resume_show_review(self, data, source="manual"):
        self._resume_draft = data
        self._resume_clear_root()
        scroll = ctk.CTkScrollableFrame(self.resume_root, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=8)

        ctk.CTkLabel(scroll, text="Review Information Before Generation",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color="#CDD6F4").pack(anchor="w")
        ctk.CTkLabel(scroll, text=f"Source: {source} · Type: {self.resume_doc_type.get()}",
                     text_color="#A6ADC8").pack(anchor="w", pady=(0, 10))

        preview = self._resume_format_text(data, self.resume_doc_type.get())
        box = ctk.CTkTextbox(scroll, height=420, fg_color="#181825", border_color="#313244",
                             font=ctk.CTkFont(family="Consolas", size=12))
        box.pack(fill="both", expand=True, pady=6)
        box.insert("1.0", preview)
        box.configure(state="disabled")
        self._resume_preview_box = box

        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=12)
        ctk.CTkButton(row, text="← Edit", fg_color="#313244",
                      command=lambda: self._resume_manual_form(prefill=data)).pack(side="left", padx=4)
        ctk.CTkButton(row, text="Generate Document", fg_color="#A6E3A1", text_color="#11111B",
                      font=ctk.CTkFont(weight="bold"), height=36,
                      command=lambda: self._resume_generate(data)).pack(side="left", padx=4)
        ctk.CTkButton(row, text="Home", fg_color="#313244", command=self._resume_show_home).pack(side="left", padx=4)

    def _resume_format_text(self, data, doc_type="ATS-Friendly Resume"):
        lines = []
        name = data.get("full_name") or "Name"
        lines.append(name.upper())
        if data.get("title"):
            lines.append(data["title"])
        contact = " | ".join(filter(None, [
            data.get("email"), data.get("phone"), data.get("location"),
            data.get("linkedin"), data.get("github"), data.get("portfolio"),
        ]))
        if contact:
            lines.append(contact)
        lines.append("=" * 60)

        if data.get("summary") or data.get("objective"):
            lines.append("\nPROFESSIONAL SUMMARY")
            lines.append("-" * 40)
            lines.append(data.get("summary") or data.get("objective") or "")

        tech = data.get("skills_technical") or []
        soft = data.get("skills_soft") or []
        if tech or soft:
            lines.append("\nSKILLS")
            lines.append("-" * 40)
            if tech:
                lines.append("Technical: " + ", ".join(tech))
            if soft:
                lines.append("Soft: " + ", ".join(soft))

        if data.get("experience"):
            lines.append("\nEXPERIENCE")
            lines.append("-" * 40)
            for e in data["experience"]:
                head = " · ".join(filter(None, [e.get("title"), e.get("org"), e.get("emp_type")]))
                dates = " – ".join(filter(None, [e.get("start"), e.get("end") or "Present"]))
                lines.append(f"{head} ({dates})" if dates else head)
                if e.get("location"):
                    lines.append(f"  {e['location']}")
                for bullet in (e.get("responsibilities") or "").splitlines():
                    if bullet.strip():
                        lines.append(f"  • {bullet.strip()}")
                for bullet in (e.get("achievements") or "").splitlines():
                    if bullet.strip():
                        lines.append(f"  • {bullet.strip()}")

        if data.get("projects"):
            lines.append("\nPROJECTS")
            lines.append("-" * 40)
            for p in data["projects"]:
                if not p.get("name"):
                    continue
                lines.append(p["name"] + (f" — {p['role']}" if p.get("role") else ""))
                if p.get("tech"):
                    lines.append(f"  Technologies: {p['tech']}")
                if p.get("description"):
                    for b in p["description"].splitlines():
                        if b.strip():
                            lines.append(f"  • {b.strip()}")
                if p.get("contributions"):
                    for b in p["contributions"].splitlines():
                        if b.strip():
                            lines.append(f"  • {b.strip()}")
                links = " | ".join(filter(None, [p.get("github"), p.get("demo")]))
                if links:
                    lines.append(f"  {links}")

        if data.get("education"):
            lines.append("\nEDUCATION")
            lines.append("-" * 40)
            for e in data["education"]:
                head = " · ".join(filter(None, [e.get("degree"), e.get("university"), e.get("department")]))
                if head:
                    lines.append(head)
                gpa_v = str(e.get("gpa") or "").strip()
                try:
                    show_gpa = gpa_v and float(gpa_v) > 0
                except Exception:
                    show_gpa = gpa_v and gpa_v not in ("0", "0.0", "0.00")
                meta = " · ".join(filter(None, [
                    e.get("grad_year") and f"Grad: {e['grad_year']}",
                    show_gpa and f"GPA: {gpa_v}",
                ]))
                if meta:
                    lines.append(f"  {meta}")
                if e.get("coursework"):
                    lines.append(f"  Coursework: {e['coursework']}")

        if data.get("certifications"):
            lines.append("\nCERTIFICATIONS")
            lines.append("-" * 40)
            for c in data["certifications"]:
                if isinstance(c, dict):
                    lines.append(" · ".join(filter(None, [c.get("name"), c.get("org"), c.get("date")])))
                else:
                    lines.append(str(c))

        if data.get("achievements"):
            lines.append("\nACHIEVEMENTS")
            lines.append("-" * 40)
            for a in data["achievements"]:
                lines.append(f"• {a}" if isinstance(a, str) else f"• {a.get('name', a)}")

        if data.get("languages"):
            lines.append("\nLANGUAGES")
            lines.append("-" * 40)
            for L in data["languages"]:
                lines.append(f"• {L}" if isinstance(L, str) else str(L))

        if doc_type.startswith("Professional") and data.get("photo"):
            lines.append("\n[Profile photo attached in PDF export]")

        return "\n".join(lines)

    def _resume_generate(self, data):
        self._resume_clear_root()
        f = ctk.CTkFrame(self.resume_root, fg_color="transparent")
        f.pack(fill="both", expand=True, padx=20, pady=20)
        ctk.CTkLabel(f, text="Creating Your Resume...", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color="#89B4FA").pack(pady=10)
        prog = ctk.CTkProgressBar(f, width=400)
        prog.pack(pady=8)
        prog.set(0)
        status = ctk.CTkLabel(f, text="", text_color="#A6ADC8")
        status.pack()

        steps = [
            "Collecting information",
            "Organizing education",
            "Analyzing skills",
            "Formatting projects",
            "Optimizing content",
            "Checking ATS compatibility",
            "Generating document",
        ]
        self._resume_gen_i = 0

        def tick():
            i = self._resume_gen_i
            if i < len(steps):
                status.configure(text=f"✓ {steps[i]}")
                prog.set((i + 1) / len(steps))
                self._resume_gen_i += 1
                self.after(280, tick)
            else:
                self._resume_show_result(data)

        tick()

    def _resume_show_result(self, data):
        doc_type = self.resume_doc_type.get()
        text = self._resume_format_text(data, doc_type)
        # Save history (text only — no fabrication)
        entry = {
            "name": f"{data.get('full_name') or 'Resume'} — {data.get('target_role') or doc_type}",
            "doc_type": doc_type,
            "target_role": data.get("target_role") or "",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data": data,
            "text": text,
            "ats_status": "Not analyzed",
        }
        self.user_data.setdefault("resume_history", []).append(entry)
        self.user_data["resume_draft"] = data
        db.save_user_profile(self.current_user_id, self.user_data)
        self._resume_last = entry

        self._resume_clear_root()
        scroll = ctk.CTkScrollableFrame(self.resume_root, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=8)

        ctk.CTkLabel(scroll, text="Your Resume is Ready!", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#A6E3A1").pack(anchor="w", pady=(0, 6))

        # A4-like preview
        page = ctk.CTkFrame(scroll, fg_color="#FAFAFA", corner_radius=4, border_width=1, border_color="#313244")
        page.pack(fill="x", pady=8)
        preview = ctk.CTkTextbox(page, height=480, fg_color="#FAFAFA", text_color="#11111B",
                                 font=ctk.CTkFont(family="Times", size=12), border_width=0)
        preview.pack(fill="both", expand=True, padx=24, pady=20)
        preview.insert("1.0", text)
        preview.configure(state="disabled")

        actions = ctk.CTkFrame(scroll, fg_color="transparent")
        actions.pack(fill="x", pady=10)

        def download_pdf():
            self._resume_export_pdf(data, text, doc_type)

        def analyze_ats():
            from tkinter import messagebox
            jd = (data.get("job_description") or "").strip()
            if not jd:
                win = ctk.CTkToplevel(self)
                win.title("Paste Target Job Description")
                win.geometry("560x420")
                win.configure(fg_color="#11111B")
                win.transient(self)
                win.lift()
                ctk.CTkLabel(
                    win,
                    text="Paste the full job description to estimate keyword alignment with your resume.",
                    text_color="#CDD6F4", wraplength=500, justify="left",
                ).pack(padx=12, pady=10)
                tb = ctk.CTkTextbox(win, height=260, fg_color="#181825")
                tb.pack(fill="both", expand=True, padx=12, pady=6)
                out = {"jd": ""}

                def ok():
                    out["jd"] = tb.get("1.0", "end").strip()
                    win.destroy()

                ctk.CTkButton(win, text="Analyze with this JD", fg_color="#A6E3A1",
                              text_color="#11111B", command=ok).pack(pady=10)
                win.wait_window()
                jd = out["jd"]
                if not jd:
                    messagebox.showinfo("ATS Analyze", "No job description provided.")
                    return
                data["job_description"] = jd
            score, matched, missing = job_matcher.match(text, jd)
            checks = []
            if data.get("full_name") and data.get("email"):
                checks.append("Contact block present")
            elif data.get("full_name"):
                checks.append("Name present (add email for stronger ATS parse)")
            if data.get("education"):
                checks.append("Education section present")
            if data.get("skills_technical") or data.get("skills_soft"):
                checks.append("Skills listed")
            if data.get("projects") or data.get("experience"):
                checks.append("Projects/Experience present")
            if data.get("summary") or data.get("objective"):
                checks.append("Summary present")
            entry["ats_status"] = f"Alignment ~{score}%"
            entry["data"] = data
            db.save_user_profile(self.current_user_id, self.user_data)
            messagebox.showinfo(
                "ATS Analysis (Estimate)",
                f"Estimated Job Alignment Score: {score}%\n\n"
                f"Structure checks:\n- " + ("\n- ".join(checks) if checks else "Sparse resume") + "\n\n"
                f"Matched keywords: {', '.join(matched[:12]) or 'None'}\n\n"
                f"Missing keywords (add only if true for you):\n{', '.join(missing[:12]) or 'None'}\n\n"
                f"This does NOT guarantee passing any ATS. Do not keyword-stuff.",
            )

        for label, cmd, color in [
            ("Edit Resume", lambda: self._resume_manual_form(prefill=data), "#313244"),
            ("Regenerate", lambda: self._resume_generate(data), "#313244"),
            ("Download PDF", download_pdf, "#89B4FA"),
            ("Analyze ATS", analyze_ats, "#FAB387"),
            ("Create New", self._resume_show_home, "#313244"),
        ]:
            ctk.CTkButton(actions, text=label, fg_color=color,
                          text_color="#11111B" if color != "#313244" else "#CDD6F4",
                          command=cmd).pack(side="left", padx=4)

    def _resume_export_pdf(self, data, text, doc_type):
        from tkinter import filedialog, messagebox
        default = f"{(data.get('full_name') or 'Resume').replace(' ', '_')}_{doc_type.replace(' ', '_')}.pdf"
        path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=default,
                                            filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.lib import colors

            def esc(s):
                """Escape user text for ReportLab Paragraph (keep our own tags separate)."""
                return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            def bold_line(s):
                return f"<b>{esc(s)}</b>"

            def body_line(s):
                return esc(s)

            doc = SimpleDocTemplate(
                path, pagesize=A4,
                leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                topMargin=0.55 * inch, bottomMargin=0.55 * inch,
            )
            styles = getSampleStyleSheet()
            name_style = ParagraphStyle(
                "RName", parent=styles["Heading1"], fontSize=18, alignment=TA_CENTER,
                textColor=colors.HexColor("#111111"), spaceAfter=2, leading=22,
            )
            sub_style = ParagraphStyle(
                "RSub", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
                textColor=colors.HexColor("#333333"), spaceAfter=3, leading=13,
            )
            h_style = ParagraphStyle(
                "RH", parent=styles["Heading2"], fontSize=11, spaceBefore=11, spaceAfter=3,
                textColor=colors.HexColor("#111111"), fontName="Helvetica-Bold",
            )
            body = ParagraphStyle(
                "RBody", parent=styles["Normal"], fontSize=10, leading=13,
                textColor=colors.HexColor("#222222"),
            )
            bullet_style = ParagraphStyle(
                "RBullet", parent=body, leftIndent=12, bulletIndent=0, spaceAfter=1,
            )

            story = []
            story.append(Paragraph(esc(data.get("full_name") or "Resume"), name_style))
            if data.get("title"):
                story.append(Paragraph(esc(data["title"]), sub_style))
            contact = "  |  ".join(filter(None, [
                data.get("email"), data.get("phone"), data.get("location"),
            ]))
            if contact:
                story.append(Paragraph(esc(contact), sub_style))
            links = "  |  ".join(filter(None, [
                data.get("linkedin"), data.get("github"), data.get("portfolio"),
            ]))
            if links:
                story.append(Paragraph(esc(links), sub_style))
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#333333")))

            def add_section(title, paragraphs):
                """paragraphs: list of (style_name, text) where text is already safe XML."""
                if not paragraphs:
                    return
                story.append(Paragraph(esc(title), h_style))
                story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#999999")))
                for kind, raw in paragraphs:
                    if not raw:
                        continue
                    if kind == "bold":
                        story.append(Paragraph(bold_line(raw), body))
                    elif kind == "bullet":
                        story.append(Paragraph("• " + body_line(raw), bullet_style))
                    else:
                        story.append(Paragraph(body_line(raw), body))
                story.append(Spacer(1, 4))

            # Summary
            summary = (data.get("summary") or data.get("objective") or "").strip()
            if summary:
                add_section("PROFESSIONAL SUMMARY", [("body", summary)])

            # Skills
            skill_paras = []
            if data.get("skills_technical"):
                skill_paras.append(("body", "Technical: " + ", ".join(data["skills_technical"])))
            if data.get("skills_soft"):
                skill_paras.append(("body", "Soft: " + ", ".join(data["skills_soft"])))
            add_section("SKILLS", skill_paras)

            # Experience
            exp_paras = []
            for e in data.get("experience") or []:
                head = " · ".join(filter(None, [e.get("title"), e.get("org"), e.get("emp_type")]))
                dates = " – ".join(filter(None, [e.get("start"), e.get("end") or "Present"]))
                if head:
                    exp_paras.append(("bold", f"{head}" + (f"  ({dates})" if dates else "")))
                if e.get("location"):
                    exp_paras.append(("body", e["location"]))
                for b in (e.get("responsibilities") or "").splitlines():
                    if b.strip():
                        exp_paras.append(("bullet", b.strip()))
                for b in (e.get("achievements") or "").splitlines():
                    if b.strip():
                        exp_paras.append(("bullet", b.strip()))
            add_section("EXPERIENCE", exp_paras)

            # Projects
            proj_paras = []
            for p in data.get("projects") or []:
                if not (p.get("name") or "").strip():
                    continue
                head = p["name"] + (f" — {p['role']}" if p.get("role") else "")
                proj_paras.append(("bold", head))
                if p.get("tech"):
                    proj_paras.append(("body", "Technologies: " + p["tech"]))
                for b in (p.get("description") or "").splitlines():
                    if b.strip():
                        proj_paras.append(("bullet", b.strip()))
                for b in (p.get("contributions") or "").splitlines():
                    if b.strip():
                        proj_paras.append(("bullet", b.strip()))
                link = " | ".join(filter(None, [p.get("github"), p.get("demo")]))
                if link:
                    proj_paras.append(("body", link))
            add_section("PROJECTS", proj_paras)

            # Education — skip GPA 0.0
            edu_paras = []
            for e in data.get("education") or []:
                head = " · ".join(filter(None, [e.get("degree"), e.get("university"), e.get("department")]))
                if head:
                    edu_paras.append(("bold", head))
                gpa_v = str(e.get("gpa") or "").strip()
                try:
                    show_gpa = gpa_v and float(gpa_v) > 0
                except Exception:
                    show_gpa = gpa_v and gpa_v not in ("0", "0.0", "0.00")
                meta = " · ".join(filter(None, [
                    e.get("grad_year") and f"Graduation: {e['grad_year']}",
                    show_gpa and f"GPA: {gpa_v}",
                ]))
                if meta:
                    edu_paras.append(("body", meta))
                if e.get("coursework"):
                    edu_paras.append(("body", "Coursework: " + e["coursework"]))
            add_section("EDUCATION", edu_paras)

            cert_paras = []
            for c in data.get("certifications") or []:
                if isinstance(c, dict):
                    line = " · ".join(filter(None, [c.get("name"), c.get("org"), c.get("date")]))
                else:
                    line = str(c)
                if line.strip():
                    cert_paras.append(("bullet", line))
            add_section("CERTIFICATIONS", cert_paras)

            ach_paras = []
            for a in data.get("achievements") or []:
                line = a if isinstance(a, str) else str(a)
                if line.strip():
                    ach_paras.append(("bullet", line.strip()))
            add_section("ACHIEVEMENTS", ach_paras)

            lang_paras = []
            for L in data.get("languages") or []:
                line = L if isinstance(L, str) else str(L)
                if line.strip():
                    lang_paras.append(("bullet", line.strip()))
            add_section("LANGUAGES", lang_paras)

            doc.build(story)
            messagebox.showinfo("PDF", f"Saved:\n{path}")
        except ImportError:
            txt_path = path if path.endswith(".txt") else path.rsplit(".", 1)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(text)
            messagebox.showinfo(
                "Export",
                f"reportlab not installed — saved text:\n{txt_path}\n\npip install reportlab",
            )
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _resume_view_saved(self, idx):
        hist = self.user_data.get("resume_history", [])
        if idx < 0 or idx >= len(hist):
            return
        entry = hist[idx]
        self._resume_show_result(entry.get("data") or {})

    def _resume_download_saved(self, idx):
        hist = self.user_data.get("resume_history", [])
        if idx < 0 or idx >= len(hist):
            return
        entry = hist[idx]
        self._resume_export_pdf(entry.get("data") or {}, entry.get("text") or "", entry.get("doc_type") or "ATS-Friendly Resume")

    def _resume_delete_saved(self, idx):
        hist = self.user_data.get("resume_history", [])
        if 0 <= idx < len(hist):
            hist.pop(idx)
            db.save_user_profile(self.current_user_id, self.user_data)
            self._resume_show_home()


if __name__ == "__main__":
    app = AcadexaAIApp()
    app.mainloop()