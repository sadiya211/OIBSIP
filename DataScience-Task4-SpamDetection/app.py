import streamlit as st
import pandas as pd
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ─── Stopwords (hardcoded — avoids MS-Store Python NLTK path bug) ─────────────
_STOP_WORDS: set = {
    "i","me","my","myself","we","our","ours","ourselves",
    "you","your","yours","yourself","yourselves",
    "he","him","his","himself","she","her","hers","herself",
    "it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those",
    "am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against",
    "between","into","through","during","before","after",
    "above","below","to","from","up","down","in","out",
    "on","off","over","under","again","further","then","once",
    "here","there","when","where","why","how","all","both",
    "each","few","more","most","other","some","such","no",
    "nor","not","only","own","same","so","than","too","very",
    "s","t","can","will","just","don","should","now",
    "d","ll","m","o","re","ve","y",
    "ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn",
    "u","ur","r","b","c","k","n","gt","lt","amp",
}
_STEMMER = PorterStemmer()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #08101e !important;
    color: #f1f5f9 !important;
}
.main .block-container {
    padding-top: 2.5rem !important;
    max-width: 460px !important;
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent !important; }

/* ── Logo ── */
.logo-wrap { text-align:center; margin-bottom:1.8rem; }
.logo-icon { font-size:3rem; }
.logo-title {
    font-size:1.75rem; font-weight:800;
    background:linear-gradient(135deg,#818cf8,#60a5fa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:.3rem 0 .15rem;
}
.logo-sub { font-size:.8rem; color:#64748b; letter-spacing:.04em; }

/* ── Toggle strip (Login / Sign Up) ── */
.toggle-strip {
    display:flex; background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.08);
    border-radius:12px; padding:4px; margin-bottom:1.6rem;
    gap:4px;
}
.toggle-btn {
    flex:1; text-align:center; padding:.55rem 0;
    border-radius:9px; font-size:.88rem; font-weight:600;
    cursor:pointer; transition:all .2s;
}
.toggle-active {
    background:linear-gradient(135deg,#6366f1,#3b82f6);
    color:#fff; box-shadow:0 3px 10px rgba(99,102,241,.4);
}
.toggle-inactive { color:#64748b; }
.toggle-inactive:hover { color:#94a3b8; }

/* ── Section label ── */
.field-label {
    font-size:.77rem; font-weight:600; color:#94a3b8;
    text-transform:uppercase; letter-spacing:.07em;
    margin-bottom:.3rem; display:block;
}

/* ── Inputs ── */
input[type="text"], input[type="password"] {
    background:#0d1525 !important;
    border:1.5px solid rgba(255,255,255,.1) !important;
    border-radius:10px !important; color:#f1f5f9 !important;
    font-family:'Inter',sans-serif !important; font-size:.92rem !important;
    transition:border-color .2s !important;
}
input[type="text"]:focus, input[type="password"]:focus {
    border-color:rgba(129,140,248,.65) !important;
    box-shadow:0 0 0 3px rgba(129,140,248,.12) !important;
}
textarea {
    background:#ffffff !important;
    border:1.5px solid #d1d5db !important;
    border-radius:12px !important;
    color:#111827 !important;
    font-family:'Inter',sans-serif !important;
    font-size:.92rem !important;
    line-height:1.65 !important;
    transition:border-color .2s !important;
    caret-color:#6366f1 !important;
}
textarea:focus {
    border-color:#6366f1 !important;
    box-shadow:0 0 0 3px rgba(99,102,241,.15) !important;
}
textarea::placeholder { color:#9ca3af !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#6366f1,#3b82f6) !important;
    border:none !important; border-radius:11px !important;
    color:#fff !important; font-weight:700 !important;
    font-size:.97rem !important; padding:.72rem 1.4rem !important;
    width:100% !important; letter-spacing:.02em !important;
    box-shadow:0 4px 18px rgba(99,102,241,.38) !important;
    transition:all .2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 6px 24px rgba(99,102,241,.55) !important;
}

/* ── Secondary button ── */
.stButton > button[kind="secondary"] {
    background:rgba(255,255,255,.04) !important;
    border:1px solid rgba(255,255,255,.1) !important;
    border-radius:10px !important; color:#94a3b8 !important;
    width:100% !important; transition:all .2s !important;
}
.stButton > button[kind="secondary"]:hover {
    background:rgba(255,255,255,.08) !important; color:#f1f5f9 !important;
}

/* ── Divider ── */
.divider-row {
    display:flex; align-items:center; gap:.75rem; margin:1.1rem 0;
}
.divider-line { flex:1; height:1px; background:rgba(255,255,255,.07); }
.divider-text { font-size:.75rem; color:#475569; white-space:nowrap; }

/* ── Result cards ── */
.result-spam {
    border-radius:16px; padding:1.5rem 1.6rem;
    background:linear-gradient(135deg,rgba(239,68,68,.14),rgba(220,38,38,.06));
    border:1.5px solid rgba(239,68,68,.42);
    animation:popIn .35s cubic-bezier(.34,1.56,.64,1);
}
.result-ham {
    border-radius:16px; padding:1.5rem 1.6rem;
    background:linear-gradient(135deg,rgba(16,185,129,.14),rgba(5,150,105,.06));
    border:1.5px solid rgba(16,185,129,.42);
    animation:popIn .35s cubic-bezier(.34,1.56,.64,1);
}
@keyframes popIn {
    0%  { opacity:0; transform:scale(.94) translateY(8px); }
    100%{ opacity:1; transform:scale(1)   translateY(0);   }
}
.result-emoji  { font-size:2.4rem; text-align:center; }
.result-verdict{ font-size:1.4rem; font-weight:800; text-align:center; margin:.3rem 0 .1rem; }
.result-msg    { font-size:.83rem; color:#94a3b8; text-align:center; }

/* ── Probability bars ── */
.bar-row {
    display:flex; justify-content:space-between; align-items:center;
    margin-top:1rem; margin-bottom:.25rem;
    font-size:.78rem; color:#94a3b8;
}
.bar-track { background:rgba(255,255,255,.07); border-radius:99px; height:8px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:99px; }

/* ── Summary box ── */
.summary-box {
    background:rgba(99,102,241,.07);
    border:1px solid rgba(99,102,241,.2);
    border-radius:12px; padding:1rem 1.2rem;
    font-size:.83rem; color:#94a3b8; line-height:1.75;
    margin-top:1rem;
}
.summary-box b { color:#c7d2fe; }

/* ── Hint ── */
.hint { font-size:.74rem; color:#475569; text-align:center; margin-top:.6rem; }

/* ── Password strength bar ── */
.strength-wrap { margin-top:.4rem; }
.strength-track { background:rgba(255,255,255,.07); border-radius:99px; height:5px; }
.strength-fill  { height:100%; border-radius:99px; transition:width .3s; }
.strength-label { font-size:.7rem; margin-top:.25rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════════════════════════════
if "logged_in"  not in st.session_state:
    st.session_state.logged_in  = False
if "username"   not in st.session_state:
    st.session_state.username   = ""
if "auth_tab"   not in st.session_state:
    st.session_state.auth_tab   = "login"      # "login" | "signup"
if "users_db"   not in st.session_state:
    # Seeded demo accounts
    st.session_state.users_db   = {
        "sadiya": "spam@2025",
        "admin":  "admin123",
        "demo":   "demo123",
    }


# ─── Model (cached) ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
    df.columns = ["label","message"]

    def _clean(txt):
        txt = txt.lower().translate(str.maketrans("","",string.punctuation))
        return " ".join(_STEMMER.stem(w) for w in txt.split() if w not in _STOP_WORDS)

    df["clean"] = df["message"].apply(_clean)
    df["y"]     = df["label"].map({"ham":0,"spam":1})
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["clean"])
    model = MultinomialNB()
    model.fit(X, df["y"])
    return model, tfidf, _clean


def _predict(text):
    model, tfidf, cleaner = _load_model()
    vec  = tfidf.transform([cleaner(text)])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return int(pred), float(prob[1]), float(prob[0])


def _summarise(text):
    sents = [s.strip() for s in text.replace("\n"," ").split(".") if len(s.strip()) > 15]
    first = (sents[0] + ".") if sents else text[:120]
    return first, len(text.split()), len(text)


def _pw_strength(pw):
    score = 0
    if len(pw) >= 8:   score += 1
    if any(c.isupper() for c in pw): score += 1
    if any(c.isdigit() for c in pw): score += 1
    if any(c in "!@#$%^&*_-" for c in pw): score += 1
    labels = ["","Weak","Fair","Good","Strong"]
    colors = ["","#ef4444","#f59e0b","#3b82f6","#10b981"]
    return score, labels[score] if score else "", colors[score] if score else ""


# ════════════════════════════════════════════════════════════════════════════
# PAGE A — AUTH  (Login / Sign Up)
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    # Logo
    st.markdown("""
    <div class="logo-wrap">
        <div class="logo-icon">🛡️</div>
        <div class="logo-title">SpamShield</div>
        <div class="logo-sub">AI-Powered Email &amp; SMS Spam Detector</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Toggle strip ──────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("🔑  Log In",
                     type="primary" if st.session_state.auth_tab == "login" else "secondary",
                     key="tab_login"):
            st.session_state.auth_tab = "login"
            st.rerun()
    with col_r:
        if st.button("✨  Sign Up",
                     type="primary" if st.session_state.auth_tab == "signup" else "secondary",
                     key="tab_signup"):
            st.session_state.auth_tab = "signup"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LOGIN form ────────────────────────────────────────────────────────
    if st.session_state.auth_tab == "login":

        with st.form("login_form", clear_on_submit=False):
            st.markdown('<span class="field-label">Username</span>', unsafe_allow_html=True)
            username = st.text_input("u", placeholder="Enter your username",
                                     label_visibility="collapsed")

            st.markdown('<span class="field-label" style="margin-top:.85rem;display:block;">Password</span>',
                        unsafe_allow_html=True)
            password = st.text_input("p", type="password", placeholder="Enter your password",
                                     label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Log In →", type="primary")

        if submitted:
            db = st.session_state.users_db
            if not username.strip():
                st.error("Please enter your username.")
            elif username not in db:
                st.error("Account not found. Please **Sign Up** first.")
            elif db[username] != password:
                st.error("Incorrect password. Please try again.")
            else:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.rerun()

        st.markdown("""
        <div class="hint">
            Demo · <b style="color:#818cf8">sadiya</b> / spam@2025 &nbsp;·&nbsp;
            <b style="color:#818cf8">demo</b> / demo123
        </div>
        """, unsafe_allow_html=True)

    # ── SIGN UP form ──────────────────────────────────────────────────────
    else:

        with st.form("signup_form", clear_on_submit=False):
            st.markdown('<span class="field-label">Full Name</span>', unsafe_allow_html=True)
            full_name = st.text_input("fn", placeholder="e.g. Sadiya Anmol",
                                      label_visibility="collapsed")

            st.markdown('<span class="field-label" style="margin-top:.85rem;display:block;">Username</span>',
                        unsafe_allow_html=True)
            new_user = st.text_input("nu", placeholder="Choose a username (no spaces)",
                                     label_visibility="collapsed")

            st.markdown('<span class="field-label" style="margin-top:.85rem;display:block;">Password</span>',
                        unsafe_allow_html=True)
            new_pass = st.text_input("np", type="password",
                                     placeholder="Min 6 characters",
                                     label_visibility="collapsed")

            st.markdown('<span class="field-label" style="margin-top:.85rem;display:block;">Confirm Password</span>',
                        unsafe_allow_html=True)
            confirm_pass = st.text_input("cp", type="password",
                                         placeholder="Re-enter your password",
                                         label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            reg = st.form_submit_button("Create Account →", type="primary")

        # Live password-strength hint (outside form so it updates as user types)
        if new_pass:
            score, label, color = _pw_strength(new_pass)
            pct = score * 25
            st.markdown(f"""
            <div class="strength-wrap">
                <div class="strength-track">
                    <div class="strength-fill" style="width:{pct}%;background:{color};"></div>
                </div>
                <div class="strength-label" style="color:{color};">
                    Password strength: <b>{label}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if reg:
            db = st.session_state.users_db
            uname = new_user.strip().lower()

            if not full_name.strip():
                st.error("Please enter your full name.")
            elif not uname:
                st.error("Please choose a username.")
            elif " " in uname:
                st.error("Username must not contain spaces.")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            elif uname in db:
                st.error(f"Username **{uname}** is already taken. Please choose another.")
            else:
                db[uname] = new_pass
                st.session_state.users_db    = db
                st.session_state.logged_in   = True
                st.session_state.username    = uname
                st.session_state.full_name   = full_name.strip()
                st.success(f"🎉 Account created! Welcome, **{full_name.strip()}**!")
                st.rerun()

        st.markdown("""
        <div class="hint">Already have an account? Switch to <b style="color:#818cf8">Log In</b> above.</div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE B — SPAM CHECKER
# ════════════════════════════════════════════════════════════════════════════
else:
    uname     = st.session_state.username
    full_name = st.session_state.get("full_name", uname.capitalize())

    # ── Top bar ──────────────────────────────────────────────────────────
    col_brand, col_out = st.columns([3,1])
    with col_brand:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:.6rem;padding:.1rem 0 1rem;">
            <span style="font-size:1.5rem">🛡️</span>
            <span style="font-size:1.1rem;font-weight:800;
                background:linear-gradient(135deg,#818cf8,#60a5fa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;">SpamShield</span>
        </div>
        """, unsafe_allow_html=True)
    with col_out:
        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.pop("full_name", None)
            st.rerun()

    st.markdown(f"""
    <p style="font-size:.84rem;color:#64748b;margin-top:-.7rem;margin-bottom:1.5rem;">
        👋 Welcome, <b style="color:#a5b4fc;">{full_name}</b>
    </p>
    """, unsafe_allow_html=True)

    # ── Message input ────────────────────────────────────────────────────
    st.markdown('<span class="field-label">📩 Paste your email or SMS</span>',
                unsafe_allow_html=True)

    message = st.text_area(
        "msg", height=180,
        placeholder="Type or paste the message you want to check here…",
        label_visibility="collapsed",
        key="checker_msg",
    )

    col_chk, col_clr = st.columns([3,1])
    with col_chk:
        check = st.button("🔍 Check for Spam", type="primary")
    with col_clr:
        if st.button("Clear", type="secondary"):
            st.session_state.checker_msg = ""
            st.rerun()

    # ── Result ───────────────────────────────────────────────────────────
    if check:
        msg = message.strip()
        if not msg:
            st.warning("⚠️  Please paste a message first.")
        else:
            with st.spinner("Analysing…"):
                is_spam, spam_p, ham_p = _predict(msg)
                first_sent, words, chars = _summarise(msg)

            st.markdown("<br>", unsafe_allow_html=True)

            if is_spam:
                css, emoji, verdict, color = (
                    "result-spam","🚨","SPAM Detected","#ef4444"
                )
                advice = "Do not click any links or share personal details."
            else:
                css, emoji, verdict, color = (
                    "result-ham","✅","Looks Legitimate","#10b981"
                )
                advice = "This message appears safe and genuine."

            # Build HTML with NO leading indentation — Markdown treats
            # lines with 4+ leading spaces as code blocks, which caused
            # the raw HTML to appear as text in the output.
            ham_pct  = f"{ham_p*100:.1f}"
            spam_pct = f"{spam_p*100:.1f}"
            conf_pct = f"{max(spam_p,ham_p)*100:.1f}"

            result_html = (
"<div class='"+css+"'>"
"<div class='result-emoji'>"+emoji+"</div>"
"<div class='result-verdict' style='color:"+color+";'>"+verdict+"</div>"
"<div class='result-msg'>"+advice+"</div>"
"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:1rem;margin-bottom:.25rem;font-size:.78rem;color:#94a3b8;'>"
"<span>Ham (safe)</span>"
"<span style='color:#10b981;font-weight:700;'>"+ham_pct+"%</span></div>"
"<div style='background:rgba(255,255,255,.1);border-radius:99px;height:8px;overflow:hidden;'>"
"<div style='width:"+ham_pct+"%;height:100%;border-radius:99px;background:#10b981;'></div></div>"
"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:.7rem;margin-bottom:.25rem;font-size:.78rem;color:#94a3b8;'>"
"<span>Spam</span>"
"<span style='color:#ef4444;font-weight:700;'>"+spam_pct+"%</span></div>"
"<div style='background:rgba(255,255,255,.1);border-radius:99px;height:8px;overflow:hidden;'>"
"<div style='width:"+spam_pct+"%;height:100%;border-radius:99px;background:#ef4444;'></div></div>"
"</div>"
"<div style='background:#1a2540;border:1.5px solid #3b4f7a;border-radius:14px;padding:1.2rem 1.4rem;font-size:.85rem;line-height:1.8;margin-top:1rem;'>"
"<div style='font-size:.9rem;font-weight:700;color:#818cf8;margin-bottom:.55rem;'>📋 Message Summary</div>"
"<div style='color:#e2e8f0;font-size:.88rem;margin-bottom:.75rem;'>"+first_sent+"</div>"
"<div style='display:flex;gap:1.4rem;flex-wrap:wrap;font-size:.82rem;'>"
"<span><span style='color:#94a3b8;'>Words </span><span style='color:#f1f5f9;font-weight:700;'>"+str(words)+"</span></span>"
"<span><span style='color:#94a3b8;'>Chars </span><span style='color:#f1f5f9;font-weight:700;'>"+str(chars)+"</span></span>"
"<span><span style='color:#94a3b8;'>Confidence </span><span style='color:#a5b4fc;font-weight:700;'>"+conf_pct+"%</span></span>"
"</div>"
"</div>"
            )
            st.markdown(result_html, unsafe_allow_html=True)

    # ── Try an example ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💡 Try an example message"):
        examples = {
            "🚨 Spam — Prize winner":  "WINNER!! You have been selected as a winner. Call 08712300971 NOW to claim your FREE £1000 prize! Reply WIN to 80085.",
            "🚨 Spam — Account alert": "URGENT: Your account will be SUSPENDED. Verify your details immediately at http://secure-login.xyz or lose access!",
            "✅ Ham — Casual text":    "Hey, are you coming to class tomorrow? Let me know, we can grab coffee before.",
            "✅ Ham — Work message":   "Hi, the meeting has been pushed to 4 pm. Can you please update the slides before then? Thanks.",
        }
        choice = st.selectbox("Pick one:", list(examples.keys()), label_visibility="collapsed")
        if st.button("Load this example →", type="secondary"):
            st.session_state.checker_msg = examples[choice]
            st.rerun()

    st.markdown("""
    <div class="hint" style="margin-top:2rem;">
        Built by <b style="color:#6366f1;">Sadiya Anmol</b> ·
        Oasis Infobyte Data Science · Task 4
    </div>
    """, unsafe_allow_html=True)