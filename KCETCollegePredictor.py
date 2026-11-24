from gemini_chatbot import render_chatbot
from pathlib import Path
import io

import pandas as pd
import requests
import streamlit as st

# -------------------------------------------------------------------
# KCET College Predictor Pro - Phase 1
# Dark UI foundation + reliable dataset loading.
# AI chatbot and improved ML prediction will be added in later phases.
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_FILE = ROOT_DIR / "CET_Database_Final2020.csv"
REMOTE_DATA_URL = (
    "https://raw.githubusercontent.com/tk227/kcet/refs/heads/main/"
    "CET_Database_Final2020.csv"
)

st.set_page_config(
    page_title="KCET Predictor Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the external stylesheet.
CSS_FILE = ROOT_DIR / "assets" / "style.css"
if CSS_FILE.exists():
    st.markdown(
        f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ------------------------------- State ------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    # Demo-only credentials. Replace with hashed persistent authentication
    # before deploying this application publicly.
    st.session_state.users = {
        "admin": {"password": "admin123", "cities": []}
    }
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


def login_user(username: str, password: str) -> bool:
    user = st.session_state.users.get(username)
    if user and user["password"] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.page = "Dashboard"
        return True
    return False


def load_dataset() -> pd.DataFrame:
    """Load the bundled CSV first; use GitHub only as a fallback."""
    if LOCAL_DATA_FILE.exists():
        data = pd.read_csv(LOCAL_DATA_FILE)
    else:
        response = requests.get(REMOTE_DATA_URL, timeout=25)
        response.raise_for_status()
        data = pd.read_csv(io.BytesIO(response.content))

    required = {"College", "Branch", "Location", "CETCode"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    data = data.dropna(subset=["College", "Branch", "Location"]).copy()
    for column in data.columns:
        if column not in required:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["College"] = data["College"].astype(str).str.strip()
    data["Branch"] = data["Branch"].astype(str).str.strip()
    data["Location"] = data["Location"].astype(str).str.strip()
    data = data[
        (data["College"] != "")
        & (data["Branch"] != "")
        & (data["Location"] != "")
    ]
    return data.reset_index(drop=True)


@st.cache_data(show_spinner="Loading KCET college data...")
def get_dataset() -> pd.DataFrame:
    return load_dataset()


def metric_card(label: str, value: str, icon: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">
                <span class="metric-label">{label}</span>
                <span class="metric-icon">{icon}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_login() -> None:
    left, right = st.columns([1.25, 0.85], gap="large")
    with left:
        st.markdown('<div class="login-brand">🎓 KCET PREDICTOR PRO</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-heading">Your next chapter<br/>starts here.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="login-copy">Explore colleges, compare cutoff ranks, '
            'and make a more informed admission decision.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="login-pill">✦  KCET admission planning workspace</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="login-panel">', unsafe_allow_html=True)
        st.markdown("### Welcome back")
        st.caption("Sign in to continue to your dashboard.")
        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input(
                    "Password", type="password", placeholder="Enter password"
                )
                submitted = st.form_submit_button("Sign in →", use_container_width=True)
            if submitted:
                if login_user(username.strip(), password):
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        with signup_tab:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a username")
                new_password = st.text_input("Choose a password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                create = st.form_submit_button("Create account", use_container_width=True)
            if create:
                if not new_username.strip() or not new_password:
                    st.warning("Enter a username and password.")
                elif new_username.strip() in st.session_state.users:
                    st.warning("That username already exists.")
                elif new_password != confirm_password:
                    st.warning("Passwords do not match.")
                else:
                    st.session_state.users[new_username.strip()] = {
                        "password": new_password,
                        "cities": [],
                    }
                    st.success("Account created. Please sign in.")

        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Demo login: `admin` / `admin123`")


if not st.session_state.logged_in:
    show_login()
    st.stop()

# ------------------------------ Data ---------------------------------
try:
    df = get_dataset()
except Exception as exc:
    st.error(f"Could not load the KCET dataset: {exc}")
    st.info(
        "Check that CET_Database_Final2020.csv is in the project folder "
        "and that it contains the expected columns."
    )
    st.stop()

category_columns = [
    col for col in df.columns
    if col not in {"CETCode", "College", "Location", "Branch"}
]
if not category_columns:
    st.error("No KCET category cutoff columns were found in the dataset.")
    st.stop()

# ----------------------------- Sidebar -------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="brand-mark">K</span>'
        '<span><b>KCET</b><small>PREDICTOR PRO</small></span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">WORKSPACE</div>', unsafe_allow_html=True)

    pages = [
        "Dashboard",
        "Cutoff Predictor",
        "ML Predictor",
        "AI Assistant",
        "My Profile",
        "About",
    ]
    icons = {
        "Dashboard": "🏠",
        "Cutoff Predictor": "🎯",
        "ML Predictor": "🤖",
        "AI Assistant": "💬",
        "My Profile": "👤",
        "About": "ℹ️"
    }
    for page_name in pages:
        selected = st.session_state.page == page_name
        if st.button(
            f"{icons[page_name]}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            st.session_state.page = page_name
            st.rerun()

    st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="user-chip"><span class="user-avatar">👤</span>'
        f'<span><b>{st.session_state.username}</b><small>Student account</small></span></div>',
        unsafe_allow_html=True,
    )
    if st.button("↪  Sign out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.page = "Dashboard"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

page = st.session_state.page

# ----------------------------- Dashboard -----------------------------
if page == "Dashboard":
    st.markdown(
        '<div class="hero-panel">'
        '<div class="hero-content">'
        '<div class="eyebrow">YOUR ADMISSION WORKSPACE</div>'
        '<div class="hero-title">Make your next move<br>with confidence.</div>'
        '<div class="hero-description">Explore colleges, compare cutoff ranks, '
        'and discover options for your KCET journey.</div>'
        '<div class="hero-tag">✦ &nbsp; KCET ADMISSION COMPANION</div>'
        '</div>'
        '<div class="hero-orbit"><div class="hero-orbit-inner">🎓</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("COLLEGES", f"{df['College'].nunique():,}", "🏛️", "Available in dataset")
    with c2:
        metric_card("BRANCHES", f"{df['Branch'].nunique():,}", "⌘", "Engineering disciplines")
    with c3:
        metric_card("CITIES", f"{df['Location'].nunique():,}", "📍", "Across Karnataka")

    st.markdown(
        '<div class="section-row"><div><div class="section-heading">Your tools</div>'
        '<div class="section-caption">Pick up where your admission planning begins.</div>'
        '</div><div class="section-chip">WORKSPACE</div></div>',
        unsafe_allow_html=True,
    )
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">🎯</div>'
            '<h3>Cutoff-based prediction</h3><p>Compare your KCET rank with '
            'the cutoff ranks available in the dataset.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open cutoff predictor →", key="quick_cutoff", use_container_width=True):
            st.session_state.page = "Cutoff Predictor"
            st.rerun()
    with q2:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">✧</div>'
            '<h3>ML prediction workspace</h3><p>The ML workflow will be reviewed '
            'and rebuilt in a later project phase.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open ML workspace →", key="quick_ml", use_container_width=True):
            st.session_state.page = "ML Predictor"
            st.rerun()

    st.markdown(
        '<div class="section-row"><div><div class="section-heading">Branch overview</div>'
        '<div class="section-caption">Most represented branches in the current dataset.</div>'
        '</div><div class="section-chip">DATA INSIGHTS</div></div>',
        unsafe_allow_html=True,
    )
    branch_counts = (
        df["Branch"].value_counts().head(8).rename_axis("Branch").reset_index(name="College entries")
    )
    st.bar_chart(branch_counts.set_index("Branch"), height=280)

    # ----------------------- Additional dashboard content --------------------
    st.markdown(
        '<div class="section-row"><div><div class="section-heading">Explore the dataset</div>'
        '<div class="section-caption">Quick summaries of the college information available here.</div>'
        '</div><div class="section-chip">COLLEGE EXPLORER</div></div>',
        unsafe_allow_html=True,
    )
    explore_col1, explore_col2 = st.columns(2, gap="large")
    with explore_col1:
        st.markdown("#### Colleges with most dataset entries")
        college_entries = (
            df["College"].value_counts().head(10)
            .rename_axis("College").reset_index(name="Dataset entries")
        )
        st.caption("Entry count is dataset coverage—not a college quality ranking.")
        st.bar_chart(college_entries.set_index("College"), height=320)
    with explore_col2:
        st.markdown("#### College entries by city")
        city_entries = (
            df["Location"].value_counts().head(10)
            .rename_axis("City").reset_index(name="Dataset entries")
        )
        st.bar_chart(city_entries.set_index("City"), height=320)

    st.markdown(
        '<div class="section-row"><div><div class="section-heading">KCET admission guide</div>'
        '<div class="section-caption">A few helpful reminders while exploring your options.</div>'
        '</div><div class="section-chip">STUDENT GUIDE</div></div>',
        unsafe_allow_html=True,
    )
    guide1, guide2, guide3 = st.columns(3, gap="medium")
    with guide1:
        with st.container(border=True):
            st.markdown("### 🎯 Understand your rank")
            st.write("Use your KCET rank to compare with cutoff ranks. A smaller numerical rank is generally more competitive.")
    with guide2:
        with st.container(border=True):
            st.markdown("### 🧭 Explore branches")
            st.write("Review different engineering branches and consider your interests, strengths and future career paths.")
    with guide3:
        with st.container(border=True):
            st.markdown("### 📍 Keep options open")
            st.write("Compare colleges across multiple cities. Use the cutoff tool to explore options rather than relying on one choice.")

    with st.expander("How to use this predictor"):
        st.markdown(
            "1. Open **Cutoff Predictor** from the sidebar.\n"
            "2. Enter your KCET rank.\n"
            "3. Choose a reservation category from the dataset columns.\n"
            "4. Select preferred branches and optionally cities.\n"
            "5. Review the matching cutoff records.\n\n"
            "**Note:** Dataset cutoff values are historical reference data and do not guarantee admission."
        )

# --------------------------- Cutoff Predictor -------------------------
elif page == "Cutoff Predictor":
    st.markdown('<div class="eyebrow">ADMISSION TOOLS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Cutoff predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Enter your rank and preferences to find '
        'matching college options.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            rank = st.number_input("Your KCET rank", min_value=1, max_value=999999, value=10000, step=100)
            category = st.selectbox("Reservation category", category_columns)
        with col2:
            branches = sorted(df["Branch"].dropna().unique().tolist())
            selected_branches = st.multiselect("Preferred branches", branches)
            cities = sorted(df["Location"].dropna().unique().tolist())
            selected_cities = st.multiselect("Preferred cities (optional)", cities)

        search = st.button("Find matching colleges  →", type="primary", use_container_width=True)

    if search:
        if not selected_branches:
            st.warning("Select at least one preferred branch.")
        else:
            results = df[df["Branch"].isin(selected_branches)].copy()
            if selected_cities:
                results = results[results["Location"].isin(selected_cities)]

            results["Cutoff"] = pd.to_numeric(results[category], errors="coerce")
            results = results[
                results["Cutoff"].notna()
                & (results["Cutoff"] > 0)
                & (rank <= results["Cutoff"])
            ]
            results = results[
                ["College", "Branch", "Location", "CETCode", "Cutoff"]
            ].sort_values("Cutoff", ascending=True)

            st.session_state.cutoff_results = results

    if "cutoff_results" in st.session_state:
        results = st.session_state.cutoff_results
        st.markdown('<div class="section-heading">Prediction results</div>', unsafe_allow_html=True)
        if results.empty:
            st.warning("No matching colleges found. Try a different branch, city, or category.")
        else:
            st.success(f"{len(results)} matching college entries found.")
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="kcet_cutoff_results.csv",
                mime="text/csv",
            )

# ------------------------------ ML Page ------------------------------

# ------------------------------ ML Page ------------------------------
elif page == "ML Predictor":
    st.markdown(
        '<div class="eyebrow">INTELLIGENT ADMISSION TOOLS</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-title">Smart KCET Rank Predictor</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Find suitable engineering colleges '
        'using your KCET rank and admission preferences.</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This predictor compares your rank with historical cutoff data. "
        "It is a data-based estimate, not a guaranteed admission result."
    )

    # ---------------- Student Details ----------------
    st.markdown("## 1. Enter your details")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            ml_rank = st.number_input(
                "Your KCET rank",
                min_value=1,
                max_value=999999,
                value=10000,
                step=100,
                key="ml_rank_input"
            )

            ml_category = st.selectbox(
                "Select category",
                category_columns,
                key="ml_category_input"
            )

        with col2:
            ml_branches = st.multiselect(
                "Preferred engineering branches",
                sorted(df["Branch"].dropna().unique()),
                key="ml_branch_input"
            )

            ml_cities = st.multiselect(
                "Preferred cities (optional)",
                sorted(df["Location"].dropna().unique()),
                key="ml_city_input"
            )

        run_prediction = st.button(
            "Analyse My College Options →",
            type="primary",
            use_container_width=True,
            key="run_ml_prediction"
        )

    # ---------------- Prediction Logic ----------------
    if run_prediction:

        if not ml_branches:
            st.warning("Please select at least one engineering branch.")
            st.session_state.pop("ml_prediction_results", None)

        else:
            prediction = df[
                df["Branch"].isin(ml_branches)
            ].copy()

            if ml_cities:
                prediction = prediction[
                    prediction["Location"].isin(ml_cities)
                ]

            prediction["Historical cutoff"] = pd.to_numeric(
                prediction[ml_category],
                errors="coerce"
            )

            prediction = prediction[
                prediction["Historical cutoff"].notna()
                & (prediction["Historical cutoff"] > 0)
            ].copy()

            prediction["Rank difference"] = (
                prediction["Historical cutoff"] - ml_rank
            )

            def classify_option(gap):
                if gap >= 5000:
                    return "Safer option"
                elif gap >= 0:
                    return "Possible option"
                return "Reach option"

            prediction["Estimated range"] = (
                prediction["Rank difference"].apply(classify_option)
            )

            prediction = prediction.sort_values(
                "Historical cutoff",
                ascending=False
            )

            st.session_state.ml_prediction_results = prediction[
                [
                    "College",
                    "Branch",
                    "Location",
                    "CETCode",
                    "Historical cutoff",
                    "Rank difference",
                    "Estimated range"
                ]
            ]

            st.session_state.ml_prediction_rank = ml_rank
            st.session_state.ml_prediction_category = ml_category

    # ---------------- Prediction Results ----------------
    if "ml_prediction_results" in st.session_state:

        prediction = st.session_state.ml_prediction_results

        st.markdown("---")
        st.markdown("## 2. Your college prediction results")

        st.caption(
            f"KCET Rank: {st.session_state.ml_prediction_rank:,} | "
            f"Category: {st.session_state.ml_prediction_category}"
        )

        if prediction.empty:
            st.warning(
                "No matching cutoff records found. "
                "Try another category or engineering branch."
            )

        else:
            safer = int(
                (prediction["Estimated range"] == "Safer option").sum()
            )

            possible = int(
                (prediction["Estimated range"] == "Possible option").sum()
            )

            reach = int(
                (prediction["Estimated range"] == "Reach option").sum()
            )

            st.markdown("### Your admission overview")

            m1, m2, m3 = st.columns(3)

            with m1:
                metric_card(
                    "SAFER OPTIONS",
                    str(safer),
                    "🟢",
                    "Larger historical cutoff margin"
                )

            with m2:
                metric_card(
                    "POSSIBLE OPTIONS",
                    str(possible),
                    "🟡",
                    "Within historical cutoff"
                )

            with m3:
                metric_card(
                    "REACH OPTIONS",
                    str(reach),
                    "🔵",
                    "More competitive cutoff"
                )

            # ---------------- Filter Results ----------------
            st.markdown("### Filter your college list")

            filter_choice = st.selectbox(
                "Choose prediction type",
                [
                    "All options",
                    "Safer option",
                    "Possible option",
                    "Reach option"
                ],
                key="ml_result_filter"
            )

            if filter_choice == "All options":
                shown = prediction
            else:
                shown = prediction[
                    prediction["Estimated range"] == filter_choice
                ]

            st.dataframe(
                shown,
                use_container_width=True,
                hide_index=True
            )

            # ---------------- Download Report ----------------
            st.download_button(
                "Download Prediction Report",
                data=shown.to_csv(index=False).encode("utf-8"),
                file_name="kcet_prediction_report.csv",
                mime="text/csv",
                key="download_ml_report"
            )

            # ---------------- Explanation ----------------
            st.markdown("---")
            st.markdown("### 3. Understand your results")

            with st.expander("What does Safer option mean?"):
                st.write(
                    "The historical cutoff rank is at least 5,000 "
                    "numerical positions larger than your rank."
                )

            with st.expander("What does Possible option mean?"):
                st.write(
                    "Your rank is within the historical cutoff, "
                    "but the margin is smaller."
                )

            with st.expander("What does Reach option mean?"):
                st.write(
                    "Your rank is numerically worse than the "
                    "historical cutoff rank."
                )

            st.warning(
                "Historical cutoffs can change according to counselling "
                "rounds, seat availability, category and admission year. "
                "Always verify the official KCET counselling information."
            )
elif page == "AI Assistant":
    render_chatbot(df)
# -------------------------------- About -------------------------------
elif page == "About":
    st.markdown('<div class="eyebrow">PROJECT INFORMATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">About KCET Predictor Pro</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A college exploration and cutoff comparison '
        'tool for Karnataka CET aspirants.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="feature-card"><h3>Current technology</h3>'
        '<p>Python · Streamlit · Pandas · CSV dataset</p>'
        '<h3>Project roadmap</h3><p>Phase 1: Dark UI and stable foundation<br>'
        'Phase 2: Gemini AI assistant<br>'
        'Phase 3: Improved admission prediction workflow<br>'
        'Phase 4: Testing and deployment preparation</p></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="app-footer">KCET PREDICTOR PRO · Admission planning made clearer</div>',
    unsafe_allow_html=True,
)
