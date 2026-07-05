from __future__ import annotations

import base64
import sqlite3
import json
import tempfile
from pathlib import Path
from datetime import datetime
import streamlit as st
from PIL import Image

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails

st.set_page_config(page_title="Scan-R", layout="wide")

DB_PATH = "medical_ai_evidence.sqlite"
SAMPLE_DIR = Path("images")

#  Session state
if "page" not in st.session_state:
    st.session_state.page = "accueil"
if "analyse_count" not in st.session_state:
    st.session_state.analyse_count = 0
if "rapport_detail" not in st.session_state:
    st.session_state.rapport_detail = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

#  Thème dynamique
if st.session_state.theme == "dark":
    THEME_VARS = """
    --bg-main:    #0D1B2A;
    --bg-card:    #1a3a6e;
    --bg-card2:   #1A3A5C;
    --accent:     #2A8FD4;
    --text-main:  #E8F4FF;
    --text-muted: #8AB4D4;
    """
else:
    THEME_VARS = """
    --bg-main:    #F4F7FB;
    --bg-card:    #1a3a6e;
    --bg-card2:   #1A3A5C;
    --accent:     #2A8FD4;
    --text-main:  #0D1B2A;
    --text-muted: #4A6580;
    """

st.markdown(f"""
<style>
:root {{
    {THEME_VARS}
}}
h1 {{ max-width: 75% !important; }}
.stApp {{ background-color: var(--bg-main); color: var(--text-main); }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stMetricValue"] {{ color: var(--accent) !important; font-weight: 700 !important; }}
.stSelectbox > div > div, .stFileUploader > div {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--bg-card2) !important;
    border-radius: 8px !important;
}}
.logo-top-right {{
    position: absolute;
    top: -60px;
    right: 24px;
    z-index: 9999;
    border-radius: 16px;
    overflow: hidden;
}}
.logo-top-right img {{
    width: 276px;
    height: 276px;
    border-radius: 16px;
}}
.step-card {{
    background: var(--bg-card);
    border: 1px solid var(--bg-card2);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
}}
.step-icon {{ font-size: 2rem; margin-bottom: 10px; filter: none !important; }}
.step-title {{ font-weight: 700; color: #E8F4FF !important; font-size: 1rem; margin-bottom: 6px; }}
.step-desc {{ color: #8AB4D4 !important; font-size: 0.85rem; line-height: 1.5; }}
.divider {{ border: none; border-top: 1px solid var(--bg-card2); margin: 28px 0; }}
.nav-card {{
    background: var(--bg-card);
    border: 1px solid var(--bg-card2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 8px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s;
    height: 130px;
}}
.nav-card:hover {{ border-color: var(--accent); }}
.nav-card .nav-title, .nav-card .nav-desc {{ color: #E8F4FF !important; }}
.nav-card-analyse .nav-title, .nav-card-analyse .nav-desc {{ color: #E8F4FF !important; }}
.stFileUploader label {{ color: var(--text-main) !important; }}
.stFileUploader button, .stFileUploader [data-testid="stFileUploaderDropzone"] span {{ color: #E8F4FF !important; }}
/* Sidebar toujours en mode nuit */
[data-testid="stSidebar"] {{
    background-color: #112240 !important;
}}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {{
    color: #E8F4FF !important;
}}
.nav-card-analyse {{
    background: #1e5aa8;
    border: 1px solid #2A8FD4;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 8px;
    text-align: center;
    cursor: pointer;
    height: 130px;
}}
.nav-icon {{ font-size: 3rem; margin-bottom: 14px; }}
.nav-title {{ font-weight: 700; color: var(--text-main); font-size: 1.1rem; margin-bottom: 8px; }}
.nav-desc {{ color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; }}
.sample-card {{
    background: var(--bg-card);
    border: 1px solid var(--bg-card2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
}}
.badge-normal {{ color: #2ecc71; font-weight: 700; }}
.badge-opacity {{ color: #f39c12; font-weight: 700; }}
.badge-uncertain {{ color: #e74c3c; font-weight: 700; }}
/* Boutons Streamlit */
.stButton > button {{
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
}}
.stButton > button:hover {{
    opacity: 0.85 !important;
}}
/* Textes labels */
label, .stSelectbox label, .stFileUploader label {{
    color: var(--text-main) !important;
}}
/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: var(--bg-card) !important;
}}
[data-testid="stSidebar"] * {{
    color: var(--text-main) !important;
}}
</style>
""", unsafe_allow_html=True)

#  Logo
def load_logo(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lstrip(".")
    return f'<img src="data:image/{ext};base64,{data}">'

st.markdown(f"""
<div class="logo-top-right">
    {load_logo("images/assets/logo final.png")}
</div>
""", unsafe_allow_html=True)

#  DB helpers
def get_connection():
    return sqlite3.connect(DB_PATH)

def save_run(image_path: str, pred: dict, mode: str):
    con = get_connection()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO runs (image_path, model_name, prompt_version, prediction_json,
                          predicted_class, confidence, latency_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_path,
        pred.get("model_name", f"toy-rule-{mode}"),
        pred.get("prompt_version", f"{mode}_v1"),
        json.dumps(pred),
        pred["predicted_class"],
        pred["confidence"],
        pred.get("latency_ms", 0),
        datetime.now().isoformat(),
    ))
    con.commit()
    con.close()

def load_runs():
    con = get_connection()
    cur = con.cursor()
    cur.execute("SELECT id, image_path, predicted_class, confidence, model_name, created_at FROM runs ORDER BY created_at DESC")
    rows = cur.fetchall()
    con.close()
    return rows

#  Sidebar navigation
with st.sidebar:
    st.markdown("### Navigation")
    if st.button(" Accueil", use_container_width=True):
        st.session_state.page = "accueil"
    if st.button("  Analyse", use_container_width=True):
        st.session_state.page = "analyse"
    if st.button("  Rapports", use_container_width=True):
        st.session_state.page = "rapports"
    if st.button("  Apprentissage", use_container_width=True):
        st.session_state.page = "apprentissage"
    st.markdown("---")
    theme_label = "Mode jour" if st.session_state.theme == "dark" else "Mode nuit"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.markdown("---")
    st.markdown("### Session en cours")
    st.metric("Images analysées", st.session_state.analyse_count)
    st.markdown("---")
    st.markdown("**Scan-R** — prototype médical")

#
# PAGE ACCUEIL
#
if st.session_state.page == "accueil":
    st.title("Scan-R — Assistant radiologue virtuel")
    st.markdown("<p style='color:#8AB4D4; font-size:1.1rem;'>Prototype médical d'analyse de radiographies thoraciques par intelligence artificielle</p>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    #  Comment ça marche
    st.markdown("#### Comment ça marche ?")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""<div class="step-card"><div class="step-icon"></div>
        <div class="step-title">1. Uploader une radio</div>
        <div class="step-desc">Déposez une radiographie thoracique frontale au format PNG ou JPG.</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""<div class="step-card"><div class="step-icon"></div>
        <div class="step-title">2. Analyse IA</div>
        <div class="step-desc">Le modèle analyse l'image et détecte les signes de pneumonie.</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""<div class="step-card"><div class="step-icon"></div>
        <div class="step-title">3. Résultats</div>
        <div class="step-desc">Classe prédite, score de confiance, observations et justification.</div></div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Accédez à vos outils")

    #  Ligne 1 : Analyse centré
    col_left, col_centre, col_right = st.columns([1, 2, 1])
    with col_centre:
        st.markdown("""
        <div class="nav-card-analyse">
            <div class="nav-icon"></div>
            <div class="nav-title">Analyse</div>
            <div class="nav-desc">Analysez une radiographie thoracique et obtenez un diagnostic assisté par IA.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Accéder à l'analyse", use_container_width=True):
            st.session_state.page = "analyse"
            st.rerun()

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    #  Ligne 2 : Rapports + Apprentissage
    col_l, col_r2, col_r3, col_rr = st.columns([1, 2, 2, 1])
    with col_r2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon"></div>
            <div class="nav-title">Historique Rapports</div>
            <div class="nav-desc">Consultez l'historique de toutes les analyses effectuées et leurs résultats.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Accéder aux rapports", use_container_width=True):
            st.session_state.page = "rapports"
            st.rerun()
    with col_r3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon"></div>
            <div class="nav-title">Espace Apprentissage</div>
            <div class="nav-desc">Explorez des exemples annotés de radiographies pour vous former à la lecture d'images.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Accéder à l'apprentissage", use_container_width=True):
            st.session_state.page = "apprentissage"
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    #  Pourquoi Scan-R ?
    st.markdown("#### Pourquoi Scan-R ?")
    st.markdown(f"""
    <div style="background:var(--bg-card); border:1px solid var(--bg-card2); border-radius:16px; padding:28px 32px; margin-bottom:24px;">
        <p style="color:#E8F4FF; font-size:1rem; line-height:1.7; margin-bottom:16px;">
            Le cancer du poumon est le 3ème cancer le plus fréquent. La capacité à le détecter rapidement est critique.
            Le CHU de Bordeaux - Pellegrin fait face à une surcharge croissante de ses services de radiologie,
            une pénurie de spécialistes et un volume d'examens en hausse constante.
        </p>
        <p style="color:#E8F4FF; font-size:1rem; line-height:1.7; margin-bottom:16px;">
            Les modèles IA existants peuvent produire des conclusions médicales convaincantes mais pas fiables,
            sans niveau de confiance, sans logs, sans gestion de l'incertitude. Le risque : une confiance aveugle dans une prédiction mal calibrée.
        </p>
        <p style="color:#E8F4FF; font-size:1rem; line-height:1.7; margin-bottom:0;">
            Scan-R répond à ce défi en proposant un prototype qui structure sa sortie,
            explicite son incertitude et permet une analyse d'erreurs, conçu pour les radiologues, internes et
            personnel médical du CHU Pellegrin.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    #  Équipe
    st.markdown("#### L'équipe")
    team = [
        ("Guillaume Pousse", "Responsable MOE / solution technique"),
        ("Jennifer Rakotoarinia", "Chef de projet"),
        ("Thomas Rychlewski", "Responsable MOA / besoins utilisateurs"),
        ("Tiphaine Peran", "Responsable documentation et soutenance"),
        ("Timothée Robin", "Responsable parties prenantes, gouvernance et communication"),
        ("Yéléna Sainte-Rose", "Responsable risques, budget et qualité"),
    ]
    cols = st.columns(3)
    for i, (name, role) in enumerate(team):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:var(--bg-card); border:1px solid var(--bg-card2); border-radius:12px;
                        padding:16px; text-align:center; margin-bottom:12px;">
                <div style="font-weight:700; color:#E8F4FF; font-size:0.95rem;">{name}</div>
                <div style="color:#8AB4D4; font-size:0.8rem; margin-top:4px;">{role}</div>
                <div style="color:#8AB4D4; font-size:0.75rem; margin-top:2px;">EFREI Paris — I1</div>
            </div>
            """, unsafe_allow_html=True)

#
# PAGE ANALYSE
#
elif st.session_state.page == "analyse":
    st.title("Espace Analyse — professionnels de santé")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])
    mode = st.selectbox("Mode", ["baseline", "improved"])

    if uploaded:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(Image.open(tmp_path), caption="Image uploadée", use_container_width=True)
        with col2:
            pred = apply_safety_guardrails(toy_predict(uploaded.name, mode=mode))
            st.session_state.analyse_count += 1
            save_run(uploaded.name, pred, mode)

            st.metric("Classe", pred["predicted_class"])

            conf = pred["confidence"]
            if conf >= 0.75:
                badge_color = "#2ecc71"
                badge_label = "Élevée"
            elif conf >= 0.55:
                badge_color = "#f39c12"
                badge_label = "Modérée"
            else:
                badge_color = "#e74c3c"
                badge_label = "Faible"

            st.markdown(f"""
            <div style="background:{badge_color}22; border:1px solid {badge_color};
                 border-radius:10px; padding:12px 16px; margin-bottom:12px;">
                <div style="font-size:0.75rem; color:#8AB4D4; text-transform:uppercase; letter-spacing:0.05em;">Confiance</div>
                <div style="font-size:1.4rem; font-weight:700; color:{badge_color};">{round(conf*100)}% — {badge_label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("**Observations**", pred["visual_evidence"])
            st.write("**Justification**", pred["justification"])
            st.write("**Limites**", pred["limitations"])
            st.json(pred)
    else:
        st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")

#
# PAGE RAPPORTS
#
elif st.session_state.page == "rapports":

    #  Vue détail d'un rapport
    if "rapport_detail" in st.session_state and st.session_state.rapport_detail is not None:
        run_id = st.session_state.rapport_detail
        con = get_connection()
        cur = con.cursor()
        cur.execute("SELECT id, image_path, model_name, prompt_version, prediction_json, predicted_class, confidence, latency_ms, created_at FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        con.close()

        if row:
            _, image_path, model_name, prompt_version, prediction_json, predicted_class, confidence, latency_ms, created_at = row
            pred = json.loads(prediction_json)
            conf_pct = round(confidence * 100)
            if confidence >= 0.75:
                badge_color = "#2ecc71"
                badge_label = "Élevée"
            elif confidence >= 0.55:
                badge_color = "#f39c12"
                badge_label = "Modérée"
            else:
                badge_color = "#e74c3c"
                badge_label = "Faible"

            if st.button("← Retour aux rapports"):
                st.session_state.rapport_detail = None
                st.rerun()

            st.title(f"Rapport — {image_path}")
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                # Chercher l'image
                possible_paths = [
                    Path("images") / image_path,
                    Path("data/sample_images") / image_path,
                    Path(image_path),
                ]
                img_found = None
                for p in possible_paths:
                    if p.exists():
                        img_found = p
                        break
                if img_found:
                    st.image(str(img_found), caption=image_path, use_container_width=True)
                else:
                    st.markdown("""
                    <div style="background:#112240; border:1px solid #1A3A5C; border-radius:10px;
                                padding:40px; text-align:center; color:#8AB4D4;">
                        Image non disponible
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.metric("Classe", predicted_class)
                st.markdown(f"""
                <div style="background:{badge_color}22; border:1px solid {badge_color};
                     border-radius:10px; padding:12px 16px; margin-bottom:12px;">
                    <div style="font-size:0.75rem; color:#8AB4D4; text-transform:uppercase; letter-spacing:0.05em;">Confiance</div>
                    <div style="font-size:1.4rem; font-weight:700; color:{badge_color};">{conf_pct}% — {badge_label}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Date :** {created_at[:19]}")
                st.markdown(f"**Modèle :** {model_name}")
                st.write("**Observations**", pred.get("visual_evidence", "—"))
                st.write("**Justification**", pred.get("justification", "—"))
                st.write("**Limites**", pred.get("limitations", "—"))

    #  Liste des rapports
    else:
        st.title("Espace Rapports — historique des analyses")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        runs = load_runs()
        if not runs:
            st.info("Aucune analyse enregistrée pour le moment.")
        else:
            st.markdown(f"**{len(runs)} analyse(s) enregistrée(s)**")
            for run in runs:
                run_id, image_path, predicted_class, confidence, model_name, created_at = run
                conf_pct = round(confidence * 100)
                if confidence >= 0.75:
                    color = "#2ecc71"
                elif confidence >= 0.55:
                    color = "#f39c12"
                else:
                    color = "#e74c3c"

                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(f"""
                    <div style="background:#112240; border:1px solid #1A3A5C; border-radius:10px;
                                padding:14px 18px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <div style="font-weight:700; color:#E8F4FF;">{image_path}</div>
                                <div style="color:#8AB4D4; font-size:0.82rem;">{created_at[:19]} — {model_name}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-weight:700; color:#E8F4FF;">{predicted_class}</div>
                                <div style="color:{color}; font-weight:700;">{conf_pct}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_btn:
                    st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
                    if st.button("Voir", key=f"btn_{run_id}"):
                        st.session_state.rapport_detail = run_id
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

#
# PAGE APPRENTISSAGE
#
elif st.session_state.page == "apprentissage":
    st.title("Espace Apprentissage — personnel médical en formation")
    st.markdown("<p style='color:#8AB4D4;'>Explorez des exemples de radiographies synthétiques annotées pour comprendre les différentes classes.</p>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    LABELS = {
        "normal": ("Normale", "#2ecc71", "Aucune opacité détectée. Les champs pulmonaires apparaissent clairs et homogènes."),
        "suspected_opacity": ("Opacité suspectée", "#f39c12", "Une zone d'opacité localisée est visible dans le champ pulmonaire, compatible avec une pneumonie."),
        "uncertain": ("Incertain", "#e74c3c", "Qualité d'image insuffisante ou signes ambigus. Une analyse complémentaire est recommandée."),
    }

    sample_images = list(SAMPLE_DIR.glob("*.png")) + list(SAMPLE_DIR.glob("*.jpg"))
    sample_images = [p for p in sample_images if p.parent.name == "images"]

    if not sample_images:
        st.info("Aucune image synthétique trouvée dans le dossier images/.")
    else:
        cols = st.columns(3)
        for i, img_path in enumerate(sample_images):
            name = img_path.name.lower()
            if "normal" in name:
                key = "normal"
            elif "suspected_opacity" in name or "opacity" in name:
                key = "suspected_opacity"
            else:
                key = "uncertain"

            label, color, description = LABELS[key]
            with cols[i % 3]:
                st.image(str(img_path), use_container_width=True)
                st.markdown(f"""
                <div class="sample-card">
                    <div style="font-weight:700; color:{color}; margin-bottom:6px;">{label}</div>
                    <div style="color:#8AB4D4; font-size:0.82rem;">{description}</div>
                </div>
                """, unsafe_allow_html=True)