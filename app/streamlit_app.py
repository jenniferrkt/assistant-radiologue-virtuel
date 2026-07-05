from __future__ import annotations

import base64
import sqlite3
import json
import tempfile
from pathlib import Path
from datetime import datetime
import streamlit as st
from PIL import Image

from src.inference import toy_predict, vlm_predict
from src.guardrails import apply_safety_guardrails

st.set_page_config(page_title="Arvi-RX", layout="wide")

DB_PATH = "medical_ai_evidence.sqlite"
SAMPLE_DIR = Path("images")

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "accueil"
if "analyse_count" not in st.session_state:
    st.session_state.analyse_count = 0
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ── Thème dynamique ───────────────────────────────────────────────────────────
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
    --bg-card:    #FFFFFF;
    --bg-card2:   #DCE6F0;
    --accent:     #1E6FB8;
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
}}
.logo-top-right img {{
    width: 276px;
    height: 276px;
    border-radius: 58px;
}}
.step-card {{
    background: var(--bg-card);
    border: 1px solid var(--bg-card2);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
}}
.step-icon {{ font-size: 2rem; margin-bottom: 10px; }}
.step-title {{ font-weight: 700; color: var(--text-main); font-size: 1rem; margin-bottom: 6px; }}
.step-desc {{ color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; }}
.divider {{ border: none; border-top: 1px solid var(--bg-card2); margin: 28px 0; }}
.nav-card {{
    background: var(--bg-card);
    border: 1px solid var(--bg-card2);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s;
    height: 220px;
}}
.nav-card:hover {{ border-color: var(--accent); }}
.nav-card-analyse {{
    background: #1e5aa8;
    border: 1px solid #2A8FD4;
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    cursor: pointer;
    height: 220px;
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
</style>
""", unsafe_allow_html=True)

# ── Logo ──────────────────────────────────────────────────────────────────────
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

# ── DB helpers ────────────────────────────────────────────────────────────────
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
        pred.get("model_name", f"model-{mode}"),
        pred.get("prompt_version", f"{mode}_v1"),
        json.dumps(pred),
        pred.get("predicted_class", "unknown"),
        pred.get("confidence", 0.0),
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

# ── Sidebar navigation ────────────────────────────────────────────────────────
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
    theme_label = "☀️ Mode jour" if st.session_state.theme == "dark" else "🌙 Mode nuit"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.markdown("---")
    st.markdown("### Session en cours")
    st.metric("Images analysées", st.session_state.analyse_count)
    st.markdown("---")
    st.markdown("**Arvi-RX** v0.1 — prototype médical")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "accueil":
    st.title("Arvi-RX — Assistant radiologue virtuel")
    st.markdown("<p style='color:#8AB4D4; font-size:1.1rem;'>Prototype médical d'analyse de radiographies thoraciques par intelligence artificielle</p>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("#### Choisissez votre espace")

    # ── Ligne 1 : Analyse centré ──────────────────────────────────────────────
    col_left, col_centre, col_right = st.columns([1, 2, 1])
    with col_centre:
        st.markdown("""
        <div class="nav-card-analyse">
            <div class="nav-icon">🩻</div>
            <div class="nav-title">Analyse</div>
            <div class="nav-desc">Analysez une radiographie thoracique et obtenez un diagnostic assisté par IA.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Accéder à l'analyse", use_container_width=True):
            st.session_state.page = "analyse"
            st.rerun()

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    # ── Ligne 2 : Rapports + Apprentissage ───────────────────────────────────
    col_l, col_r2, col_r3, col_rr = st.columns([1, 2, 2, 1])
    with col_r2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">📋</div>
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
            <div class="nav-icon">🎓</div>
            <div class="nav-title">Espace Apprentissage</div>
            <div class="nav-desc">Explorez des exemples annotés de radiographies pour vous former à la lecture d'images.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Accéder à l'apprentissage", use_container_width=True):
            st.session_state.page = "apprentissage"
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Comment ça marche ?")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""<div class="step-card"><div class="step-icon">🩻</div>
        <div class="step-title">1. Uploader une radio</div>
        <div class="step-desc">Déposez une radiographie thoracique frontale au format PNG ou JPG.</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""<div class="step-card"><div class="step-icon">🤖</div>
        <div class="step-title">2. Analyse IA</div>
        <div class="step-desc">Le modèle analyse l'image et détecte les signes de pneumonie.</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""<div class="step-card"><div class="step-icon">📋</div>
        <div class="step-title">3. Résultats</div>
        <div class="step-desc">Classe prédite, score de confiance, observations et justification.</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "analyse":
    st.title("Espace Analyse — professionnels de santé")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])
    moteur = st.selectbox(
        "Moteur d'analyse", 
        [
            "Simulation (Test rapide)", 
            "MedGemma 4B - Baseline", 
            "MedGemma 4B - Improved (Recommandé)"
        ]
    )

    if uploaded:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(Image.open(tmp_path), caption="Image uploadée", use_container_width=True)
            
        with col2:
            # 1. Sélection et exécution du bon moteur
            if moteur == "Simulation (Test rapide)":
                raw_pred = toy_predict(tmp_path, mode="baseline")
                mode_for_db = "simulation"
            else:
                nom_fichier_prompt = "baseline_prompt.txt" if moteur == "MedGemma 4B - Baseline" else "improved_prompt.txt"
                chemin_prompt = f"./prompts/{nom_fichier_prompt}"
                
                try:
                    with open(chemin_prompt, "r", encoding="utf-8") as f:
                        prompt_content = f.read()
                except FileNotFoundError:
                    st.error(f"Erreur : Le fichier de prompt '{nom_fichier_prompt}' est introuvable dans le dossier 'prompts/'.")
                    prompt_content = "Analyse cette radiographie. Renvoie uniquement un JSON valide avec les clés : image_quality, predicted_class, confidence, visual_evidence, justification, limitations, warning."

                raw_pred = vlm_predict(tmp_path, prompt=prompt_content)
                mode_for_db = "baseline" if moteur == "MedGemma 4B - Baseline" else "improved"

            # 2. Application des garde-fous de sécurité communs
            pred = apply_safety_guardrails(raw_pred)

            # 3. Sauvegarde et mise à jour de la session
            st.session_state.analyse_count += 1
            save_run(uploaded.name, pred, mode_for_db)

            # 4. Affichage du design
            st.metric("Classe", pred.get("predicted_class", "Erreur"))

            conf = pred.get("confidence", 0.0)
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
            
            # Affichage élégant de l'alerte de sécurité si présente
            if "warning" in pred and pred["warning"]:
                st.warning(pred["warning"])

            st.write("**Observations**", pred.get("visual_evidence", ["Non disponible"]))
            st.write("**Justification**", pred.get("justification", "Non disponible"))
            st.write("**Limites**", pred.get("limitations", ["Non disponible"]))
            st.json(pred)
    else:
        st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE RAPPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "rapports":
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE APPRENTISSAGE
# ══════════════════════════════════════════════════════════════════════════════
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