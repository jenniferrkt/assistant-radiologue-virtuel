from __future__ import annotations

import tempfile
from pathlib import Path
import streamlit as st
from PIL import Image


from src.inference import toy_predict, vlm_predict
from src.guardrails import apply_safety_guardrails


import base64




st.set_page_config(page_title="Arvi-RX", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg-main:    #0D1B2A;
    --bg-card:    #112240;
    --bg-card2:   #1A3A5C;
    --accent:     #2A8FD4;
    --text-main:  #E8F4FF;
    --text-muted: #8AB4D4;
}
.stApp { background-color: var(--bg-main); color: var(--text-main); }
#MainMenu, footer { visibility: hidden; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-weight: 700 !important; }
.stSelectbox > div > div, .stFileUploader > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--bg-card2) !important;
    border-radius: 8px !important;
}
.logo-top-right {
    position: fixed;
    top: 55px;
    right: 24px;
    z-index: 9999;
}
.logo-top-right img {
    width: 200px;
    height: 200px;
    border-radius: 28px;
}
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--bg-card2);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
}
.step-icon { font-size: 2rem; margin-bottom: 10px; }
.step-title { font-weight: 700; color: var(--text-main); font-size: 1rem; margin-bottom: 6px; }
.step-desc { color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; }
.divider { border: none; border-top: 1px solid var(--bg-card2); margin: 28px 0; }
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
    {load_logo("images/assets/logo_temp.png")}
</div>
""", unsafe_allow_html=True)

# ── Compteur de session ───────────────────────────────────────────────────────
if "analyse_count" not in st.session_state:
    st.session_state.analyse_count = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Session en cours")
    st.metric("Images analysées", st.session_state.analyse_count)
    st.markdown("---")
    st.markdown("**Arvi-RX** v0.1 — prototype pédagogique")

# ── Titre ─────────────────────────────────────────────────────────────────────
st.title("Assistant radiologue virtuel — prototype pédagogique")

# ── Section comment ça marche ─────────────────────────────────────────────────
st.markdown("#### Comment ça marche ?")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">🩻</div>
        <div class="step-title">1. Uploader une radio</div>
        <div class="step-desc">Déposez une radiographie thoracique frontale au format PNG ou JPG.</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">🤖</div>
        <div class="step-title">2. Analyse IA</div>
        <div class="step-desc">Le modèle analyse l'image et détecte les signes de pneumonie.</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">📋</div>
        <div class="step-title">3. Résultats</div>
        <div class="step-desc">Classe prédite, score de confiance, observations et justification.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Code du prof intact ───────────────────────────────────────────────────────
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
            
        else:
            # On détermine quel fichier de prompt charger
            nom_fichier_prompt = "baseline_prompt.txt" if moteur == "MedGemma 4B - Baseline" else "improved_prompt.txt"
            chemin_prompt = f"./prompts/{nom_fichier_prompt}"
            
            # Chargement sécurisé du prompt
            try:
                with open(chemin_prompt, "r", encoding="utf-8") as f:
                    prompt_content = f.read()
            except FileNotFoundError:
                st.error(f"Erreur : Le fichier de prompt '{nom_fichier_prompt}' est introuvable dans le dossier 'prompts/'.")
                # Prompt de secours minimaliste en cas de problème
                prompt_content = "Analyse cette radiographie. Renvoie uniquement un JSON valide avec les clés : image_quality, predicted_class, confidence, visual_evidence, justification, limitations, warning."

            # Importation dynamique et appel de la vraie IA
            from src.inference import vlm_predict
            raw_pred = vlm_predict(tmp_path, prompt=prompt_content)

        # 2. Application des garde-fous de sécurité communs
        pred = apply_safety_guardrails(raw_pred)
        
        # 3. Affichage des résultats
        st.session_state.analyse_count += 1
        st.metric("Classe", pred.get("predicted_class", "Erreur"))
        st.metric("Confiance", pred.get("confidence", 0.0))
        st.write("**Observations**", pred.get("visual_evidence", ["Non disponible"]))
        st.write("**Justification**", pred.get("justification", "Non disponible"))
        st.write("**Limites**", pred.get("limitations", ["Non disponible"]))
        
        # Affichage élégant de l'alerte de sécurité si présente
        if "warning" in pred and pred["warning"]:
            st.warning(pred["warning"])
            
        st.json(pred)
else:
    st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")