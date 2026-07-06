# Assistant radiologue virtuel responsable — Scan-R

**Équipe :** DS6-E — Jennifer Rakotoarinia, Yelena Sainte-Rose, Tiphaine Peran, Thomas Rychlewski, Timothé Robin, Guillaume Pousse
**École :** EFREI Paris
**Année académique :** 2025-2026
**Cadrage du projet :** Badr Tajini

## Contexte

Ce dépôt est notre prototype pédagogique d'IA médicale multimodale. L'objectif n'était pas de sortir un modèle spectaculaire, mais d'apprendre à construire une chaîne prudente, traçable et évaluée autour d'une radiographie thoracique frontale : une baseline simple, des garde-fous solides, une évaluation honnête, et des limites assumées.

> **Position non clinique.** Ce dépôt n'est pas un dispositif médical. Il ne doit jamais être utilisé pour diagnostiquer, trier ou orienter un patient. Toute sortie reste un résultat expérimental, à vérifier par un professionnel qualifié.

## Ce que fait le projet

| Élément | Cadrage |
|---|---|
| Entrée | Une radiographie thoracique frontale |
| Sorties | `normal`, `suspected_opacity`, `uncertain` |
| Preuve minimale | JSON valide, warning, logs, métriques, cas d'erreur |
| Données | Synthétiques (test pipeline) et 20 images réelles RSNA Pneumonia (évaluation) |
| Finalité | Prototype éducatif de data/IA, pas une aide au diagnostic réelle |

## Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python eval/run_evaluation.py --mode toy
streamlit run app/streamlit_app.py
```

Une interface Gradio alternative est aussi disponible :

```bash
python app/gradio_app.py
```

## Smoke test avant chaque livraison

Avant une soutenance, un push ou une livraison, on lance ce contrôle court :

```bash
pip install -r requirements-test.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src api app eval finetuning tests
python eval/run_evaluation.py --mode toy \
  --out-dir /tmp/assistant-radio-eval \
  --db-path /tmp/assistant-radio-evidence.sqlite
```

Il vérifie la structure du dépôt, le contrat du dataset synthétique, le schéma de sortie, les garde-fous, l'API de démonstration, la compilation Python et l'évaluation jouet. Le pipeline `.github/workflows/ci.yml` relance automatiquement ces vérifications à chaque push et pull request.

## API de démonstration

```bash
uvicorn api.main:app --reload
```

Exemple :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/sample_images/image01_suspected_opacity.png"
```

La réponse contient une classe, une confiance, des observations visuelles, une justification, des limites et l'avertissement non clinique.

> **Note :** l'API de démonstration (`api/main.py`) tourne actuellement en mode `toy_predict` (prédicteur jouet, sans appel modèle réel). L'inférence réelle (`vlm_predict`, MedGemma via serveur local) est disponible et branchée dans l'interface Streamlit (`app/streamlit_app.py`, sélecteur de mode toy/baseline/improved).

## Organisation du dépôt
```
assistant-radiologue-virtuel/
├── .github/workflows/ci.yml           # Intégration continue
├── api/main.py                        # API FastAPI de démonstration (mode toy)
├── app/
│   ├── gradio_app.py                  # Interface Gradio
│   └── streamlit_app.py               # Interface Streamlit "Scan-R" (toy/baseline/improved)
├── data/
│   ├── sample_images/                 # Images jouet (CXR_SYN_*) + 20 vraies images RSNA (image01-07_*)
│   ├── README.md                      # Description des données synthétiques et réelles
│   ├── cases.csv                      # Les 20 vrais cas RSNA utilisés pour l'évaluation
│   └── synthetic_cases.csv            # Cas synthétiques annotés (test pipeline)
├── docs/                              # Appel d'offre, architecture, éthique, évaluation
├── eval/
│   ├── error_register_final.csv       # Registre d'erreurs consolidé (20 images RSNA)
│   └── run_evaluation.py              # Script d'évaluation
├── finetuning/
│   ├── gemma4_unsloth_lora_stub.py    # Fine-tuning LoRA avec Unsloth - Gemma 4
│   └── medgemma_peft_qlora_stub.py    # Stub expérimental MedGemma/PEFT
├── notebooks/
│   ├── 01_baseline_vlm.ipynb          # (à compléter)
│   ├── 02_prompt_comparison.ipynb     # (à compléter) comparaison V1/V2/V3
│   ├── 03_optional_finetuning_lora.ipynb  # (à compléter) fine-tuning LoRA
│   └── 04_error_register_generation.ipynb # Génération du registre d'erreurs sur les 20 images RSNA
├── prompts/                           # Prompts baseline / few-shot / amélioré + schéma JSON
├── sql/schema.sql                     # Schéma de la base SQLite
├── src/                               # Inférence, garde-fous, métriques, prétraitement, base de données
├── tests/test_repository_smoke.py     # Smoke tests
├── notebook_gemma_trainer.ipynb       # Entraînement Gemma 4 (LoRA, résultats réels)
├── LICENSE
├── pyproject.toml
├── requirements-test.txt
└── requirements.txt
```

## Livrables

| Niveau | Attendu |
|---|---|
| **MUST** | Baseline reproductible, sortie JSON valide, warning obligatoire, logs, métriques, mini-rapport |
| **SHOULD** | Prompt amélioré, règle d'incertitude, comparaison baseline/amélioration, analyse d'erreurs |
| **COULD** | LoRA expérimental, MedGemma/PEFT, localisation visuelle, ablations de prompts |

## Notre travail de fine-tuning : Unsloth + Gemma-4-E4B

En parallèle de l'amélioration des prompts, on a testé une piste plus avancée : le fine-tuning LoRA de Gemma-4-E4B via Unsloth, entraîné sur le vrai dataset Kaggle RSNA Pneumonia téléchargé en entier (~11 Go).

**Résultats obtenus** (50 exemples de validation) :
- Accuracy : 0.800
- Macro-F1 : 0.783

**Limites qu'on assume** : entraîné sur un sous-ensemble du dataset, pas de validation clinique, le champ `confidence` n'est pas fourni par la vérité terrain RSNA d'origine (mode `class_only`).

## Points de vigilance

- Ne pas inventer d'information clinique absente de l'image.
- Ne pas supprimer la classe `uncertain` ; c'est un garde-fou, pas un échec.
- Ne pas afficher uniquement des réussites en soutenance.
- Ne jamais commiter de données patient réelles, identifiantes ou ambiguës.
- Ne pas présenter le prototype comme validé médicalement.

## Licence et sources externes

Le code pédagogique est publié sous licence MIT (voir `LICENSE`). Les datasets externes, modèles et bibliothèques utilisés conservent leurs licences propres : on vérifie et documente les droits d'usage avant toute expérimentation.

**Exigence minimale :** indiquer dans le rapport la source, la version, la licence ou les conditions d'accès, les restrictions de redistribution, les traitements d'anonymisation et les limites d'interprétation. Aucun fichier patient réel, même pseudonymisé, n'est ajouté au dépôt sans autorisation explicite et traçable.
