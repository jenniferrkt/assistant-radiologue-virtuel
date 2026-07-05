# Assistant radiologue virtuel responsable

**Auteur :** Badr Tajini
**Solution Delivery — Filière Data**
**École :** EFREI
**Année académique :** 2025-2026

## Contexte

Ce dépôt est notre prototype pédagogique d'IA médicale multimodale. L'objectif n'était pas de sortir un modèle spectaculaire, mais d'apprendre à construire une chaîne prudente, traçable et évaluée autour d'une radiographie thoracique frontale : une baseline simple, des garde-fous solides, une évaluation honnête, et des limites assumées.

> **Position non clinique.** Ce dépôt n'est pas un dispositif médical. Il ne doit jamais être utilisé pour diagnostiquer, trier ou orienter un patient. Toute sortie reste un résultat expérimental, à vérifier par un professionnel qualifié.

## Ce que fait le projet

| Élément | Cadrage |
|---|---|
| Entrée | Une radiographie thoracique frontale |
| Sorties | `normal`, `suspected_opacity`, `uncertain` |
| Preuve minimale | JSON valide, warning, logs, métriques, cas d'erreur |
| Données | Synthétiques ou publiques, autorisées et dé-identifiées |
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
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
```

La réponse contient une classe, une confiance, des observations visuelles, une justification, des limites et l'avertissement non clinique.

## Organisation du dépôt
```
assistant-radiologue-virtuel/
├── .github/workflows/ci.yml           # Intégration continue
├── api/main.py                        # API FastAPI de démonstration
├── app/
│   ├── gradio_app.py                  # Interface Gradio
│   └── streamlit_app.py               # Interface Streamlit
├── data/
│   ├── sample_images/                 # Images jouet pour tests et démos
│   ├── README.md                      # Description du dataset synthétique
│   └── synthetic_cases.csv            # Cas synthétiques annotés
├── docs/                              # Appel d'offre, architecture, éthique, évaluation
├── eval/
│   ├── error_register_final.csv       # Registre d'erreurs consolidé
│   └── run_evaluation.py              # Script d'évaluation
├── finetuning/
│   ├── gemma4_unsloth_lora_stub.py    # Fine-tuning LoRA avec Unsloth - Gemma 4
│   └── medgemma_peft_qlora_stub.py    # Stub expérimental MedGemma/PEFT
├── images/                            # Jeux d'images par classe (normal, suspected_opacity, uncertain)
├── notebooks/                         # Baseline, comparaison de prompts, fine-tuning, registre d'erreurs
├── prompts/                           # Prompts baseline / few-shot / amélioré + schéma JSON
├── sql/schema.sql                     # Schéma de la base SQLite
├── src/                               # Inférence, garde-fous, métriques, prétraitement, base de données
├── tests/test_repository_smoke.py     # Smoke tests
├── notebook_gemma_trainer.ipynb       # Entraînement Gemma 4
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

## Le modèle qu'on a utilisé : Unsloth - Gemma 4

Pour la partie fine-tuning expérimental (`finetuning/gemma4_unsloth_lora_stub.py` et `notebook_gemma_trainer.ipynb`), on a choisi **Unsloth - Gemma 4** pour faire du fine-tuning LoRA/QLoRA, mais seulement après avoir mis en place une baseline simple et reproductible — le fine-tuning vient en complément, jamais en remplacement de la baseline.

- **Usage :** fine-tuning LoRA/QLoRA expérimental sur la tâche de classification `normal` / `suspected_opacity` / `uncertain`, à titre pédagogique uniquement.
- **Références citées :** guide Gemma 4, catalogue des modèles Unsloth, blog Unsloth.
- **Limites assumées :** résultats obtenus sur données synthétiques/publiques dé-identifiées, aucune validation clinique, pas de garantie de généralisation à des cas réels.

Comme pour toute ressource externe mobilisée dans le projet, on documente dans le rapport la source exacte, la version utilisée, la licence, les conditions d'accès, les restrictions de redistribution et les limites d'interprétation.

## Points de vigilance

- Ne pas inventer d'information clinique absente de l'image.
- Ne pas supprimer la classe `uncertain` ; c'est un garde-fou, pas un échec.
- Ne pas afficher uniquement des réussites en soutenance.
- Ne jamais commiter de données patient réelles, identifiantes ou ambiguës.
- Ne pas présenter le prototype comme validé médicalement.

## Licence et sources externes

Le code pédagogique est publié sous licence MIT (voir `LICENSE`). Les datasets externes, modèles et bibliothèques utilisés conservent leurs licences propres : on vérifie et documente les droits d'usage avant toute expérimentation.

**Exigence minimale :** indiquer dans le rapport la source, la version, la licence ou les conditions d'accès, les restrictions de redistribution, les traitements d'anonymisation et les limites d'interprétation. Aucun fichier patient réel, même pseudonymisé, n'est ajouté au dépôt sans autorisation explicite et traçable.
