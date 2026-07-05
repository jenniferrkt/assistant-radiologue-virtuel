from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# On importe maintenant nos deux moteurs d'inférence
from src.inference import toy_predict, vlm_predict
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics
from src.database import insert_run, init_db


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(engine: str, mode: str, db_path: Path) -> tuple[list[dict], dict]:
    # TODO plus tard : Remplacer 'synthetic_cases.csv' par le CSV contenant vos vraies 30 radios RSNA
    cases = read_cases(ROOT / 'data' / 'cases.csv')
    rows = []
    init_db(db_path)
    
    print(f"\n--- Lancement de l'évaluation : Moteur = {engine.upper()} | Mode = {mode.upper()} ---", file=sys.stderr)
    
    for case in cases:
        image_path = ROOT / case['image_path']
        print(f"Analyse en cours : {image_path.name}...", file=sys.stderr)
        
        # 1. Choix du moteur d'inférence
        if engine == 'toy':
            raw_pred = toy_predict(image_path, mode=mode)
        elif engine == 'medgemma':
            # Chargement dynamique du bon prompt (baseline_prompt.txt ou improved_prompt.txt)
            prompt_file = ROOT / 'prompts' / f'{mode}_prompt.txt'
            try:
                prompt_content = prompt_file.read_text(encoding='utf-8')
            except FileNotFoundError:
                print(f"ATTENTION : Le fichier {prompt_file} est introuvable. Arrêt.")
                sys.exit(1)
                
            raw_pred = vlm_predict(image_path, prompt=prompt_content)

        # 2. Application des règles et validation
        pred = apply_safety_guardrails(raw_pred)
        valid, errors = validate_prediction(pred)
        
        # 3. Formatage pour le CSV
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred.get('predicted_class', 'uncertain'),
            'confidence': pred.get('confidence', 0.0),
            'json_valid': valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
        }
        rows.append(row)
        
        # 4. Insertion dans la base SQLite
        insert_run(db_path, case['case_id'], str(image_path), pred)
        
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    # On garde 'toy' comme moteur par défaut
    parser.add_argument('--engine', choices=['toy', 'medgemma'], default='toy')
    # CRUCIAL : On rajoute 'toy' dans les choix de mode pour ne pas casser le test du prof !
    parser.add_argument('--mode', choices=['toy', 'all', 'baseline', 'improved'], default='toy')
    
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    args = parser.parse_args()
    
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- LOGIQUE DE RÉTROCOMPATIBILITÉ ---
    engine = args.engine
    mode_arg = args.mode

    # Si le prof a lancé '--mode toy', on s'adapte automatiquement :
    # Cela veut dire qu'on utilise le moteur 'toy' et qu'on va faire tourner 'baseline' et 'improved'
    if mode_arg == 'toy':
        engine = 'toy'
        modes = ['baseline', 'improved']
    elif mode_arg == 'all':
        modes = ['baseline', 'improved']
    else:
        modes = [mode_arg]
    # -------------------------------------
    
    summary = []
    for mode in modes:
        # On passe le bon 'engine' et le bon 'mode' détectés
        rows, metrics = run(engine, mode, args.db_path)
        
        # Sauvegarde des résultats
        prefix = f"{engine}_{mode}"
        write_csv(out_dir / f'{prefix}_predictions.csv', rows)
        (out_dir / f'{prefix}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        
        summary.append({'engine': engine, 'mode': mode, **metrics})
        
    write_csv(out_dir / 'before_after_summary.csv', summary)    
    print("\nRÉSUMÉ DES PERFORMANCES :", file=sys.stderr)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()