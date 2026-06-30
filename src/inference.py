from __future__ import annotations

from pathlib import Path
import time
import json
import base64
from typing import Any
from openai import OpenAI

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# ---------------------------------------------------------
# CONFIGURATION DU SERVEUR LOCAL (LM Studio ou Ollama)
# ---------------------------------------------------------

LOCAL_SERVER_URL = "http://localhost:1234/v1"
MODEL_NAME = "medgemma-4b"

client = OpenAI(base_url=LOCAL_SERVER_URL, api_key="not-needed")


def encode_image_to_base64(image_path: str | Path) -> str:
    """Convertit l'image en chaîne base64 pour l'envoi via l'API locale."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.
    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def vlm_predict(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Appel au modèle VLM local (MedGemma) via l'API OpenAI compatible.
    
    Prend l'image, la passe au modèle avec le prompt (issu de prompts/improved_prompt),
    et retourne le dictionnaire JSON formaté avec les métadonnées de latence.
    """
    start = time.perf_counter()
    quality = basic_quality_flag(image_path) # On peut toujours le calculer en amont

    try:
        base64_image = encode_image_to_base64(image_path)

        # Appel au modèle
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyse cette radiographie selon les règles établies et fournis uniquement le JSON."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1, # Très important pour limiter les hallucinations
        )

        # Récupération et nettoyage de la réponse (gestion des balises markdown ```json)
        raw_output = response.choices[0].message.content
        cleaned_output = raw_output.replace("```json", "").replace("```", "").strip()
        
        # Parse du JSON généré par l'IA
        result_dict = json.loads(cleaned_output)

    except json.JSONDecodeError:
        # Fallback si le modèle déraille et ne renvoie pas un JSON valide
        result_dict = {
            "image_quality": quality,
            "predicted_class": "uncertain",
            "confidence": 0.0,
            "visual_evidence": ["Parsing error"],
            "justification": "Le modèle n'a pas respecté le format JSON demandé.",
            "limitations": ["Erreur de formatage du modèle"],
            "warning": WARNING
        }
    except Exception as e:
        # Fallback pour toute erreur de connexion 
        result_dict = {
            "image_quality": quality,
            "predicted_class": "uncertain",
            "confidence": 0.0,
            "visual_evidence": ["Connection error"],
            "justification": f"Erreur de communication avec le serveur local: {str(e)}",
            "limitations": ["Erreur réseau ou serveur inactif"],
            "warning": WARNING
        }

    # Calcul de la latence
    latency_ms = int((time.perf_counter() - start) * 1000)

    # Injection des métadonnées (pour s'assurer qu'elles sont toujours là, 
    # même si l'IA oublie de les générer ou si l'on est tombé dans un bloc except)
    result_dict["latency_ms"] = latency_ms
    result_dict["model_name"] = MODEL_NAME
    result_dict["prompt_version"] = "improved_prompt_v1" # À rendre dynamique si besoin

    # Sécurité supplémentaire exigée par votre prompt:
    # "If confidence < 0.60, predicted_class must be 'uncertain'"
    if result_dict.get("confidence", 0.0) < 0.60:
         result_dict["predicted_class"] = "uncertain"

    return result_dict