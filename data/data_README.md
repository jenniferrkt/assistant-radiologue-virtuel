# Données

Ce dossier contient un jeu **synthétique jouet** destiné à tester l'architecture, les logs, les métriques et l'interface. Il ne s'agit pas d'un dataset médical réel.

Pour un vrai projet, utiliser un dataset autorisé comme RSNA Pneumonia, CheXpert, MIMIC-CXR ou NIH ChestXray, en respectant les licences et les conditions d'accès.

## `synthetic_cases.csv`

Colonnes :

- `case_id`
- `image_path`
- `source`
- `label`
- `split`
- `quality`
- `notes`

## Images synthétiques

Les images dans `sample_images/` imitent grossièrement une radiographie thoracique uniquement pour vérifier les flux de code. Elles ne doivent pas être utilisées pour évaluer une performance médicale.

## Images réelles (RSNA)

Le dossier `sample_images/` contient aussi 20 vraies radiographies issues du dataset Kaggle RSNA Pneumonia (source : iamtapendu/rsna-pneumonia-processed-dataset), nommées `imageXX_<classe>.png`. Ce sont ces 20 images, référencées dans `cases.csv`, qui ont servi à l'évaluation réelle du modèle (voir `eval/error_register_final.csv` et `notebooks/04_error_register_generation.ipynb`).

Les classes ont été assignées manuellement par l'équipe via le nom de fichier (revue visuelle), sans conservation de l'identifiant patient d'origine — il n'est donc pas possible de recroiser rétroactivement avec le fichier de métadonnées officiel Kaggle (`stage2_train_metadata.csv`).

Ne pas confondre avec les images `CXR_SYN_*` du fichier `synthetic_cases.csv`, qui restent purement synthétiques et ne servent qu'à valider le pipeline logiciel.
