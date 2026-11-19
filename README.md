# Industrial PCB Defect Detection & Quality Control using YOLO11 & Fuzzy Logic

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![Scikit-Fuzzy](https://img.shields.io/badge/Fuzzy%20Logic-Mamdani-orange.svg)](https://pythonhosted.org/scikit-fuzzy/)
[![Dataset](https://img.shields.io/badge/Dataset-PCB--IND%20v4-success.svg)](https://doi.org/10.5281/zenodo.19723114)

Système complet d'**inspection optique automatisée (AOI)** pour circuits imprimés (PCB), combinant le modèle de détection d'objets **YOLO11m** et un moteur d'aide à la décision par **logique floue (Mamdani)** pour la qualification du niveau de risque et la conformité industrielle.

---

## Architecture du Pipeline

```
Images PCB (ROI 300x300) 
       │
       ▼
[ Détection YOLO11m ] ────► Boîtes englobantes, Classes, Confiances
       │
       ▼
[ Optimisation du Seuil ] ──► Seuil optimal maximisant le F1-Score sur validation
       │
       ▼
[ Moteur Flou Mamdani ] ──► Gravité max + Nombre de défauts + Confiance moyenne
       │
       ▼
[ Décision Opérationnelle ] ──► ACCEPTER | RÉPARER / INSPECTER | REJETER
```

---

## Dataset : PCB-IND v4

Le projet exploite le dataset industriel réel **PCB-IND v4** composé de **4 789 images ROI** (300 × 300 px) réparties en :
- **Train** : 3 833 images
- **Validation** : 478 images
- **Test officiel** : 478 images

### Classes de défauts supervisées :
| ID | Classe | Description & Impact industriel |
|---:|---|---|
| 0 | `mouse_bite` | Entaille sur piste conductrice |
| 1 | `missing_copper` | Manque de cuivre |
| 2 | `scratch` | Rayure superficielle |
| 3 | `spurious_copper` | Cuivre parasite |
| 4 | `copper_burr` | Bavure de cuivre |
| 5 | `stain` | Tache superficielle |
| 6 | `short` | Court-circuit (critique) |
| 7 | `open` | Circuit ouvert (critique) |

---

## Système d'Aide à la Décision Floue (Mamdani)

Pour aller au-delà de la simple détection visuelle, un système d'inférence floue de type Mamdani évalue la criticité globale de la carte :

1. **Variables d'entrée (Antécédents)** :
   - `nombre` : Nombre de défauts détectés sur la ROI [0 à 12] (*faible*, *moyen*, *élevé*).
   - `gravite` : Score de sévérité maximale selon la nature des défauts [0 à 10] (*faible*, *moyenne*, *critique*). Les courts-circuits (`short`) et coupures (`open`) ont le poids le plus élevé.
   - `confiance` : Indice de certitude moyen des détections [0 à 1] (*faible*, *moyenne*, *forte*).
2. **Variable de sortie (Conséquent)** :
   - `risque` : Indice de risque calculé [0 à 100] (*faible*, *moyen*, *élevé*).
3. **Qualification finale** :
   - Risque < 30 : **ACCEPTER**
   - 30 ≤ Risque < 65 : **RÉPARER / INSPECTER**
   - Risque ≥ 65 : **REJETER**
   - Confiance < 0.45 : **VÉRIFICATION MANUELLE**

---

## Structure du Répertoire

```
.
├── Projet_PCB_IND_YOLO11_Logique_Floue.ipynb   # Notebook complet et propre (code + visualisations)
├── requirements.txt                            # Dépendances Python du projet
├── README.md                                   # Documentation du projet
└── .gitignore                                  # Fichiers exclus du versionnement
```

---

## Installation & Exécution

### 1. Cloner le dépôt
```bash
git clone https://github.com/firasmrabet/PCB-Defect-Detection-YOLO11-Fuzzy-Logic.git
cd PCB-Defect-Detection-YOLO11-Fuzzy-Logic
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer le Notebook
Ouvrez le fichier `Projet_PCB_IND_YOLO11_Logique_Floue.ipynb` sous **VS Code**, **JupyterLab** ou **Google Colab** (avec accélération GPU T4 recommandée pour l'entraînement).

---

## Références
- **Dataset PCB-IND v4** : [Zenodo - DOI: 10.5281/zenodo.19723114](https://doi.org/10.5281/zenodo.19723114)
- **Ultralytics YOLO11** : [Documentation officielle](https://docs.ultralytics.com/)
- **Scikit-Fuzzy** : [Fuzzy Logic Toolbox for Python](https://pythonhosted.org/scikit-fuzzy/)
