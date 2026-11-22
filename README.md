# Industrial PCB Defect Detection & Quality Control using YOLO11 & Fuzzy Logic

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![Fuzzy Logic](https://img.shields.io/badge/Fuzzy%20Logic-Mamdani-orange.svg)](https://pythonhosted.org/scikit-fuzzy/)
[![Dataset](https://img.shields.io/badge/Dataset-PCB--IND%20v4-success.svg)](https://doi.org/10.5281/zenodo.19723114)
[![Standard](https://img.shields.io/badge/Quality%20Standard-IPC--A--610-blueviolet.svg)](https://www.ipc.org/)

Système d'**inspection optique automatisée (AOI)** pour circuits imprimés (PCB), combinant le modèle de détection d'objets **YOLO11m** et un moteur d'aide à la décision par **logique floue (Mamdani)** avec **architecture industrielle à double seuil** pour une politique "Zéro Défaut" (Zero Defect Policy).

---

## 🏭 Stratégie Industrielle : Double Seuil & Politique Zéro Défaut

Dans les lignes d'assemblage CMS (SMT) réelles, deux pièges majeurs menacent la rentabilité :
1. **Le faux négatif (critique)** : un PCB défectueux mais peu contrasté échappe à la machine et part chez le client final.
2. **La fausse alerte abusive (coûteuse)** : rejeter et mettre au rebut des cartes conformes à la moindre ombre.

Pour résoudre ce dilemme, le système implémente une **architecture à double seuil** conforme aux meilleures pratiques industrielles (Post-AOI Review Station) :

```
                                  Image PCB (ROI 300x300)
                                             │
                                             ▼
                     [ Détection YOLO11m sensible (conf >= 0.10) ]
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
        Aucune détection (< 0.10)                            Détections trouvées
                    │                                                 │
                    ▼                                                 ▼
        ┌───────────────────────┐                     ┌───────────────────────────────┐
        │       ACCEPTER        │                     │   Séparation des Détections   │
        │  Carte 100% conforme  │                     └───────────────┬───────────────┘
        └───────────────────────┘                                     │
                                            ┌─────────────────────────┴─────────────────────────┐
                                            ▼                                                   ▼
                               Détections Suspectes                                    Défauts Confirmés
                             (0.10 <= conf < 0.55)                                      (conf >= 0.55)
                                            │                                                   │
                                            ▼                                                   ▼
                             ┌─────────────────────────────┐                    ┌───────────────────────────────┐
                             │    VÉRIFICATION MANUELLE    │                    │     Moteur Flou (Mamdani)     │
                             │ (Post-AOI Review Station)   │                    │ (Nombre, Gravité, Confiance)  │
                             └─────────────────────────────┘                    └───────────────┬───────────────┘
                                                                                                │
                                                                   ┌────────────────────────────┴────────────────────────────┐
                                                                   ▼                                                         ▼
                                                       ┌───────────────────────┐                                 ┌───────────────────────┐
                                                       │  RÉPARER / INSPECTER  │                                 │        REJETER        │
                                                       │ (30 <= Risque < 65)   │                                 │    (Risque >= 65)     │
                                                       └───────────────────────┘                                 └───────────────────────┘
```

---

## 🚀 Performances & Résultats (Modèle Optimisé - Batch 8)

L'entraînement optimisé avec une taille de lot de **Batch = 8** sur GPU NVIDIA RTX (8 Go VRAM) a permis une convergence stable et rapide, atteignant des performances de pointe sur l'ensemble de test officiel :

| Métrique | Valeur Test | Description |
|---|:---:|---|
| **mAP@50** | **72.34 %** | Précision moyenne à IoU 0.50 |
| **mAP@50-95** | **49.73 %** | Précision moyenne sur l'intervalle IoU [0.50, 0.95] |
| **Précision globale** | **84.16 %** | Taux de vrais positifs parmi les prédictions (seuil 0.550) |
| **Rappel global** | **62.16 %** | Taux de détection des défauts réels (seuil 0.550) |
| **F1-Score optimal** | **71.50 %** | Compromis optimal précision-rappel |
| **Seuil de certitude opérationnel** | **0.550** | Optimisé automatiquement sur le jeu de validation |
| **Seuil de suspicion préventif** | **0.100** | Capture toutes les anomalies diffuses pour contrôle humain |
| **Latence d'inférence** | **19.00 ms / image** | Traitement temps réel (~52 FPS) |

### Performances détaillées par classe de défaut (Seuil optimal 0.550) :

| Classe | Précision | Rappel | F1-Score | TP | FP | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `scratch` (Rayure) | **97.96 %** | 52.17 % | 68.09 % | 48 | 1 | 44 |
| `copper_burr` (Bavure cuivre) | **84.93 %** | **81.58 %** | **83.22 %** | 62 | 11 | 14 |
| `missing_copper` (Manque cuivre) | **83.06 %** | **78.03 %** | **80.47 %** | 103 | 21 | 29 |
| `short` (Court-circuit) | **83.78 %** | 65.96 % | 73.81 % | 31 | 6 | 16 |
| `mouse_bite` (Entaille) | **82.26 %** | 56.04 % | 66.67 % | 51 | 11 | 40 |
| `spurious_copper` (Cuivre parasite) | **76.67 %** | 53.49 % | 63.01 % | 23 | 7 | 20 |
| `stain` (Tache) | 72.00 % | 34.62 % | 46.75 % | 18 | 7 | 34 |
| `open` (Circuit ouvert) | **100.00 %** | 28.57 % | 44.44 % | 4 | 0 | 10 |

---

## 📦 Dataset : PCB-IND v4

Le projet exploite le dataset industriel réel **PCB-IND v4** composé de **4 789 images ROI** (300 × 300 px) réparties en :
- **Train** : 3 833 images
- **Validation** : 478 images
- **Test officiel** : 478 images

### Classes de défauts supervisées :
| ID | Classe | Criticité | Description |
|---:|---|:---:|---|
| 0 | `mouse_bite` | Moyenne | Entaille sur piste conductrice |
| 1 | `missing_copper` | Moyenne | Manque de cuivre |
| 2 | `scratch` | Faible | Rayure superficielle |
| 3 | `spurious_copper` | Moyenne | Cuivre parasite résiduel |
| 4 | `copper_burr` | Moyenne | Bavure de cuivre |
| 5 | `stain` | Faible | Tache superficielle non conductrice |
| 6 | `short` | **Critique** | Court-circuit entre deux pistes |
| 7 | `open` | **Critique** | Coupure de piste / circuit ouvert |

---

## 🧠 Système d'Aide à la Décision Floue (Mamdani)

Pour aller au-delà de la simple détection visuelle, un système d'inférence floue de type Mamdani qualifie la criticité globale de la carte :

1. **Variables d'entrée (Antécédents)** :
   - `nombre` : Nombre de défauts détectés sur la ROI [0 à 12] (*faible*, *moyen*, *élevé*).
   - `gravite` : Score de sévérité maximale selon la nature des défauts [0 à 10] (*faible*, *moyenne*, *critique*). Les courts-circuits (`short`) et coupures (`open`) ont le poids maximal.
   - `confiance` : Indice de certitude moyen des détections [0 à 1] (*faible*, *moyenne*, *forte*).
2. **Variable de sortie (Conséquent)** :
   - `risque` : Indice de risque calculé [0 à 100] (*faible*, *moyen*, *élevé*).
3. **Qualification industrielle** :
   - **ACCEPTER** (Risque < 30) : Conforme certifié.
   - **VÉRIFICATION MANUELLE** : Présence de doutes/suspicions ou confiance globale < 0.45.
   - **RÉPARER / INSPECTER** (30 ≤ Risque < 65) : Carte réparable (défauts modérés).
   - **REJETER** (Risque ≥ 65) : Défaut bloquant (court-circuit ou cumul sévère).

---

## 📂 Structure du Répertoire

```
.
├── Projet_PCB_IND_YOLO11_Logique_Floue_Batch8.ipynb   # Notebook principal complet (Entraînement Batch 8 + Évaluation + Moteur Flou Industriel)
├── requirements.txt                                  # Dépendances Python du projet
├── README.md                                         # Documentation et rapport de performance industrielle
└── .gitignore                                        # Fichiers exclus (modèles .pt, datasets bruts, runs)
```

---

## 💻 Installation & Exécution

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
Ouvrez `Projet_PCB_IND_YOLO11_Logique_Floue_Batch8.ipynb` sous **VS Code**, **JupyterLab** ou **Google Colab** (avec accélération GPU).

---

## 📚 Références
- **Dataset PCB-IND v4** : [Zenodo - DOI: 10.5281/zenodo.19723114](https://doi.org/10.5281/zenodo.19723114)
- **Ultralytics YOLO11** : [Documentation officielle](https://docs.ultralytics.com/)
- **Scikit-Fuzzy** : [Fuzzy Logic Toolbox for Python](https://pythonhosted.org/scikit-fuzzy/)
- **Norme IPC-A-610** : [Acceptabilité des assemblages électroniques](https://www.ipc.org/)
