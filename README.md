# 🧬 Doppelganger IA

Application de reconnaissance faciale qui trouve les visages les plus proches d'une photo donnée, à partir d'une base de référence, grâce à un modèle de deep learning.

## ✨ Fonctionnalités

- **Recherche de sosie** : importe une photo ou utilise ta caméra pour trouver les visages les plus proches dans la base.
- **Enregistrement de visages** : ajoute de nouvelles personnes à ta propre base personnalisée.
- **Exploration de la base** : filtre et parcourt tous les visages indexés (LFW + visages personnalisés).
- **Synchronisation automatique** : les photos ajoutées manuellement dans `custom_faces/` sont encodées et indexées en un clic.
- **Seuil de reconnaissance ajustable** pour contrôler la sensibilité du matching.

## 🧠 Architecture technique

```
Image
  │
  ▼
MTCNN               → détection du visage
  │
  ▼
InceptionResnetV1    → embedding facial
(VGGFace2)
  │
  ▼
Distance cosinus     → comparaison avec la base
  │
  ▼
Top-K correspondances
```

## 🛠️ Stack

- [Streamlit](https://streamlit.io/) — interface web
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN + InceptionResnetV1 (VGGFace2)
- PyTorch, NumPy, scikit-learn, Pillow

## 📦 Installation

```bash
git clone https://github.com/emmanueldavchristid-hue/DOPPELGANGER-IA.git
cd DOPPELGANGER-IA
pip install -r requirements.txt
```

## ▶️ Lancer l'application

```bash
streamlit run doppelganger_ia.py
```

L'application est ensuite accessible sur [http://localhost:8501](http://localhost:8501).

## 📁 Structure du projet

```
DOPPELGANGER-IA/
├── doppelganger_ia.py     # Application principale (Streamlit)
├── lfw_embeds.npy         # Embeddings pré-calculés (base LFW)
├── lfw_meta.pkl           # Métadonnées associées (noms, images, source)
├── custom_faces/          # Visages ajoutés manuellement (auto-généré)
├── requirements.txt
└── README.md
```

## 🔎 Comment ça marche

1. Une photo est importée ou capturée via la caméra.
2. **MTCNN** détecte et recadre le visage présent dans l'image.
3. **InceptionResnetV1** (pré-entraîné sur VGGFace2) transforme le visage en un vecteur d'embedding.
4. Cet embedding est comparé à ceux de la base via une **distance cosinus**.
5. Les correspondances les plus proches (Top-K) sont affichées, avec un niveau de proximité qualitatif (Très proche / Proche / Éloigné) selon le seuil configuré.

## ⚠️ Limites

- La qualité du matching dépend fortement de la qualité et du cadrage de la photo (visage bien visible, bon éclairage).
- Le seuil de reconnaissance est empirique et peut nécessiter un ajustement selon la base utilisée.
- Ce projet est à visée pédagogique/démonstrative et ne doit pas être utilisé pour de l'identification de personnes dans un contexte réel sans consentement.

## 👤 Auteur

Développé par **Christ-Emmanuel Mouhi (Jason)** — Data Scientist & AI Engineer.
Portfolio : [portfolio-mouhi.vercel.app](https://portfolio-mouhi.vercel.app)
