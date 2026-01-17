## Sujet 1 : Prédiction du cours d’une action du S&P 500

### Objectif  
Construire un modèle de deep learning pour prédire l’évolution à court terme du cours d’une action ou de l’indice S&P 500 à partir de données historiques, avec une interface interactive et une démonstration visuelle des prédictions.

### Description  
Les étudiants collectent des données historiques du S&P 500 puis développent un modèle séquentiel (LSTM) capable de prévoir les tendances des prix. Ils créeront une interface simple où l’utilisateur peut sélectionner des périodes ou actions, lancer des prédictions et visualiser les résultats graphiquement. Une démonstration fonctionnelle permettra de montrer les capacités du modèle à prévoir l’évolution boursière.

### Tâches clés  
- Collecte et préparation de données historiques (Yahoo Finance, Kaggle, Investing.com)  
- Implémentation d’un modèle LSTM ou similaire  
- Développement d’une interface graphique (ex. application web Flask, Dash, Streamlit)  
- Visualisation interactive des prix réels vs prédits  
- Mise en place d’une démonstration utilisateur simple  

### Compétences développées  
- Séries temporelles et deep learning séquentiel  
- Manipulation de données financières  
- Évaluation des modèles prédictifs  
- Visualisation interactive  

### Sources de données possibles  
- [Yahoo Finance S&P 500](https://finance.yahoo.com)  
- [Kaggle S&P 500 Stock Data](https://www.kaggle.com)  
- [Investing.com Historical Data](https://www.investing.com)  

---

## Sujet 2 : Doppleganger — IA qui trouve la personne qui vous ressemble le plus

### Objectif  
Créer une IA capable d’identifier dans une base d’images la personne la plus ressemblante à une photo donnée, grâce à un modèle d’extraction et comparaison de caractéristiques faciales, accompagné d’une interface conviviale.

### Description  
À partir des datasets publics comme LFW ou Tufts Face Database, les étudiants appliqueront un CNN pré-entraîné pour générer des embeddings faciaux. Une interface permettra à un utilisateur de charger sa photo et d’obtenir en retour les visages les plus similaires dans la base. Une démonstration affichera la recherche et l’affichage des résultats.

### Tâches clés  
- Prétraitement et extraction de caractéristiques faciales  
- Développement de la fonction de similarité (distance cosinus, euclidienne)  
- Interface pour téléchargement d’image et affichage des résultats  
- Démo interactive montrant la recherche de “doppleganger”  

### Compétences développées  
- Vision par ordinateur et extraction de features  
- Utilisation de modèles pré-entraînés (transfer learning)  
- Calcul de similarité dans des espaces vectoriels  
- Développement d’interface utilisateur  

### Sources de données possibles  
- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)  
- [Tufts Face Database](https://web.archive.org/web/20120414172630/http://vision.cs.tufts.edu/data/datasets.html)  
- [Google Facial Expression Comparison Dataset]  

---

## Sujet 3 : Système de recommandation et d’achats

### Objectif  
Développer un système de recommandation de produits basé sur les interactions utilisateurs, avec une interface utilisateur permettant de tester la pertinence des recommandations.

### Description  
En s’appuyant sur des datasets comme MovieLens ou Amazon Reviews, les étudiants implémenteront un modèle de filtrage collaboratif ou basé sur contenu pour fournir des suggestions personnalisées. Une interface web affichera les recommandations pour un utilisateur donné, recueillant ainsi un feedback potentiel. Une démonstration permettra d’illustrer l’expérience utilisateur en conditions réelles.

### Tâches clés  
- Analyse et préparation des données utilisateurs-produits  
- Implémentation du modèle de recommandation  
- Conception d’une interface interactive (recherche, affichage recommandations)  
- Mise en place d’une démonstration utilisateur pour tester et valider  

### Compétences développées  
- Systèmes de recommandation  
- Manipulation des données utilisateurs et produits  
- Apprentissage profond appliqué au filtrage collaboratif  
- Développement d’interface utilisateur  

### Sources de données possibles  
- [MovieLens dataset](https://grouplens.org/datasets/movielens/)  
- Amazon Reviews dataset (disponible sur Kaggle)  
- Yelp dataset  

---

# Importance de créer une interface et une démonstration

Créer une interface utilisateur interactive et une démonstration fonctionnelle est fortement recommandé pour :

- Rendre le projet accessible, compréhensible et concret pour les évaluateurs et utilisateurs  
- Permettre une expérimentation directe des modèles et visualiser les résultats en temps réel  
- Exposer l’ensemble du pipeline fonctionnel de bout en bout : de la collecte de données à l’interaction utilisateur  
- Renforcer l’expérience pédagogique et le réalisme du projet
