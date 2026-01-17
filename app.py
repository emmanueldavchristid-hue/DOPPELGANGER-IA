import streamlit as st
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_distances
import pickle
import os
import time  # ✅ correction: importer le module time
from glob import glob


# ---------------------------
# 1. Configuration Streamlit
# ---------------------------
st.set_page_config(page_title="Doppelganger IA", layout="wide")

st.title("🧍 Doppelganger IA")
st.markdown("**Trouve ton sosie dans une base de visages (LFW + visages personnalisés).**")

st.sidebar.title("⚙️ Paramètres globaux")
mode = st.sidebar.radio("Mode d'entrée", ["🔍 Recherche (upload/caméra)", "📝 Enregistrement uniquement"])
UNKNOWN_THRESHOLD = st.sidebar.slider("Seuil 'inconnu' (distance cosinus)", 0.3, 0.8, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Rappel** : distance cosinus → 0 = très proche, plus petit = plus similaire.")


# ---------------------------
# 2. Chargement des modèles + base
# ---------------------------
def load_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modèles
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Base de données
    if not (os.path.exists("lfw_embeds.npy") and os.path.exists("lfw_meta.pkl")):
        return None, None, None, None, device

    embeds = np.load("lfw_embeds.npy")
    with open("lfw_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    # Assurer la présence du champ 'source'
    if "source" not in meta:
        meta["source"] = ["LFW"] * len(meta["names"])

    return mtcnn, resnet, embeds, meta, device


with st.spinner("Chargement du moteur d'IA..."):
    mtcnn, resnet, embeds, meta, device = load_engine()

if mtcnn is None:
    st.error("Fichiers lfw_embeds.npy ou lfw_meta.pkl introuvables. Lance d'abord le notebook pour les générer.")
    st.stop()

# Stats base
total_faces = len(meta["names"])
n_lfw = meta["source"].count("LFW")
n_custom = meta["source"].count("Custom") if "Custom" in meta["source"] else 0

st.sidebar.markdown("---")
st.sidebar.markdown("**Base de visages**")
st.sidebar.markdown(f"- Total : {total_faces}")
st.sidebar.markdown(f"- LFW : {n_lfw}")
st.sidebar.markdown(f"- Perso : {n_custom}")


# ---------------------------
# 2.5. Fonction de synchronisation custom_faces
# ---------------------------
def sync_custom_faces():
    """Synchronise les images du dossier custom_faces/ avec la base."""
    global embeds, meta

    if not os.path.exists("custom_faces"):
        return 0, 0, []

    image_files = glob("custom_faces/*.jpg") + glob("custom_faces/*.jpeg") + glob("custom_faces/*.png")

    if len(image_files) == 0:
        return 0, 0, []

    # Charger la base actuelle depuis disque
    embeds_old = np.load("lfw_embeds.npy")
    with open("lfw_meta.pkl", "rb") as f:
        meta_old = pickle.load(f)

    if "source" not in meta_old:
        meta_old["source"] = ["LFW"] * len(meta_old["names"])

    new_embeds = []
    new_names = []
    new_images = []
    new_sources = []
    failed = []

    for img_path in image_files:
        try:
            filename = os.path.basename(img_path)
            name = filename.rsplit('_', 1)[0] if '_' in filename else filename.split('.')[0]

            img = Image.open(img_path).convert("RGB")
            face = mtcnn(img)

            if face is None:
                failed.append(filename)
                continue

            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]

            new_embeds.append(emb)
            new_names.append(name)
            new_images.append(np.array(img))
            new_sources.append("Custom")

        except Exception:
            failed.append(filename)

    if len(new_embeds) > 0:
        embeds_new = np.vstack([embeds_old, np.array(new_embeds)])
        meta_old["names"].extend(new_names)
        meta_old["images"].extend(new_images)
        meta_old["source"].extend(new_sources)

        np.save("lfw_embeds.npy", embeds_new)
        with open("lfw_meta.pkl", "wb") as f:
            pickle.dump(meta_old, f)

        # Mettre à jour les variables globales
        embeds = embeds_new
        meta = meta_old

    return len(new_embeds), len(image_files), failed


# Bouton de synchronisation dans la sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Synchroniser custom_faces/"):
    with st.spinner("Synchronisation en cours..."):
        added, total, failed = sync_custom_faces()
        if added > 0:
            st.sidebar.success(f"✅ {added}/{total} visages ajoutés !")
            if len(failed) > 0:
                st.sidebar.warning(f"⚠️ {len(failed)} échecs")
            time.sleep(1)
            st.rerun()
        elif total > 0:
            st.sidebar.error(f"❌ Aucun visage détecté sur {total} images")
        else:
            st.sidebar.info("ℹ️ Dossier custom_faces/ vide")


# ---------------------------
# 3. Fonctions principales
# ---------------------------
def find_doppelganger_from_image(image: Image.Image, top_k=3):
    """Retourne liste de résultats + meilleure distance."""
    face = mtcnn(image)
    if face is None:
        return None, None

    with torch.no_grad():
        q_emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()

    dists = cosine_distances(q_emb, embeds).flatten()
    best_idx = np.argsort(dists)[:top_k]

    results = []
    for idx in best_idx:
        results.append({
            "name": meta["names"][idx],
            "score": float(dists[idx]),
            "image": meta["images"][idx],
            "source": meta["source"][idx]
        })
    return results, float(dists[best_idx[0]])


def save_face(image: Image.Image, label: str = "Custom_user"):
    """Sauvegarde un visage dans custom_faces/ et met à jour la base en mémoire + disque."""
    global embeds, meta

    os.makedirs("custom_faces", exist_ok=True)
    filename = f"{label}_{int(time.time())}.jpg"  # ✅ correction: time.time()
    save_path = os.path.join("custom_faces", filename)
    image.save(save_path)

    face = mtcnn(image)
    if face is None:
        return False

    with torch.no_grad():
        new_emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]

    # Charger depuis disque
    embeds_old = np.load("lfw_embeds.npy")
    with open("lfw_meta.pkl", "rb") as f:
        meta_old = pickle.load(f)

    if "source" not in meta_old:
        meta_old["source"] = ["LFW"] * len(meta_old["names"])

    # Mise à jour
    embeds_new = np.vstack([embeds_old, new_emb])
    meta_old["names"].append(label)
    meta_old["images"].append(np.array(image))
    meta_old["source"].append("Custom")

    # Sauvegarde disque
    np.save("lfw_embeds.npy", embeds_new)
    with open("lfw_meta.pkl", "wb") as f:
        pickle.dump(meta_old, f)

    # Mettre à jour les variables globales
    embeds = embeds_new
    meta = meta_old

    return True


# ---------------------------
# 4. Tabs : Recherche vs Base
# ---------------------------
tab_app, tab_db = st.tabs(["🔍 Application de recherche", "📚 Explorer la base"])

with tab_app:
    st.subheader("Mode : recherche de sosie")
    sub_col_left, sub_col_right = st.columns([1, 2])

    with sub_col_left:
        st.markdown("### 1. Entrée image")
        input_mode = st.radio("Source d'image", ["Upload image", "Caméra (snapshot)"], horizontal=True)

        img = None
        if input_mode == "Upload image":
            uploaded_file = st.file_uploader("Choisis une image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Image chargée", use_container_width=True)

        elif input_mode == "Caméra (snapshot)":
            cam_img = st.camera_input("Prends une photo")
            if cam_img is not None:
                img = Image.open(cam_img).convert("RGB")
                st.image(img, caption="Capture caméra", use_container_width=True)

        top_k = st.slider("Nombre de sosies à afficher", 1, 5, 3)

        if img is None:
            st.info("Charge une image ou utilise la caméra pour démarrer.")
        else:
            if mode == "🔍 Recherche (upload/caméra)":
                if st.button("🔍 Chercher mon sosie"):
                    with st.spinner("Analyse de ton visage..."):
                        results, best_score = find_doppelganger_from_image(img, top_k=top_k)

                    if results is None:
                        st.error("Aucun visage détecté. Essaie une photo plus nette ou bien de face.")
                    else:
                        with sub_col_right:
                            st.markdown("### 2. Résultats")

                            if best_score > UNKNOWN_THRESHOLD:
                                st.warning(
                                    f"Visage considéré comme **non reconnu** "
                                    f"(meilleure distance = {best_score:.3f} > seuil {UNKNOWN_THRESHOLD:.3f})."
                                )
                                new_label = st.text_input(
                                    "Nom / identifiant pour enregistrer ce visage",
                                    value="Unknown_user"
                                )
                                if st.button("💾 Ajouter ce visage à la base"):
                                    ok = save_face(img, label=new_label)
                                    if ok:
                                        st.success("✅ Visage ajouté à la base pour de futures identifications.")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("❌ Impossible de détecter correctement le visage pour l'enregistrer.")
                            else:
                                st.success(
                                    f"Visage reconnu (distance={best_score:.3f} ≤ seuil {UNKNOWN_THRESHOLD:.3f})."
                                )

                            cols = st.columns(top_k)
                            for i, (c, res) in enumerate(zip(cols, results)):
                                with c:
                                    c.image(res["image"], use_container_width=True)
                                    c.markdown(f"**{res['name']}**")
                                    src = res.get("source", "NA")
                                    similarity = max(0.0, (1.0 - res["score"])) * 100
                                    c.caption(
                                        f"Source : {src} \n\n"
                                        f"Distance : {res['score']:.3f} | Similarité ≈ {similarity:.1f}%"
                                    )
                                    if i == 0:
                                        c.success("Meilleur match")

            elif mode == "📝 Enregistrement uniquement":
                st.markdown("### Mode Enregistrement")
                new_label = st.text_input("Nom / identifiant du visage à enregistrer", value="New_user")

                if st.button("💾 Enregistrer ce visage sans recherche"):
                    # Test de détection d'abord
                    face_test = mtcnn(img)
                    if face_test is None:
                        st.error("❌ Aucun visage détecté ! Essaie avec :")
                        st.markdown("""
                        - Une photo plus nette
                        - Un visage bien centré et de face
                        - Un bon éclairage
                        - Une résolution suffisante
                        """)
                    else:
                        ok = save_face(img, label=new_label)
                        if ok:
                            st.success(f"✅ Visage '{new_label}' enregistré dans la base.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Erreur lors de l'enregistrement.")


with tab_db:
    st.subheader("Aperçu de la base de visages")
    st.write(f"Nombre total de visages : {len(meta['names'])}")

    # Filtrage par source
    filter_source = st.radio("Filtrer par source", ["Tous", "LFW uniquement", "Custom uniquement"], horizontal=True)

    # Préparation des données filtrées
    if filter_source == "LFW uniquement":
        indices = [i for i, s in enumerate(meta["source"]) if s == "LFW"]
    elif filter_source == "Custom uniquement":
        indices = [i for i, s in enumerate(meta["source"]) if s == "Custom"]
    else:
        indices = list(range(len(meta["names"])))

    st.write(f"Visages affichés : {len(indices)}")

    # Affichage visuel
    if len(indices) > 0:
        n_show = st.slider(
            "Nombre de visages à afficher visuellement",
            1,
            min(20, len(indices)),
            min(6, len(indices))
        )

        cols = st.columns(n_show)
        for i, c in enumerate(cols):
            if i >= len(indices):
                break
            idx = indices[i]
            with c:
                c.image(meta["images"][idx], use_container_width=True)
                label = meta["names"][idx]
                src = meta["source"][idx] if "source" in meta else "NA"
                c.caption(f"{label} ({src})")
    else:
        st.info("Aucun visage dans cette catégorie.")


# ---------------------------
# 5. Infos sur le projet
# ---------------------------
st.markdown("---")
with st.expander("ℹ️ Détails techniques du modèle"):
    st.markdown("""
    - Détection du visage avec **MTCNN** (Multi-task Cascaded CNN).  
    - Extraction d'empreinte faciale (embedding 512D) avec **InceptionResnetV1 pré-entraîné sur VGGFace2**.  
    - Similarité mesurée par **distance cosinus** dans un espace de dimension 512.  
    - Base initiale : ~3000 visages du dataset **LFW**, enrichie par des visages **Custom** ajoutés à la volée.  
    - Interface en **Streamlit** avec : upload d'image, capture caméra, modes Recherche et Enregistrement, et onglet d'exploration de la base.  
    
    **Pour ajouter des visages manuellement :**
    1. Place tes images (.jpg, .jpeg, .png) dans le dossier `custom_faces/`
    2. Clique sur "🔄 Synchroniser custom_faces/" dans la sidebar
    3. Les nouveaux visages seront automatiquement détectés et ajoutés à la base
    """)
