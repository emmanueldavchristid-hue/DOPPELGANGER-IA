import streamlit as st
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_distances
import pickle
import os
import glob
import hashlib
from io import BytesIO


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Doppelganger IA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDS_PATH = os.path.join(BASE_DIR, "lfw_embeds.npy")
META_PATH = os.path.join(BASE_DIR, "lfw_meta.pkl")
CUSTOM_DIR = os.path.join(BASE_DIR, "custom_faces")

os.makedirs(CUSTOM_DIR, exist_ok=True)


# ============================================================
# HELPER : RENDU HTML SÛR
# ============================================================
# IMPORTANT : Markdown traite tout texte indenté de 4+ espaces
# comme un bloc de code. Comme le HTML ci-dessous suit
# l'indentation du code Python, il était rendu comme du texte
# brut au lieu d'être interprété. Cette fonction "aplatit"
# l'indentation avant l'injection dans st.markdown.

def html(content: str):
    flattened = "\n".join(line.strip() for line in content.strip("\n").splitlines())
    st.markdown(flattened, unsafe_allow_html=True)


# ============================================================
# STYLE PREMIUM
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,0.14), transparent 32%),
        radial-gradient(circle at 90% 10%, rgba(6,182,212,0.12), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(168,85,247,0.08), transparent 35%),
        #070b14;
    color: #f8fafc;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #080c15 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white;
}

/* Hero */

.hero {
    padding: 34px 38px;
    border-radius: 24px;
    margin-bottom: 28px;
    background:
        linear-gradient(135deg, rgba(99,102,241,0.22), rgba(6,182,212,0.12) 55%, rgba(15,23,42,0.8));
    border: 1px solid rgba(129,140,248,0.22);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    animation: fadeInUp 0.5s ease both;
}

.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(99,102,241,0.16);
    border: 1px solid rgba(129,140,248,0.28);
    color: #c7d2fe;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 42px;
    font-weight: 800;
    margin: 12px 0 4px 0;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg, #ffffff, #c7d2fe, #67e8f9, #ffffff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 6s linear infinite;
}

.hero-subtitle {
    color: #a7b0c0;
    font-size: 16px;
    max-width: 750px;
    line-height: 1.6;
}

/* Cards */

.card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 14px 40px rgba(0,0,0,0.20);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.5s ease both;
}

.card:hover {
    transform: translateY(-3px);
    border-color: rgba(129,140,248,0.35);
    box-shadow: 0 18px 50px rgba(0,0,0,0.30);
}

.card-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 5px;
}

.card-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 18px;
}

/* Result principal */

.match-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.82));
    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.30), 0 0 40px rgba(99,102,241,0.07);
    animation: fadeInUp 0.45s ease both;
}

.match-label {
    color: #818cf8;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.match-name {
    font-family: 'Sora', sans-serif;
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin: 5px 0 12px 0;
}

.distance {
    font-size: 14px;
    color: #cbd5e1;
}

.badge-strong {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(16,185,129,0.14);
    border: 1px solid rgba(16,185,129,0.25);
    color: #6ee7b7;
    font-size: 12px;
    font-weight: 700;
}

.badge-medium {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fcd34d;
    font-size: 12px;
    font-weight: 700;
}

.badge-weak {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(148,163,184,0.10);
    border: 1px solid rgba(148,163,184,0.20);
    color: #cbd5e1;
    font-size: 12px;
    font-weight: 700;
}

/* Stats */

.stat-card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    min-height: 110px;
    transition: border-color 0.25s ease;
}

.stat-card:hover {
    border-color: rgba(129,140,248,0.3);
}

.stat-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.stat-value {
    font-family: 'Sora', sans-serif;
    color: white;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
}

/* Section */

.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: white;
    margin: 12px 0 5px 0;
}

.section-subtitle {
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Step cards (À propos) */

.step-card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 22px;
    height: 100%;
    transition: transform 0.25s ease, border-color 0.25s ease;
}

.step-card:hover {
    transform: translateY(-3px);
    border-color: rgba(129,140,248,0.35);
}

.step-icon {
    font-size: 32px;
}

.step-title {
    font-family: 'Sora', sans-serif;
    color: #f8fafc;
    font-size: 17px;
    font-weight: 700;
    margin: 10px 0 6px 0;
}

.step-text {
    color: #94a3b8;
    font-size: 13.5px;
    line-height: 1.6;
}

/* Buttons */

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(129,140,248,0.25);
    font-weight: 700;
    min-height: 45px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: rgba(129,140,248,0.55);
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(99,102,241,0.15);
}

/* Tabs */

button[data-baseweb="tab"] {
    font-weight: 700;
    color: #94a3b8;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #c7d2fe;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.55);
    border-radius: 18px;
}

/* Images */

img {
    border-radius: 16px;
}

/* Dividers */

hr {
    border-color: rgba(255,255,255,0.07);
}

/* Expander */

details {
    background: rgba(15,23,42,0.45);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* Hide Streamlit branding */

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def image_to_bytes_hash(image):
    """Crée une empreinte de l'image pour éviter les doublons."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return hashlib.md5(buffer.getvalue()).hexdigest()


def proximity_level(distance, threshold):
    """Retourne un niveau qualitatif basé sur la distance et le seuil."""
    if distance <= threshold * 0.65:
        return "Très proche", "strong"
    elif distance <= threshold:
        return "Proche", "medium"
    else:
        return "Éloigné", "weak"


def badge_html(label, style):
    return f'<span class="badge-{style}">{label}</span>'


def as_batch(face_tensor):
    """S'assure que le tenseur de visage a une dimension de batch."""
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    return face_tensor


def compute_embedding(face_tensor, resnet, device):
    """Calcule l'embedding facial pour un tenseur de visage détecté."""
    with torch.no_grad():
        face_tensor = as_batch(face_tensor)
        embedding = resnet(face_tensor.to(device)).cpu().numpy()
    return embedding[0]


# ============================================================
# CHARGEMENT DU MOTEUR IA
# ============================================================

@st.cache_resource(show_spinner=False)
def load_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        device=device
    )

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    return mtcnn, resnet, device


@st.cache_data(show_spinner=False)
def load_database():
    embeds = np.load(EMBEDS_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    if "source" not in meta:
        meta["source"] = ["LFW"] * len(meta["names"])

    if "image_hash" not in meta:
        meta["image_hash"] = [""] * len(meta["names"])

    return embeds, meta


# ============================================================
# SYNCHRONISATION DES VISAGES PERSONNALISÉS
# ============================================================

def sync_custom_faces():
    mtcnn, resnet, device = load_engine()
    embeds, meta = load_database()

    custom_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        custom_files.extend(glob.glob(os.path.join(CUSTOM_DIR, ext)))

    if not custom_files:
        return 0

    existing_hashes = set(h for h in meta["image_hash"] if h)

    new_embeddings, new_names, new_images, new_sources, new_hashes = [], [], [], [], []

    for path in custom_files:
        try:
            image = Image.open(path).convert("RGB")
            image_hash = image_to_bytes_hash(image)

            if image_hash in existing_hashes:
                continue

            face = mtcnn(image)
            if face is None:
                continue

            embedding = compute_embedding(face, resnet, device)

            new_embeddings.append(embedding)
            new_names.append(os.path.splitext(os.path.basename(path))[0])
            new_images.append(image)
            new_sources.append("Custom")
            new_hashes.append(image_hash)

        except Exception:
            continue

    if not new_embeddings:
        return 0

    embeds = np.vstack([embeds, np.array(new_embeddings)])
    meta["names"].extend(new_names)
    meta["images"].extend(new_images)
    meta["source"].extend(new_sources)
    meta["image_hash"].extend(new_hashes)

    np.save(EMBEDS_PATH, embeds)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    load_database.clear()
    return len(new_embeddings)


# ============================================================
# SAUVEGARDE D'UN NOUVEAU VISAGE
# ============================================================

def save_face(image, name):
    mtcnn, resnet, device = load_engine()

    face = mtcnn(image)
    if face is None:
        return False, "Aucun visage détecté."

    filename = name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not filename:
        filename = "visage"

    path = os.path.join(CUSTOM_DIR, filename + ".jpg")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(CUSTOM_DIR, f"{filename}_{counter}.jpg")
        counter += 1

    image.save(path, quality=95)

    embedding = compute_embedding(face, resnet, device)

    embeds, meta = load_database()
    embeds = np.vstack([embeds, embedding])

    meta["names"].append(filename)
    meta["images"].append(image)
    meta["source"].append("Custom")
    meta["image_hash"].append(image_to_bytes_hash(image))

    np.save(EMBEDS_PATH, embeds)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    load_database.clear()
    return True, path


# ============================================================
# RECHERCHE DU SOSIE
# ============================================================

def find_doppelganger_from_image(image, top_k=5):
    mtcnn, resnet, device = load_engine()
    embeds, meta = load_database()

    face = mtcnn(image)
    if face is None:
        return None

    query_embedding = compute_embedding(face, resnet, device).reshape(1, -1)

    distances = cosine_distances(query_embedding, embeds)[0]
    indices = np.argsort(distances)[:top_k]

    return [
        {
            "name": meta["names"][idx],
            "distance": float(distances[idx]),
            "image": meta["images"][idx],
            "source": meta["source"][idx],
        }
        for idx in indices
    ]


# ============================================================
# CHARGEMENT
# ============================================================

try:
    mtcnn, resnet, device = load_engine()
    embeds, meta = load_database()
except Exception as e:
    st.error(f"Impossible de charger le moteur IA : {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

html("""
<div class="hero">
    <div class="hero-badge">IA • FACE MATCHING • FACENET</div>
    <div class="hero-title">Doppelganger IA</div>
    <div class="hero-subtitle">
        Analyse ton visage et découvre les visages les plus proches dans une base
        de référence grâce à un modèle de reconnaissance faciale basé sur
        l'apprentissage profond.
    </div>
</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🧬 Doppelganger IA")
    st.caption("Face Matching Engine")
    st.divider()

    st.markdown("### ⚙️ Configuration")

    mode = st.radio("Mode", ["🔍 Recherche", "📝 Enregistrement"], index=0)

    UNKNOWN_THRESHOLD = st.slider(
        "Seuil de reconnaissance",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.01,
        help="Une distance inférieure au seuil est considérée comme une correspondance."
    )

    st.divider()

    total_faces = len(meta["names"])
    lfw_count = sum(1 for x in meta["source"] if x == "LFW")
    custom_count = sum(1 for x in meta["source"] if x == "Custom")

    st.markdown("### 📊 Base de données")

    html(f"""
    <div class="stat-card">
        <div class="stat-label">Visages indexés</div>
        <div class="stat-value">{total_faces:,}</div>
    </div>
    """)

    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("LFW", f"{lfw_count:,}")
    with c2:
        st.metric("Custom", f"{custom_count:,}")

    st.divider()

    device_label = "GPU CUDA" if torch.cuda.is_available() else "CPU"

    st.markdown(f"**🧠 Moteur**\n\nFaceNet / VGGFace2\n\n**💻 Hardware**\n\n{device_label}")

    st.divider()

    if st.button("🔄 Synchroniser les visages", width="stretch"):
        with st.spinner("Synchronisation..."):
            added = sync_custom_faces()

        if added > 0:
            st.success(f"{added} nouveau(x) visage(s) ajouté(s).")
            st.rerun()
        else:
            st.info("La base est déjà synchronisée.")


# ============================================================
# ONGLETS
# ============================================================

tab_search, tab_database, tab_about = st.tabs(
    ["🔍  Analyser", "📚  Explorer la base", "🧠  À propos"]
)


# ============================================================
# ONGLET RECHERCHE
# ============================================================

with tab_search:

    html("""
    <div class="section-title">Analyse faciale</div>
    <div class="section-subtitle">
        Importe une photo ou utilise ta caméra pour lancer la recherche
        du visage le plus proche.
    </div>
    """)

    if mode == "🔍 Recherche":

        input_col, result_col = st.columns([0.95, 1.05], gap="large")

        with input_col:
            html("""
            <div class="card">
                <div class="card-title">📸 Ton visage</div>
                <div class="card-subtitle">
                    Choisis une image contenant un visage clairement visible.
                </div>
            </div>
            """)

            st.write("")

            source = st.radio(
                "Source",
                ["📁 Importer une photo", "📷 Utiliser la caméra"],
                horizontal=True
            )

            image = None

            if source == "📁 Importer une photo":
                uploaded = st.file_uploader(
                    "Dépose ton image ici",
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed"
                )
                if uploaded is not None:
                    image = Image.open(uploaded).convert("RGB")
            else:
                camera = st.camera_input("Prendre une photo")
                if camera is not None:
                    image = Image.open(camera).convert("RGB")

            if image is not None:
                st.image(image, width="stretch")
                st.caption("Image prête pour l'analyse.")

            analyze = st.button("✨ Trouver mon doppelganger", width="stretch", type="primary")

        with result_col:
            html("""
            <div class="card">
                <div class="card-title">🧬 Résultat IA</div>
                <div class="card-subtitle">
                    Les correspondances les plus proches apparaîtront ici.
                </div>
            </div>
            """)

            st.write("")

            if analyze:
                if image is None:
                    st.warning("Ajoute d'abord une image.")
                else:
                    with st.spinner("Analyse du visage en cours..."):
                        results = find_doppelganger_from_image(image, top_k=5)

                    if results is None:
                        st.error("😕 Aucun visage n'a été détecté dans cette image.")
                    else:
                        best = results[0]
                        is_unknown = best["distance"] > UNKNOWN_THRESHOLD

                        if is_unknown:
                            st.warning(
                                "⚠️ Aucun match suffisamment proche n'a été trouvé "
                                "avec le seuil actuel."
                            )
                        else:
                            level, style = proximity_level(best["distance"], UNKNOWN_THRESHOLD)

                            html(f"""
                            <div class="match-card">
                                <div class="match-label">✨ Correspondance principale</div>
                                <div class="match-name">{best["name"]}</div>
                                {badge_html(level, style)}
                                <br><br>
                                <div class="distance">
                                    Distance cosinus : <strong>{best["distance"]:.4f}</strong>
                                </div>
                                <div class="distance">
                                    Source : <strong>{best["source"]}</strong>
                                </div>
                            </div>
                            """)

                            st.write("")
                            st.image(best["image"], width="stretch")

                        st.markdown("### 🏆 Top correspondances")

                        for rank, result in enumerate(results, start=1):
                            level, style = proximity_level(result["distance"], UNKNOWN_THRESHOLD)

                            col_img, col_info = st.columns([0.30, 0.70])

                            with col_img:
                                st.image(result["image"], width="stretch")

                            with col_info:
                                st.markdown(f"**#{rank} — {result['name']}**")
                                st.caption(f"Source : {result['source']}")
                                st.markdown(badge_html(level, style), unsafe_allow_html=True)
                                st.write(f"Distance : `{result['distance']:.4f}`")

                            st.divider()

            else:
                html("""
                <div style="min-height:330px; display:flex; align-items:center;
                            justify-content:center; text-align:center; color:#64748b;">
                    <div>
                        <div style="font-size:58px; margin-bottom:15px;">🧬</div>
                        <div style="font-size:19px; font-weight:700; color:#cbd5e1;">
                            Prêt pour l'analyse
                        </div>
                        <div style="margin-top:8px; font-size:13px; max-width:380px;">
                            Le moteur IA comparera ton visage avec les visages
                            présents dans la base.
                        </div>
                    </div>
                </div>
                """)

    # ========================================================
    # MODE ENREGISTREMENT
    # ========================================================

    else:
        left, right = st.columns([0.9, 1.1], gap="large")

        with left:
            html("""
            <div class="card">
                <div class="card-title">📝 Ajouter un visage</div>
                <div class="card-subtitle">
                    Ajoute une nouvelle personne à ta base personnalisée.
                </div>
            </div>
            """)

            st.write("")

            name = st.text_input("Nom de la personne", placeholder="Exemple : Emmanuel")
            uploaded = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
            camera = st.camera_input("Ou prendre une photo")

            image = None
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
            elif camera is not None:
                image = Image.open(camera).convert("RGB")

            if image is not None:
                st.image(image, width="stretch")

            save_button = st.button("💾 Enregistrer le visage", width="stretch", type="primary")

        with right:
            if save_button:
                if not name.strip():
                    st.warning("Entre un nom avant d'enregistrer.")
                elif image is None:
                    st.warning("Ajoute une photo ou prends une photo.")
                else:
                    with st.spinner("Encodage du visage..."):
                        success, message = save_face(image, name)

                    if success:
                        st.success("✅ Visage ajouté avec succès !")
                        st.info("Le nouveau visage est maintenant disponible dans les recherches.")
                        st.rerun()
                    else:
                        st.error(message)
            else:
                html("""
                <div class="match-card">
                    <div class="match-label">BASE PERSONNALISÉE</div>
                    <div class="match-name">Construis ta propre galerie</div>
                    <p style="color:#94a3b8; line-height:1.7;">
                        Les visages enregistrés ici sont ajoutés à la base de
                        comparaison et peuvent ensuite être retrouvés lors
                        d'une recherche.
                    </p>
                    <br>
                    <div style="font-size:35px;">👤 ➜ 🧬 ➜ 🔎</div>
                </div>
                """)


# ============================================================
# EXPLORATION DE LA BASE
# ============================================================

with tab_database:
    html("""
    <div class="section-title">Explorer la base</div>
    <div class="section-subtitle">
        Visualise les visages indexés dans le moteur de reconnaissance.
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        source_filter = st.selectbox("Source", ["Toutes", "LFW", "Custom"])

    with c2:
        search_name = st.text_input("Rechercher un nom", placeholder="Exemple : John...")

    with c3:
        max_images = st.selectbox("Nombre de visages", [12, 24, 36, 60], index=1)

    filtered = []
    for i, name in enumerate(meta["names"]):
        source = meta["source"][i]

        if source_filter != "Toutes" and source != source_filter:
            continue
        if search_name and search_name.lower() not in name.lower():
            continue

        filtered.append(i)

    st.write(f"**{len(filtered):,} visage(s) trouvé(s)**")

    selected = filtered[:max_images]

    if not selected:
        st.info("Aucun visage ne correspond aux critères.")
    else:
        for start in range(0, len(selected), 4):
            row = selected[start:start + 4]
            cols = st.columns(len(row), gap="medium")

            for col, idx in zip(cols, row):
                with col:
                    st.image(meta["images"][idx], width="stretch")
                    st.markdown(f"**{meta['names'][idx]}**")
                    st.caption(meta["source"][idx])


# ============================================================
# À PROPOS
# ============================================================

with tab_about:
    html("""
    <div class="section-title">Comment fonctionne Doppelganger IA ?</div>
    <div class="section-subtitle">Une chaîne de traitement basée sur le deep learning.</div>
    """)

    steps = [
        ("📸", "01 · Image", "Une photo est importée ou capturée avec la caméra."),
        ("👁️", "02 · Détection", "MTCNN localise le visage présent dans l'image."),
        ("🧠", "03 · Embedding", "FaceNet transforme le visage en représentation numérique."),
        ("🔎", "04 · Matching", "Les embeddings sont comparés avec une distance cosinus."),
    ]

    cols = st.columns(4)
    for col, (icon, title, text) in zip(cols, steps):
        with col:
            html(f"""
            <div class="step-card">
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-text">{text}</div>
            </div>
            """)

    st.write("")
    st.divider()

    st.markdown("### 🧠 Architecture technique")

    st.code(
        """
Image
  │
  ▼
MTCNN
  │
  ▼
Détection du visage
  │
  ▼
InceptionResnetV1
(VGGFace2)
  │
  ▼
Embedding facial
  │
  ▼
Distance cosinus
  │
  ▼
Top-K correspondances
        """,
        language="text"
    )

    html("""
    <div class="card">
        <strong>⚠️ À propos du score</strong>
        <p style="color:#94a3b8; line-height:1.7;">
            La distance cosinus affichée représente une mesure de proximité entre
            les représentations faciales. Une distance plus faible indique une
            représentation plus proche. Le seuil configuré permet de déterminer
            si une correspondance est suffisamment proche pour être considérée
            comme reconnue.
        </p>
    </div>
    """)


# ============================================================
# FOOTER
# ============================================================

html("""
<br><br>
<div style="text-align:center; color:#475569; font-size:12px; padding:25px;">
    Doppelganger IA · Deep Learning Face Matching
    <br>
    FaceNet • MTCNN • VGGFace2 • Streamlit
</div>
""")