# realtime_cam.py
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_distances
import pickle

UNKNOWN_THRESHOLD = 0.5  # à aligner avec ton app

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeds = np.load("lfw_embeds.npy")
with open("lfw_meta.pkl", "rb") as f:
    meta = pickle.load(f)

if "source" not in meta:
    meta["source"] = ["LFW"] * len(meta["names"])


def recognise_frame(frame):
    """Retourne label + distance pour la frame BGR."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is None:
        return "No face", None

    with torch.no_grad():
        q_emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()

    dists = cosine_distances(q_emb, embeds).flatten()
    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])
    best_name = meta["names"][best_idx]

    if best_dist > UNKNOWN_THRESHOLD:
        return f"Unknown ({best_dist:.3f})", best_dist
    else:
        return f"{best_name} ({best_dist:.3f})", best_dist


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    label, dist = recognise_frame(frame)

    # rectangle approximatif + texte
    x1, y1, x2, y2 = int(w * 0.3), int(h * 0.2), int(w * 0.7), int(h * 0.8)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Doppelganger IA - Temps réel", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
