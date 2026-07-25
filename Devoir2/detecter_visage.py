"""
Détection automatique de la région du visage (ROI) dans une paire d'images
stéréo, à l'aide du détecteur Haar Cascade fourni avec OpenCV.

Utilisation :
    python detecter_visage.py --gauche visage/left_01.jpg --droite visage/right_01.jpg
"""

import argparse
import cv2
import numpy as np


def detecter_visage(img, marge=0.3):
    """Détecte le plus grand visage dans l'image et retourne une boîte
    englobante (x, y, w, h) avec une marge ajoutée autour.

    marge : fraction de la taille du visage à ajouter de chaque côté
            (0.3 = 30% de marge, pour inclure un peu de contexte autour
            du visage sans aller jusqu'au chandail/arrière-plan)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chemin du classifieur Haar fourni avec OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    visages = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(visages) == 0:
        return None

    # Si plusieurs visages détectés, on garde le plus grand (le plus proche/évident)
    x, y, w, h = max(visages, key=lambda v: v[2] * v[3])

    # Ajout d'une marge autour du visage détecté
    mx = int(w * marge)
    my = int(h * marge)

    h_img, w_img = img.shape[:2]
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w_img, x + w + mx)
    y1 = min(h_img, y + h + my)

    return (x0, y0, x1 - x0, y1 - y0)


def parse_args():
    parser = argparse.ArgumentParser(description="Détection de la région du visage")
    parser.add_argument("--gauche", type=str, required=True)
    parser.add_argument("--droite", type=str, required=True)
    parser.add_argument("--marge", type=float, default=0.3, help="Marge autour du visage (fraction)")
    parser.add_argument("--out_prefix", type=str, default="visage_detecte")
    return parser.parse_args()


def main():
    args = parse_args()

    img_g = cv2.imread(args.gauche)
    img_d = cv2.imread(args.droite)

    if img_g is None or img_d is None:
        raise FileNotFoundError("Impossible de lire une des deux images.")

    roi_g = detecter_visage(img_g, args.marge)
    roi_d = detecter_visage(img_d, args.marge)

    if roi_g is None:
        print("[AVERTISSEMENT] Aucun visage détecté dans l'image gauche.")
    else:
        print(f"Visage détecté (gauche) : x={roi_g[0]}, y={roi_g[1]}, w={roi_g[2]}, h={roi_g[3]}")

    if roi_d is None:
        print("[AVERTISSEMENT] Aucun visage détecté dans l'image droite.")
    else:
        print(f"Visage détecté (droite) : x={roi_d[0]}, y={roi_d[1]}, w={roi_d[2]}, h={roi_d[3]}")

    # Visualisation : dessiner le rectangle détecté sur chaque image
    vis_g = img_g.copy()
    vis_d = img_d.copy()

    if roi_g:
        x, y, w, h = roi_g
        cv2.rectangle(vis_g, (x, y), (x + w, y + h), (0, 255, 0), 3)
    if roi_d:
        x, y, w, h = roi_d
        cv2.rectangle(vis_d, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imwrite(f"{args.out_prefix}_gauche.png", vis_g)
    cv2.imwrite(f"{args.out_prefix}_droite.png", vis_d)
    print(f"\nVérifiez visuellement : {args.out_prefix}_gauche.png / {args.out_prefix}_droite.png")

    if roi_g and roi_d:
        print(f"\n--- À utiliser dans votre script principal ---")
        print(f"roi_g = {roi_g}  # (x, y, w, h)")
        print(f"roi_d = {roi_d}  # (x, y, w, h)")


if __name__ == "__main__":
    main()