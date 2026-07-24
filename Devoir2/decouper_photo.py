"""
Découpe des images stéréo combinées (gauche | droite côte à côte) en deux images séparées.

Vos photos ressemblent à celle jointe : une seule image contenant la vue gauche
et la vue droite collées horizontalement. Ce script les sépare en deux fichiers
prêts à être utilisés par calibrage_stereo.py.

Organisation :
    photos_combinees/*.jpg   (vos 8 photos telles quelles)
        -> calib_images/left_01.jpg, left_02.jpg, ...
        -> calib_images/right_01.jpg, right_02.jpg, ...

Utilisation :
    python decouper_images.py --input "photos_combinees/*.jpg" --out calib_images
"""

import argparse
import glob
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Découpe des images stéréo combinées")
    parser.add_argument("--input", type=str, default="photos_combinees/*.jpg",
                         help="Motif glob des photos combinées")
    parser.add_argument("--out", type=str, default="calib_images",
                         help="Dossier de sortie pour les images gauche/droite")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    fichiers = sorted(glob.glob(args.input))
    if not fichiers:
        print("Aucune image trouvée avec ce motif.")
        return

    for i, fpath in enumerate(fichiers, start=1):
        img = cv2.imread(fpath)
        if img is None:
            print(f"[AVERTISSEMENT] Impossible de lire {fpath}, ignorée.")
            continue

        h, w = img.shape[:2]
        milieu = w // 2

        img_gauche = img[:, :milieu]
        img_droite = img[:, milieu:]

        nom_gauche = os.path.join(args.out, f"left_{i:02d}.jpg")
        nom_droite = os.path.join(args.out, f"right_{i:02d}.jpg")

        cv2.imwrite(nom_gauche, img_gauche)
        cv2.imwrite(nom_droite, img_droite)

        print(f"{os.path.basename(fpath)} -> {nom_gauche} ({img_gauche.shape[1]}x{img_gauche.shape[0]}), "
              f"{nom_droite} ({img_droite.shape[1]}x{img_droite.shape[0]})")

    print(f"\nTerminé. Images séparées dans le dossier : {args.out}")


if __name__ == "__main__":
    main()