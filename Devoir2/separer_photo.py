import glob
import os
import cv2

def separer_images(dossier_entree, dossier_sortie):
    os.makedirs(dossier_sortie, exist_ok=True)

    fichiers = sorted(glob.glob(os.path.join(dossier_entree, "*.jpg")))

    for i, fpath in enumerate(fichiers, start=1):
        img = cv2.imread(fpath)

        h, w = img.shape[:2]
        milieu = w // 2

        img_gauche = img[:, :milieu]
        img_droite = img[:, milieu:]

        nom_gauche = os.path.join(dossier_sortie, f"left_{i:02d}.jpg")
        nom_droite = os.path.join(dossier_sortie, f"right_{i:02d}.jpg")

        cv2.imwrite(nom_gauche, img_gauche)
        cv2.imwrite(nom_droite, img_droite)


if __name__ == "__main__":
    dossier_entree = "photos_damier/experimentation_damier_udes"
    dossier_sortie = "calib_images/calib_images_udes"
    separer_images(dossier_entree, dossier_sortie)