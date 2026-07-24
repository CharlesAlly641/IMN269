"""
Utilisation :
    python calibrage_stereo.py --left "calib_images/left_*.jpg" --right "calib_images/right_*.jpg"
"""

import argparse
import glob
import os
import sys

import numpy as np
import cv2

# Paramètres du damier
CHECKERBOARD = (9, 6)     # nombre de coins INTERNES (largeur, hauteur) -> 10x7 cases = 70 cases
SQUARE_SIZE_MM = 35.0     # taille réelle d'une case, mesurée après impression (mm)

# Coefficients de distorsion fournis dans l'énoncé : k1 = k2 = 0
FIX_DISTORTION_TO_ZERO = True


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrage stéréo avec OpenCV")
    parser.add_argument("--left", type=str, default="calib_images/left_*.jpg",
                         help="Motif glob pour les images de la caméra gauche")
    parser.add_argument("--right", type=str, default="calib_images/right_*.jpg",
                         help="Motif glob pour les images de la caméra droite")
    parser.add_argument("--out", type=str, default="calibration_stereo.npz",
                         help="Fichier de sortie pour sauvegarder les résultats")
    parser.add_argument("--show", action="store_true",
                         help="Afficher la détection des coins sur chaque image")
    return parser.parse_args()


def detecter_points(images_left, images_right, checkerboard, square_size, show=False):
    """Détecte les coins du damier sur toutes les paires d'images."""

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # Points 3D du damier dans son propre repère (Z = 0, plan)
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    img_size = None
    paires_valides = 0
    paires_rejetees = []

    for fname_l, fname_r in zip(images_left, images_right):
        img_l = cv2.imread(fname_l)
        img_r = cv2.imread(fname_r)

        if img_l is None or img_r is None:
            print(f"[AVERTISSEMENT] Impossible de lire : {fname_l} ou {fname_r}")
            paires_rejetees.append((fname_l, fname_r))
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray_l.shape[::-1]  # (width, height)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard, flags)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard, flags)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            paires_valides += 1

            if show:
                vis_l = img_l.copy()
                vis_r = img_r.copy()
                cv2.drawChessboardCorners(vis_l, checkerboard, corners_l, ret_l)
                cv2.drawChessboardCorners(vis_r, checkerboard, corners_r, ret_r)
                combo = np.hstack([vis_l, vis_r])
                combo = cv2.resize(combo, None, fx=0.5, fy=0.5)
                cv2.imshow("Detection damier (gauche | droite)", combo)
                cv2.waitKey(300)
        else:
            print(f"[INFO] Damier non détecté : {os.path.basename(fname_l)} / {os.path.basename(fname_r)}")
            paires_rejetees.append((fname_l, fname_r))

    if show:
        cv2.destroyAllWindows()

    print(f"\nPaires valides utilisées : {paires_valides}")
    if paires_rejetees:
        print(f"Paires rejetées ({len(paires_rejetees)}) :")
        for l, r in paires_rejetees:
            print(f"  - {os.path.basename(l)} / {os.path.basename(r)}")

    return objpoints, imgpoints_left, imgpoints_right, img_size


def calibrer_camera(objpoints, imgpoints, img_size, fix_distortion=True, label=""):
    """Calibrage intrinsèque d'une caméra."""
    flags = 0
    if fix_distortion:
        # k1 = k2 = 0 comme fourni dans l'énoncé ; on fixe aussi k3, p1, p2 par défaut
        flags = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, flags=flags)

    print(f"\n--- Calibrage caméra {label} ---")
    print(f"Erreur de reprojection (RMS) : {ret:.4f} pixels")
    print(f"Matrice intrinsèque :\n{mtx}")
    print(f"Coefficients de distorsion :\n{dist.ravel()}")

    return ret, mtx, dist, rvecs, tvecs


def calculer_erreur_reprojection(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Calcule l'erreur moyenne de reprojection (validation du calibrage).

    Utilise numpy plutôt que cv2.norm pour éviter les erreurs de type/forme
    qui peuvent survenir selon la version d'OpenCV (ex. CV_32FC1 vs CV_32FC2).
    """
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        # On force les deux tableaux au même format (N, 2) en float64
        pts_detectes = np.asarray(imgpoints[i], dtype=np.float64).reshape(-1, 2)
        pts_projetes = np.asarray(imgpoints_proj, dtype=np.float64).reshape(-1, 2)

        # Erreur euclidienne moyenne pour cette image
        diff = pts_detectes - pts_projetes
        error = np.sqrt(np.sum(diff ** 2, axis=1)).mean()

        total_error += error
        total_points += 1
    return total_error / total_points if total_points else float("nan")


def main():
    args = parse_args()

    images_left = sorted(glob.glob(args.left))
    images_right = sorted(glob.glob(args.right))

    if not images_left or not images_right:
        print("Aucune image trouvée. Vérifiez les chemins --left et --right.")
        sys.exit(1)

    if len(images_left) != len(images_right):
        print(f"[ERREUR] Nombre d'images différent : {len(images_left)} gauche vs {len(images_right)} droite.")
        sys.exit(1)

    print(f"{len(images_left)} paires d'images trouvées.")
    print(f"Damier : {CHECKERBOARD[0]}x{CHECKERBOARD[1]} coins internes "
          f"({(CHECKERBOARD[0]+1)*(CHECKERBOARD[1]+1)} cases), taille de case = {SQUARE_SIZE_MM} mm")

    # 1) Détection des coins sur toutes les paires
    objpoints, imgpoints_left, imgpoints_right, img_size = detecter_points(
        images_left, images_right, CHECKERBOARD, SQUARE_SIZE_MM, show=args.show)

    if len(objpoints) < 5:
        print("\n[ATTENTION] Moins de 5 paires valides détectées. "
              "Le calibrage risque d'être peu précis. Ajoutez plus d'images sous des angles variés.")

    # 2) Calibrage intrinsèque de chaque caméra
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = calibrer_camera(
        objpoints, imgpoints_left, img_size, FIX_DISTORTION_TO_ZERO, label="GAUCHE")
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = calibrer_camera(
        objpoints, imgpoints_right, img_size, FIX_DISTORTION_TO_ZERO, label="DROITE")

    # 3) Calibrage stéréo (paramètres extrinsèques R, T entre les deux caméras)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags_stereo = cv2.CALIB_FIX_INTRINSIC  # on garde les intrinsèques déjà calculés

    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, criteria=criteria_stereo, flags=flags_stereo)

    print("\n=== Résultat du calibrage stéréo ===")
    print(f"Erreur de reprojection stéréo (RMS) : {ret_stereo:.4f} pixels")
    print(f"R (rotation caméra droite par rapport à gauche) :\n{R}")
    print(f"T (translation caméra droite par rapport à gauche, mm) :\n{T.ravel()}")
    print(f"Distance entre les caméras (baseline) : {np.linalg.norm(T):.2f} mm")
    print(f"E (matrice essentielle) :\n{E}")
    print(f"F (matrice fondamentale) :\n{F}")

    # 4) Validation : erreur de reprojection moyenne par caméra
    err_l = calculer_erreur_reprojection(objpoints, imgpoints_left, rvecs_l, tvecs_l, mtx_l, dist_l)
    err_r = calculer_erreur_reprojection(objpoints, imgpoints_right, rvecs_r, tvecs_r, mtx_r, dist_r)
    print(f"\nErreur de reprojection moyenne - gauche : {err_l:.4f} pixels")
    print(f"Erreur de reprojection moyenne - droite : {err_r:.4f} pixels")
    if max(err_l, err_r) < 1.0:
        print("=> Calibrage jugé satisfaisant (erreur < 1 pixel).")
    else:
        print("=> Erreur élevée : envisager d'ajouter des paires d'images sous des angles plus variés, "
              "de vérifier la planéité du damier ou l'éclairage.")

    # 5) Sauvegarde des résultats
    np.savez(args.out,
              mtx_l=mtx_l, dist_l=dist_l,
              mtx_r=mtx_r, dist_r=dist_r,
              R=R, T=T, E=E, F=F,
              img_size=img_size,
              square_size_mm=SQUARE_SIZE_MM,
              checkerboard=CHECKERBOARD,
              reproj_error_left=err_l,
              reproj_error_right=err_r,
              reproj_error_stereo=ret_stereo)

    print(f"\nRésultats sauvegardés dans : {args.out}")


if __name__ == "__main__":
    main()