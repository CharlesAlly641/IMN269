import cv2
import numpy as np


def detecteur_harris(img, max_points=500, min_distance=5):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(
        gray_image,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=min_distance,
        useHarrisDetector=True,
        k=0.04
    )
    if pts is None:
        return np.array([], dtype=np.float32)
    return pts.reshape(-1, 2)


def correlation_normalisee(img1, img2, pts1, pts2, W=7, seuil=0.75):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray_img1.shape
    correspondant_g = []
    correspondant_d = []

    for p1 in pts1:
        x1, y1 = int(p1[0]), int(p1[1])
        if y1 - W < 0 or y1 + W >= h or x1 - W < 0 or x1 + W >= w:
            continue

        f1 = np.float32(gray_img1[y1 - W: y1 + W + 1, x1 - W: x1 + W + 1])
        f1c = f1 - np.mean(f1)
        norm_f1 = np.linalg.norm(f1c)
        if norm_f1 == 0:
            continue

        meilleur_score = -1
        meilleur_p2 = None

        for p2 in pts2:
            x2, y2 = int(p2[0]), int(p2[1])
            if y2 - W < 0 or y2 + W >= h or x2 - W < 0 or x2 + W >= w:
                continue

            f2 = np.float32(gray_img2[y2 - W: y2 + W + 1, x2 - W: x2 + W + 1])
            f2c = f2 - np.mean(f2)
            norm_f2 = np.linalg.norm(f2c)
            if norm_f2 == 0:
                continue

            score_cn = np.sum(f1c * f2c) / (norm_f1 * norm_f2)

            if score_cn > meilleur_score:
                meilleur_score = score_cn
                meilleur_p2 = p2

        if meilleur_score > seuil:
            correspondant_g.append(p1)
            correspondant_d.append(meilleur_p2)

    return np.array(correspondant_g), np.array(correspondant_d)


def matrice_fondamentale(pts_g, pts_d):
    N = pts_g.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x_g, y_g = pts_g[i][0], pts_g[i][1]
        x_d, y_d = pts_d[i][0], pts_d[i][1]
        A[i] = [
            x_d * x_g, x_d * y_g, x_d,
            y_d * x_g, y_d * y_g, y_d,
            x_g, y_g, 1,
        ]

    V = np.linalg.svd(A).Vh
    F = V[-1].reshape(3, 3)

    U, D, Vt = np.linalg.svd(F)
    D[2] = 0
    return np.dot(U, np.dot(np.diag(D), Vt))


def ransac(pcs_g, pcs_d, N_iterations=2000, t_seuil=0.5):
    num_pairs = pcs_g.shape[0]
    if num_pairs < 8:
        raise ValueError("Il faut au moins 8 paires initiales.")

    meilleur_S_k_taille = -1
    meilleur_S_k = None
    meilleure_F_initiale = None

    ones = np.ones((num_pairs, 1))
    pts_g_h = np.hstack((pcs_g, ones))
    pts_d_h = np.hstack((pcs_d, ones))

    for k in range(N_iterations):
        indices = np.random.choice(num_pairs, 8, replace=False)
        F_k = matrice_fondamentale(pcs_g[indices], pcs_d[indices])

        num = np.sum(np.dot(pts_d_h, F_k) * pts_g_h, axis=1)

        F_pg = np.dot(pts_g_h, F_k.T)
        pd_F = np.dot(pts_d_h, F_k)
        denom = (
            F_pg[:, 0] ** 2 + F_pg[:, 1] ** 2
            + pd_F[:, 0] ** 2 + pd_F[:, 1] ** 2
        )

        d_i = (num ** 2) / (denom + 1e-12)

        condition = d_i < t_seuil
        S_k_taille = np.sum(condition)

        if S_k_taille > meilleur_S_k_taille:
            meilleur_S_k_taille = S_k_taille
            meilleur_S_k = condition
            meilleure_F_initiale = F_k

    regulier_g = pcs_g[meilleur_S_k]
    regulier_d = pcs_d[meilleur_S_k]
    abberant_g = pcs_g[~meilleur_S_k]
    abberant_d = pcs_d[~meilleur_S_k]

    F_optimisee = matrice_fondamentale(regulier_g, regulier_d)

    return F_optimisee, (regulier_g, regulier_d), (abberant_g, abberant_d)


def dessiner_correspondances(img1, img2, pts1, pts2, couleur, epaisseur=1):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)

    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    for (xg, yg), (xd, yd) in zip(pts1, pts2):
        p1 = (int(xg), int(yg))
        p2 = (int(xd) + w1, int(yd))
        cv2.circle(canvas, p1, 3, couleur, -1)
        cv2.circle(canvas, p2, 3, couleur, -1)
        cv2.line(canvas, p1, p2, couleur, epaisseur, lineType=cv2.LINE_AA)

    return canvas


def erreur_epipolaire(F, pts_g, pts_d):
    ones = np.ones((len(pts_g), 1))
    pg_h = np.hstack([pts_g, ones])
    pd_h = np.hstack([pts_d, ones])
    erreurs = np.abs(np.sum((pd_h @ F) * pg_h, axis=1))
    return erreurs


if __name__ == "__main__":
    img_g = cv2.imread("visage/charles/left_01.jpg")
    img_d = cv2.imread("visage/charles/right_01.jpg")

    roi_g = (766, 142, 544, 544)
    roi_d = (603, 144, 571, 571)

    xg, yg, wg, hg = roi_g
    xd, yd, wd, hd = roi_d

    # --- Ajustement 1 : resserrer la ROI pour réduire le poids de la ligne des cheveux ---
    # On retire environ 20% du haut de la boîte (zone racine des cheveux)
    recul_haut = 0.20
    yg2 = yg + int(hg * recul_haut)
    hg2 = hg - int(hg * recul_haut)
    yd2 = yd + int(hd * recul_haut)
    hd2 = hd - int(hd * recul_haut)

    img_g_visage = img_g[yg2:yg2 + hg2, xg:xg + wg]
    img_d_visage = img_d[yd2:yd2 + hd2, xd:xd + wd]

    # --- Ajustement 2 : minDistance plus grand pour espacer les points ---
    pts_g = detecteur_harris(img_g_visage, max_points=800, min_distance=12)
    pts_d = detecteur_harris(img_d_visage, max_points=800, min_distance=12)
    print(f"Points détectés - gauche : {len(pts_g)}, droite : {len(pts_d)}")

    # --- Ajustement 3 : seuil de corrélation et fenêtre plus stricts ---
    pcs_g, pcs_d = correlation_normalisee(
        img_g_visage, img_d_visage, pts_g, pts_d, W=7, seuil=0.75)
    print(f"Correspondances avant RANSAC : {len(pcs_g)}")

    if len(pcs_g) < 8:
        raise RuntimeError(
            f"Seulement {len(pcs_g)} correspondances trouvées (minimum 8 requis). "
            "Baissez --seuil ou augmentez max_points.")

    # --- Ajustement 4 : RANSAC plus strict ---
    F, (reguliers_g, reguliers_d), (abberants_g, abberants_d) = ransac(
        pcs_g, pcs_d, N_iterations=2000, t_seuil=0.5)

    print("\nMatrice fondamentale F :")
    print(F)
    print(f"MC régulières : {len(reguliers_g)}")
    print(f"MC aberrantes : {len(abberants_g)}")

    # --- Diagnostic quantitatif ---
    erreurs = erreur_epipolaire(F, reguliers_g, reguliers_d)
    print(f"\nErreur épipolaire moyenne : {erreurs.mean():.4f}")
    print(f"Erreur épipolaire max : {erreurs.max():.4f}")

    indices_tries = np.argsort(erreurs)[::-1]
    print("\nLes 5 points réguliers avec la plus grande erreur (candidats suspects) :")
    for i in indices_tries[:5]:
        print(f"  gauche {reguliers_g[i]} <-> droite {reguliers_d[i]}, erreur = {erreurs[i]:.4f}")

    # --- Visualisation ---
    img_regulieres = dessiner_correspondances(
        img_g_visage, img_d_visage, reguliers_g, reguliers_d, couleur=(0, 255, 0))
    img_aberrantes = dessiner_correspondances(
        img_g_visage, img_d_visage, abberants_g, abberants_d, couleur=(0, 0, 255))

    cv2.imwrite("correspondances_regulieres.png", img_regulieres)
    cv2.imwrite("correspondances_aberrantes.png", img_aberrantes)
    print("\nImages sauvegardées : correspondances_regulieres.png, correspondances_aberrantes.png")