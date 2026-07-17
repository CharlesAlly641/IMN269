import cv2
import numpy as np

def detecter_harris_points(img, max_points=500):
    """
    Trouve tous les points d'intérêt (PI) dans chacune des images
    """
    # Source : https://www.geeksforgeeks.org/python/python-corner-detection-with-harris-corner-detection-method-using-opencv/
    # Gestion de la conversion de couleur
    operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)

    # Calcul de la réponse de Harris
    dst = cv2.cornerHarris(operatedImage, blockSize=2, ksize=3, k=0.04)

    # Appliquer le seuil initial
    seuil = 0.01 * dst.max()
    coordonnees = np.argwhere(dst > seuil)

    if len(coordonnees) == 0:
        return np.array([], dtype=np.float32)

    # Récupérer la valeur de la réponse de Harris pour chaque point détecté
    scores = dst[coordonnees[:, 0], coordonnees[:, 1]]

    # Trier les indices du plus grand score au plus petit
    indices_tries = np.argsort(scores)[::-1]
    coordonnees_triees = coordonnees[indices_tries]

    # Limiter aux N meilleurs points
    if len(coordonnees_triees) > max_points:
        coordonnees_triees = coordonnees_triees[:max_points]

    # Inversion (ligne, col) -> (x, y) pour OpenCV
    pts = np.float32(coordonnees_triees[:, ::-1])

    return pts


def correlation_normalisee_fenetre(img1, img2, pts1, pts2, W=5, seuil=0.7):
    """Calcule la Corrélation Normalisée (CN) entre tous les PIs

    Conserve le correspondant qui maximise la CN si max > seuil.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    pcs_initiales_g = []
    pcs_initiales_d = []

    for p1 in pts1:
        x1, y1 = int(p1[0]), int(p1[1])
        # Éviter les bords de l'image pour la fenêtre
        if y1 - W < 0 or y1 + W >= h or x1 - W < 0 or x1 + W >= w:
            continue

        # Centrage de la fenêtre de corrélation sur le point (x1, y1)
        f1 = np.float32(gray1[y1 - W : y1 + W + 1, x1 - W : x1 + W + 1])
        norm_f1 = np.linalg.norm(f1 - np.mean(f1))
        if norm_f1 == 0:
            continue

        meilleur_score = -1
        meilleur_p2 = None

        for p2 in pts2:
            x2, y2 = int(p2[0]), int(p2[1])
            # Éviter les bords de l'image pour la fenêtre
            if y2 - W < 0 or y2 + W >= h or x2 - W < 0 or x2 + W >= w:
                continue

            # Centrage de la fenêtre de corrélation sur le point (x2, y2)
            f2 = np.float32(gray2[y2 - W : y2 + W + 1, x2 - W : x2 + W + 1])
            norm_f2 = np.linalg.norm(f2 - np.mean(f2))
            if norm_f2 == 0:
                continue

            # Calcul de la corrélation normalisée
            score_cn = np.sum(f1 * f2) / (norm_f1 * norm_f2)

            if score_cn > meilleur_score:
                # Enregistrement du point et du score si c'est le meilleur qu'on a rencontré
                meilleur_score = score_cn
                meilleur_p2 = p2

        # Validation du meilleur score par le seuil
        if meilleur_score > seuil:
            # On enregistre nos correspondants
            pcs_initiales_g.append(p1)
            pcs_initiales_d.append(meilleur_p2)

    return np.array(pcs_initiales_g), np.array(pcs_initiales_d)


def estimer_f_8points(pts_g, pts_d):
    """Calcule F à l'aide de la SVD linéaire (Chapitre 5)"""
    N = pts_g.shape[0]
    A = np.zeros((N, 9))
    # Construction de la matrice A n x 9 des coefficients du système
    for i in range(N):
        x_g, y_g = pts_g[i][0], pts_g[i][1]
        x_d, y_d = pts_d[i][0], pts_d[i][1]
        A[i] = [
            x_d * x_g,
            x_d * y_g,
            x_d,
            y_d * x_g,
            y_d * y_g,
            y_d,
            x_g,
            y_g,
            1,
        ]

    # Décomposition de A en valeurs singulières
    _, _, V = np.linalg.svd(A)
    # La plus petite valeur singulière est stockée à la dernière position
    F = V[-1].reshape(3, 3)

    # Contrainte de rang 2 forcée
    U, D, Vt = np.linalg.svd(F)
    D[2] = 0
    return np.dot(U, np.dot(np.diag(D), Vt))


def ransac_chapitre6(pcs_g, pcs_d, N_iterations=1000, t_seuil=1.0):
    """Pages 34-37 : Algorithme 6.1 d'estimation robuste de F par RANSAC"""
    num_pairs = pcs_g.shape[0]
    if num_pairs < 8:
        raise ValueError("Il faut au moins 8 paires initiales.")

    meilleur_S_k_taille = -1
    meilleur_masque_inliers = None
    meilleure_F_initiale = None

    # Coordonnées homogènes pour le calcul des distances
    ones = np.ones((num_pairs, 1))
    pts_g_h = np.hstack((pcs_g, ones))
    pts_d_h = np.hstack((pcs_d, ones))

    for k in range(N_iterations):
        # Choisir au hasard 8 paires et calculer F_k
        indices = np.random.choice(num_pairs, 8, replace=False)
        F_k = estimer_f_8points(pcs_g[indices], pcs_d[indices])

        # Calculer la distance de sampson
        # Numérateur : (p_d^T * F * p_g)
        num = np.sum(np.dot(pts_d_h, F_k) * pts_g_h, axis=1)

        # Dénominateur : ||F_k * p_g||^2 + ||F_k^T * p_d||^2 (uniquement composantes x et y en 2D)
        F_pg = np.dot(F_k, pts_g_h)
        pd_F = np.dot(pts_d_h, F_k)
        denom = (
            F_pg[:, 0] ** 2
            + F_pg[:, 1] ** 2
            + pd_F[:, 0] ** 2
            + pd_F[:, 1] ** 2
        )

        if (num == 0) :
            d_i = 0
        else :
            d_i = num / denom


        # Construction de l'ensemble consensus S_k
        masque_inliers = d_i < t_seuil
        S_k_taille = np.sum(masque_inliers)

        # On garde le S_k contenant le plus d'éléments
        if S_k_taille > meilleur_S_k_taille:
            meilleur_S_k_taille = S_k_taille
            meilleur_masque_inliers = masque_inliers
            meilleure_F_initiale = F_k

    # Séparation finale des données aberrantes et régulières
    inliers_g = pcs_g[meilleur_masque_inliers]
    inliers_d = pcs_d[meilleur_masque_inliers]

    outliers_g = pcs_g[~meilleur_masque_inliers]
    outliers_d = pcs_d[~meilleur_masque_inliers]

    # Réévaluer F avec toutes les PCs régulières
    F_optimisee = estimer_f_8points(inliers_g, inliers_d)

    return F_optimisee, (inliers_g, inliers_d), (outliers_g, outliers_d)