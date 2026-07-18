import cv2
import numpy as np

# Trouve tous les points d'intérêt (PI) dans chacune des images
def detecteur_harris(img, max_points=500):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Identifie les N coins les plus marqués dans une image en niveaux de gris à l'aide du détecteur de Harris.
    pts = cv2.goodFeaturesToTrack(
        gray_image,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=5,
        useHarrisDetector=True,
        k=0.04
    )
    if pts is None:
        return np.array([], dtype=np.float32)
    # Paire les coordonnées 2 par 2
    return pts.reshape(-1, 2)


def correlation_normalisee(img1, img2, pts1, pts2, W=5, seuil=0.7):
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray_img1.shape
    correspondant_g = []
    correspondant_d = []

    for p1 in pts1:
        x1, y1 = int(p1[0]), int(p1[1])
        # Éviter les bords de l'image pour la fenêtre
        if y1 - W < 0 or y1 + W >= h or x1 - W < 0 or x1 + W >= w:
            continue

        # Centrage de la fenêtre de corrélation sur le point (x1, y1)
        f1 = np.float32(gray_img1[y1 - W : y1 + W + 1, x1 - W : x1 + W + 1])
        # Élimination des écarts de luminosité globale entre les deux images
        f1c = f1 - np.mean(f1)
        norm_f1 = np.linalg.norm(f1c)
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
            f2 = np.float32(gray_img2[y2 - W : y2 + W + 1, x2 - W : x2 + W + 1])
            # Élimination des écarts de luminosité globale entre les deux images
            f2c = f2 - np.mean(f2)
            norm_f2 = np.linalg.norm(f2c)
            if norm_f2 == 0:
                continue

            # Calcul de la corrélation normalisée, permet de ne pas dépendre de la moyenne des niveaux de gris.
            score_cn = np.sum(f1c * f2c) / (norm_f1 * norm_f2)

            if score_cn > meilleur_score:
                # Enregistrement du point et du score si c'est le meilleur qu'on a rencontré
                meilleur_score = score_cn
                meilleur_p2 = p2

        # Validation du meilleur score par le seuil
        if meilleur_score > seuil:
            # On enregistre le couple de correspondant pour le point p1 actuel
            correspondant_g.append(p1)
            correspondant_d.append(meilleur_p2)

    return np.array(correspondant_g), np.array(correspondant_d)


def matrice_fondamentale(pts_g, pts_d):
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
    V = np.linalg.svd(A).Vh
    # La plus petite valeur singulière est stockée à la dernière position
    F = V[-1].reshape(3, 3)

    # Application de la contrainte de rang 2
    U, D, Vt = np.linalg.svd(F)
    D[2] = 0
    return np.dot(U, np.dot(np.diag(D), Vt))


def ransac(pcs_g, pcs_d, N_iterations=1000, t_seuil=1.0):
    # Vérification du nombre de points (minimum 8 pour estimer F)
    num_pairs = pcs_g.shape[0]
    if num_pairs < 8:
        raise ValueError("Il faut au moins 8 paires initiales.")

    meilleur_S_k_taille = -1
    meilleur_S_k = None
    meilleure_F_initiale = None

    # Coordonnées homogènes pour le calcul de la distance de sampson
    ones = np.ones((num_pairs, 1))
    pts_g_h = np.hstack((pcs_g, ones))
    pts_d_h = np.hstack((pcs_d, ones))

    for k in range(N_iterations):
        # Choisir au hasard 8 paires et calculer F_k
        indices = np.random.choice(num_pairs, 8, replace=False)
        F_k = matrice_fondamentale(pcs_g[indices], pcs_d[indices])

        # Calculer la distance de sampson
        # Numérateur : (p_d^T * F * p_g)
        num = np.sum(np.dot(pts_d_h, F_k) * pts_g_h, axis=1)

        # Dénominateur : ||F_k * p_g||^2 + ||F_k^T * p_d||^2 (uniquement composantes x et y en 2D)
        F_pg = np.dot(pts_g_h, F_k.T)
        pd_F = np.dot(pts_d_h, F_k)
        denom = (
            F_pg[:, 0] ** 2
            + F_pg[:, 1] ** 2
            + pd_F[:, 0] ** 2
            + pd_F[:, 1] ** 2
        )

        d_i = (num**2) / (denom + 1e-12)


        # Construction de l'ensemble consensus S_k (retrait des valeurs abberantes)
        condition = d_i < t_seuil
        S_k_taille = np.sum(condition) # True = 1, False = 0

        # On garde le S_k contenant le plus d'éléments
        if S_k_taille > meilleur_S_k_taille:
            meilleur_S_k_taille = S_k_taille
            meilleur_S_k = condition
            meilleure_F_initiale = F_k

    # Séparation finale des données aberrantes et régulières
    regulier_g = pcs_g[meilleur_S_k]
    regulier_d = pcs_d[meilleur_S_k]
    abberant_g = pcs_g[~meilleur_S_k]
    abberant_d = pcs_d[~meilleur_S_k]

    # Réévaluer F avec toutes les mises en corespondance régulières
    F_optimisee = matrice_fondamentale(regulier_g, regulier_d)

    return F_optimisee, (regulier_g, regulier_d), (abberant_g, abberant_d)

# Applications sur les images
img_g = cv2.imread("image_gauche.jpg")
img_d = cv2.imread("image_droite.jpg")

pts_g = detecteur_harris(img_g, 500)
pts_d = detecteur_harris(img_d, 500)

pcs_g, pcs_d =  correlation_normalisee(img_g, img_d, pts_g, pts_d, 5, 0.7)

F, (reguliers_g, reguliers_d), (abberants_g, abberants_d) = ransac(pcs_g, pcs_d, 1000, 1.0)

print("Matrice fondamentale F :")
print(F)
print(f"MC régulières : {len(reguliers_g)}")
print(f"MC aberrantes : {len(aberrants_g)}")
