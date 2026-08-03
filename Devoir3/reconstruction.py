import cv2
import numpy as np

def triangulation(pt_g, pt_d, R, T, M_int_g, M_int_d):
    """Méthode de reconstruction 3D par triangulation pour système convergents avec erreurs"""

    pg_h = np.array([pt_g[0], pt_g[1], 1.0])
    pd_h = np.array([pt_d[0], pt_d[1], 1.0])

    # Calculer les directions des rayons optiques (pg_h à O_g et pg_d à O_d)
    pg_dir = np.dot(np.linalg.inv(M_int_g), pg_h)
    pd_dir = np.dot(np.linalg.inv(M_int_d), pd_h)

    # Calculer segment c (p_g x R^t p_d)
    Rt_pd = np.dot(R.T, pd_dir)
    n = np.cross(pg_dir, Rt_pd)

    # Exprimer O_d dans le repère de gauche
    Od = np.dot(-R.T, T)

    # Résolution du système linéaire
    A = np.column_stack([pg_dir, -Rt_pd, n])
    b, a, c = np.linalg.solve(A, Od)

    Pg = b * pg_dir
    Ps = Pg + (c / 2) * n

    return Ps


if __name__ == "__main__":
    # Paramètres intrinsèques et extrinsèques estimés avec le calibrage
    Mg = np.array([[619.77, 0.00, 945.49], [0.00, 618.66, 521.45], [0.00, 0.00, 1.00]])
    Md = np.array([[614.61, 0.00, 955.12], [0.00, 615.15, 525.02], [0.00, 0.00, 1.00]])
    R = np.array([[9.98e-01, 7.86e-04, -6.30e-02], [7.04e-04, 1.00e+00, 2.36e-02], [6.30e-02, -2.36e-02, 9.98e-01]])
    T = np.array([-61.12, 0.53, -3.43])

    # Les 11 correspondances validées avec Ransac
    correspondances_validees = [
        ((878, 420), (761, 436)),
        ((1171, 354), (1017, 376)),
        ((916, 572), (803, 593)),
        ((908, 495), (780, 511)),
        ((947, 161), (798, 187)),
        ((906, 193), (765, 215)),
        ((1067, 369), (922, 391)),
        ((1011, 435), (896, 454)),
        ((945, 211), (808, 201)),
        ((988, 499), (844, 517)),
        ((974, 189), (825, 211))
    ]

    # Matrice de projection pour la caméra gauche considéré comme à l'origine du système
    P1 = np.dot(Mg, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Utilisation de l'image de gauche
    image = cv2.imread("left_01.jpg")
    image_reprojetee = image.copy()

    nuage_de_points = []

    for i, (pt_g, pt_d) in enumerate(correspondances_validees, 1):
        p_3d = triangulation(pt_g, pt_d, R, T, Mg, Md)

        # Extraction de la couleur du pixel dans l'image
        x, y = int(pt_g[0]), int(pt_g[1])
        b, g, r = image[y, x]

        nuage_de_points.append({
            "id": i,
            "3D": p_3d,
            "couleur_rgb": (r, g, b)
        })

        print(f"Point {i:2d} : X={p_3d[0]:.1f}, Y={p_3d[1]:.1f}, Z={p_3d[2]:.1f} mm, RGB=({r},{g},{b})")

        # Passage du point 3D en coordonnées homogènes 4D
        p_3d_homog = np.array([p_3d[0], p_3d[1], p_3d[2], 1.0])

        # Reprojection du point 3D sur l'image
        p_proj = np.dot(P1, p_3d_homog)

        # Normalisation des points pour passer au repère pixel (2D)
        u = int(p_proj[0] / p_proj[2])
        v = int(p_proj[1] / p_proj[2])

        # Erreur de reprojection : distance en pixels entre le point détecté original et le point 3D reprojeté
        erreur_reproj = np.sqrt((pt_g[0] - u) ** 2 + (pt_g[1] - v) ** 2)
        print(f"           Erreur de reprojection : {erreur_reproj:.2f} pixels")

        # Affichage sur l'image avec la couleur RGB appropriée
        cv2.circle(image_reprojetee, (u, v), radius=6, color=(int(b), int(g), int(r)), thickness=-1)
        cv2.putText(image_reprojetee, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite("visage_reprojete.jpg", image_reprojetee)