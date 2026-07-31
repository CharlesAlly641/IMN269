import cv2
import numpy as np

def triangulation(pt_g, pt_d, R, T, M_int_g, M_int_d):
    # Matrices de projection avec la caméra gauche considéré comme l'origine
    P1 = np.dot(M_int_g, np.hstack((np.eye(3), np.zeros((3, 1)))))
    RT = np.hstack((R, T.reshape(3, 1)))
    P2 = np.dot(M_int_d, RT)

    pts_g = np.array([pt_g], dtype=np.float32)
    pts_d = np.array([pt_d], dtype=np.float32)

    # Triangulation pour trouver le point projeté
    pts_4d_homog = cv2.triangulatePoints(P1, P2, pts_g.T, pts_d.T)

    # Conversion des points 4d homogènes dans l'espace euclidien 3D
    pts_3d = pts_4d_homog[:3, :] / pts_4d_homog[3, :]
    return pts_3d.T[0]


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

        # Affichage sur l'image avec la couleur RGB appropriée
        cv2.circle(image_reprojetee, (u, v), radius=6, color=(int(b), int(g), int(r)), thickness=-1)
        cv2.putText(image_reprojetee, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite("visage_reprojete.jpg", image_reprojetee)