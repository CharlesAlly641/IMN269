import numpy as np
import matplotlib.pyplot as plt
import cv2

# Récupération de l'image
image = cv2.imread("Sol.png")
hauteur, largeur, _ = image.shape

# Points des droites parallèles
p1 = np.array([45, 128, 1])
p2 = np.array([258, 49, 1])
p3 = np.array([211, 266, 1])
p4 = np.array([440, 141, 1])

# Recherche du premier point de fuite
d1 = np.cross(p1, p2)
d2 = np.cross(p3, p4)
f1 = np.cross(d1, d2)

# Recherche du deuxième point de fuite
d3 = np.cross(p1, p3)
d4 = np.cross(p2, p4)
f2 = np.cross(d3, d4)

# Calcul de la ligne d'horizon
d = np.cross(f1, f2)

# Normalisation de la ligne d'horizon
# En géométrie projective, d et lambda*d représentent la même droite.
# OpenCV applique la transformation via une division cartésienne.
# Sans cette normalisation, les valeurs de d sont trop grandes,
# ce qui écrase toutes les coordonnées vers (0,0) et produit une image de sortie noire.
# À l'inverse, des valeurs trop petites donneraient une image géante.
d = d / d[2]

# Matrice de transformation H2
H2 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [d[0], d[1], d[2]]
])

# Application de la matrice à toute l'image
image_rectifiee = cv2.warpPerspective(image, H2, (largeur, hauteur))

plt.figure(figsize=(10, 5))

plt.imshow(image)
plt.axis('off')
plt.show()
plt.imshow(image_rectifiee)
plt.axis('off')
plt.savefig("rectifiee.jpg")
plt.show()