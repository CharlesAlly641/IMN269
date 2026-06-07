import numpy as np
import matplotlib.pyplot as plt
import cv2

# Récupération de l'image
image = cv2.imread("Sol.png")
hauteur = image.shape[0]
largeur = image.shape[1]

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

# Selon ce qu'on a vu dans le cours, d et lambda*d représentent la même droite.
# On normalise d par d[2] pour obtenir la forme d = (d1, d2, 1), ce qui permet
# une correspondance directe avec les points définis plus haute qui sont situés
# sur le plan z = 1. Sinon, on obtient une image de sortie complètement noire.
d = d / d[2]

# Matrice de transformation H2
H2 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [d[0], d[1], d[2]]
])

# Application de la matrice à toute l'image
# Prend en paramètre l'image d'entrée, la matrice de transformation et les dimensions de l'image de sortie
image_rectifiee = cv2.warpPerspective(image, H2, (largeur, hauteur))

plt.figure(figsize=(10, 5))

plt.imshow(image)
plt.axis('off')
plt.show()
plt.imshow(image_rectifiee)
plt.axis('off')
plt.savefig("rectifiee.jpg")
plt.show()