import cv2

def detecter_visage(img, marge=0.3):
    """Détecte le plus grand visage dans l'image et retourne une boîte
    englobante (x, y, w, h) avec une marge ajoutée autour.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    visages = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    x, y, w, h = visages[0]

    # Ajout d'une marge autour du visage détecté
    mx = int(w * marge)
    my = int(h * marge)

    h_img, w_img = img.shape[:2]
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(w_img, x + w + mx)
    y1 = min(h_img, y + h + my)

    return x0, y0, x1 - x0, y1 - y0