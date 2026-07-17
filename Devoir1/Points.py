import cv2
import matplotlib
matplotlib.use('TkAgg')  # Backend interactif
import matplotlib.pyplot as plt

image = cv2.imread("Sol.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(image_rgb)
ax.set_title("Clique sur tes points pour obtenir leurs coordonnées")

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"np.array([{x}, {y}, 1])")
        ax.plot(x, y, 'ro', markersize=8)
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show(block=True)