from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2

# y, x = np.ogrid[-5:5:100j, -5:5:100j]
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
x, y = np.meshgrid(x, y)


# ax = fig.gca(projection='3d')


# Gradienten anhand von Funktionen zeigen
# Funktion1 - Parabel, eine Hauptkrümmung, entlang des "Tals", aber keinerlei Krümmung --> Kante
# z=x*x # - (y-y)
# Funktion2 - 2 Hauptkrümmungen, eine positiv und eine negativ - Änderung in zwei Richtungen --> Ecke
z=x*y
levels=np.arange(-20, 20, 5)
fig = plt.figure()
ax=fig.add_subplot(1,2,1)
ax.contourf(x, y, z, levels, cmap=cm.gray)
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_zlabel("Z")
ax=fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(x, y, z, cmap=cm.gray)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()