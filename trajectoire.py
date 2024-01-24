import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PTS_to_SAVE = np.load('PTS_to_SAVE.npy')
print(len(PTS_to_SAVE))
PTS_to_SAVE = PTS_to_SAVE[120:150,:]
print(len(PTS_to_SAVE))
COINS = np.array([[20,0, 700],[140,200,700],[20,200,700],[140,0,700]])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(PTS_to_SAVE)):
    if PTS_to_SAVE[i,0]>COINS[0,0] and PTS_to_SAVE[i,0]<COINS[1,0] and PTS_to_SAVE[i,1]>COINS[0,1] and PTS_to_SAVE[i,1]<COINS[2,1]:
        ax.scatter(PTS_to_SAVE[i,0],PTS_to_SAVE[i,1],PTS_to_SAVE[i,2], c='g')
    else:
        ax.scatter(PTS_to_SAVE[i,0],PTS_to_SAVE[i,1],PTS_to_SAVE[i,2], c='b')
# ax.scatter(PTS_to_SAVE[:,0],PTS_to_SAVE[:,1],PTS_to_SAVE[:,2])
ax.scatter(COINS[:,0],COINS[:,1],COINS[:,2], c='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')



PTS_to_SAVE_LUC = np.load('PTS_to_SAVE_LUC.npy')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PTS_to_SAVE_LUC[:,0],PTS_to_SAVE_LUC[:,1],PTS_to_SAVE_LUC[:,2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

### create an animation by plotting one point at the time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
L = []
delta = 100
def animate(i):
    abc = ax.scatter(PTS_to_SAVE_LUC[i,0],PTS_to_SAVE_LUC[i,1],PTS_to_SAVE_LUC[i,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    L.append(abc)
    if i == delta:
        L[0].remove()
    if i > delta:
        ### remove points
        L[i-delta].remove()
    return ax

### réduire le nombre de points à afficher
ani = animation.FuncAnimation(fig, animate, frames=len(PTS_to_SAVE_LUC), interval=10e-10, blit=False, repeat=False)
ax.axes.set_xlim3d(left=-50, right=250)
ax.axes.set_ylim3d(bottom=-200, top=150)
ax.axes.set_zlim3d(bottom=400, top=1200)

print(len(PTS_to_SAVE_LUC))

plt.show()
