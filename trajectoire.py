import numpy as np
import matplotlib.pyplot as plt

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

plt.show()
