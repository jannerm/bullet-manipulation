import numpy as np
import pdb

import roboverse.bullet as bullet
import roboverse.bullet.objects as objects
import roboverse.devices as devices

bullet.connect()
bullet.setup()

## load meshes
bowl_sliding = objects.bowl_sliding()
spam = objects.spam()
table = objects.table()

pos = np.array([0.5, 0, 0])
theta = [0.7071,0.7071,0,0]

while True:
    bullet.step()
