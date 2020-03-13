import pybullet as p

from roboverse.bullet.misc import deg_to_quat

def set_body_state(body, pos, quat):
	p.resetBasePositionAndOrientation(body, pos, quat)
