import numpy as np
import pdb

import bullet
import pygame
from pygame.locals import QUIT, KEYDOWN, KEYUP
import time
#import devices

#space_mouse = devices.SpaceMouse()
#space_mouse.start_control()

bullet.connect()
bullet.setup()

## load meshes
sawyer = bullet.load_urdf('sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
table = bullet.load_urdf('table/table.urdf', [.75, -.2, -1], [0, 0, 0.707107, 0.707107], scale=1.0)
duck = bullet.load_urdf('duck_vhacd.urdf', [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)
#a = bullet.load_urdf('/home/jonathan/Desktop/model/urdf/model.urdf', [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)
#duck = bullet.load_obj('duck_vhacd.obj', 'duck.obj',  [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)
#duck = bullet.load_urdf('/home/jonathan/Desktop/object/urdf/test.urdf', [.75, -.2, 0], [0, 0, 1, 0], scale=0.8)
#lego = bullet.load_urdf('lego/lego.urdf', [.75, .2, 0], [0, 0, 1, 0], rgba=[1,0,0,1], scale=1.5)
#furniture = bullet.load_obj('/home/jonathan/Desktop/ae3257e7e0dca9a4fc8569054682bff9/output.obj', '/home/jonathan/Desktop/ae3257e7e0dca9a4fc8569054682bff9/model.obj' , [.75, .2, 0], [1, 0, 0, 1], scale=0.3f
fileName = '/home/jonathan/Desktop/plane/newsdf.sdf'
#k = bullet.load_obj(fileName, fileName, [.75, .2, 0], [1, 0, 0, 1], scale=0.k = bullet.load_sdf(k = bullet.load_sdf('/home/jonathan/Desktop/1eccbbbf1503c888f691355a196da5f/models/newsdf.sdf', [.75, -.3, 0], [1, 0, 0, 1], scale=0.3)
duck = bullet.load_sdf('/home/jonathan/Desktop/Projects/objects/5982f083a4a939607eee615e75bc3b77/newsdf.sdf', [.75, -.5, .2], [0, 0, 0, 1], scale=0.1)

end_effector = bullet.get_index_by_attribute(sawyer, 'link_name', 'right_l6')
pos = np.array([0.5, 0, 0])
theta =  np.array([0.7071,0.7071,0,0])
bullet.position_control(sawyer, end_effector, pos, theta)

#intializekeyboard
char_to_action = {
    'w': (np.array([0, 1, 0]), 'x'),
    'a': (np.array([-1, 0, 0]), 'x'),
    's': (np.array([0, -1, 0]), 'x'),
    'd': (np.array([1, 0, 0]), 'x'),
    'q': (np.array([1, -1, 0]), 'x'),
    'e': (np.array([-1, -1, 0]), 'x'),
    'z': (np.array([1, 1, 0]), 'x'),
    'c': (np.array([-1, 1, 0]), 'x'),
    'k': (np.array([0, 0, 1]), 'x'),
    'j': (np.array([0, 0, -1]), 'x'),
    'h': (np.array([1, 0, 0, 0]), 'theta'),
    'l': (np.array([-1, 0, 0, 0]), 'theta'),
    'c': (np.array([0, 0, 0, 0]), 'theta'), 
    'u': (0, 'gripper'),
    'i': (1, 'gripper'),
    'r': 'reset'
}

pressed_keys = {
    'w': False,
    'a': False,
    's': False,
    'd': False,
    'q': False,
    'e': False,
    'z': False,
    'c': False,
    'k': False,
    'j': False,
    'h': False,
    'l': False,
    'c': False,
    'p': False,
    'u': False,
    'i': False
    #'r': 'reset'
}

def startEnv():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    millis = int(round(time.time() * 1000))
    count = 1
    while True:
        if(int(round(time.time() * 1000)) - millis < 8):
            continue
        else: 
            millis = int(round(time.time() * 1000))
        dx = np.array([0, 0, 0])
        dtheta = np.array([0, 0, 0, 0])
        gripper = 0
        done = False
        
        for event in pygame.event.get():
            event_happened = True
            if event.type == QUIT:
                sys.exit()
            if event.type == KEYDOWN:
                char = chr(event.dict['key'])
                if char in pressed_keys.keys():
                    pressed_keys[char] = True
                #print(char)
            if event.type == KEYUP:
                char = chr(event.dict['key'])
                if char in pressed_keys.keys():
                    pressed_keys[char] = False
                #print(char)
                
        for i in pressed_keys.items():
            if i[1]:
                new_action = char_to_action.get(i[0], None)
                if new_action[1] == 'x':
                    dx += new_action[0]
                elif new_action[1] == 'theta':
                    dtheta += new_action[0]
                elif new_action[1] == 'gripper':
                    gripper = new_action[0]

        millis = int(round(time.time() * 1000))
        move(dx, dtheta, gripper)
        if done:
            return
            #wait(screen, 5)


def move(dx, dtheta, gripper):
    #delta = space_mouse.control
    global pos, theta
    pos += dx * 0.1
    theta += dtheta * 0.01
    #gripper = gripper == -1 else gripper
    #print(dx, pos, theta, gripper)
    #gripper = 1
    bullet.sawyer_ik(sawyer, end_effector, pos, theta, gripper)
    bullet.step()
    pos = bullet.get_link_state(sawyer, end_effector, 'pos')

startEnv()
#pygame.init()
#screen = pygame.display.set_mode((400, 300))
#while 1:
    #print(keyboard.is_pressed('q'))
    #print(k)
