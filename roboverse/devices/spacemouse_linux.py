"""Driver class for SpaceMouse controller on Linux.

SpaceMouse setup: http://spacenav.sourceforge.net/
Specifically, you only need to set up these two
 - https://github.com/FreeSpacenav/spacenavd (./configure, make, sudo make install, and then `sudo /etc/init.d/spacenavd start`)
 - https://github.com/FreeSpacenav/libspnav (`./configure`, `make`, `sudo make install`)

Python module: https://github.com/mastersign/pyspacenav
 - run `pip install -e .`

It's a bit confusing, but if the wire is away from you, then
 - forward: positive z
 - right: positive x
 - up: positive y
"""

import time
import threading
from collections import namedtuple
import numpy as np
import pdb
import spacenav
import atexit

from roboverse.devices.transform_utils import rotation_matrix


class SpaceMouseLinux:
    """A minimalistic driver class for SpaceMouse with spacenav library."""

    def __init__(self, blocking=True, max_action_norm=1., xyz='right,up,forward'):

        self._single_click_and_hold = False
        self._xyz_control = [0., 0., 0.]
        self._xyz_rot = [0., 0., 0.]
        self._reset_state = 0
        self._rotation_matrix = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        self._blocking = blocking
        self._max_action_norm = max_action_norm

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def get_controller_state(self):
        """Returns the current state of the 3d mouse.

        :returns: dictionary of pos, orn, grasp, and reset"""

        return dict(
            dpos=self.xyz_control,
            rotation=self.rotation,
            grasp=self.control_gripper,
            reset=self._reset_state
        )

    def run(self):
        """Listener method that keeps pulling new messages."""
        try:
            # open the connection
            print("Opening connection to SpaceNav driver ...")
            spacenav.open()
            print("... connection established.")
            # register the close function if no exception was raised
            atexit.register(spacenav.close)
        except spacenav.ConnectionError:
            # give some user advice if the connection failed
            print("No connection to the SpaceNav driver. Is spacenavd running?")
            sys.exit(-1)

        # reset exit condition
        stop = False

        # loop over space navigator events
        x0 = y0 = z0 = 0
        last_xyz = (0, 0, 0)
        while not stop:
            # wait for next event
            if self._blocking:
                event = spacenav.wait()
            else:
                event = spacenav.poll()
                if event is None:
                    time.sleep(0.01)
                    continue

            # if event signals the release of the first button
            if (
                    isinstance(event, spacenav.ButtonEvent)
                    and event.button == 0
                    and event.pressed == 0
            ):
                # print('Calibrating space mouse to zero to {}.'.format(last_xyz))
                # x0, y0, z0 = last_xyz
                self._reset_state = True

            if (
                    isinstance(event, spacenav.ButtonEvent)
                    and event.button == 1
            ):
                self._single_click_and_hold = event.pressed

            if isinstance(event, spacenav.MotionEvent):
                last_xyz = (event.x, event.y, event.z)

                self._xyz_control = [
                    event.x - x0,
                    event.y - y0,
                    event.z - z0,
                ]
                self._xyz_rot = [
                    event.rx,
                    event.ry,
                    event.rz,
                ]
                # self._update_rotation_matrix()  # TODO: test this

    @property
    def xyz_control(self):
        """Returns 6-DoF control."""
        xyz = np.array([
            self._xyz_control[2],
            self._xyz_control[0],
            self._xyz_control[1],
        ])
        return xyz * self._max_action_norm / 350.

    def _update_rotation_matrix(self):
        roll, pitch, yaw = np.array(self._xyz_rot) / 350.
        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1., 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1., 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.], point=None)[:3, :3]
        self._rotation_matrix = self._rotation_matrix.dot(drot1.dot(drot2.dot(drot3)))

    @property
    def rotation(self):
        """Returns 6-DoF control."""
        return self._rotation_matrix

    @property
    def control_gripper(self):
        """Maps internal states into gripper commands."""
        if self._single_click_and_hold:
            return 1.0
        else:
            return -1.0

    def get_action(self):
        dpos = self.xyz_control
        gripper = np.array([self.control_gripper])
        action = np.concatenate([dpos, gripper])
        return action


if __name__ == "__main__":
    space_mouse = SpaceMouseLinux(blocking=False)
    space_mouse.start_control()
    for i in range(10000):
        print(space_mouse.control, space_mouse.control_gripper)
        time.sleep(0.02)

    # basic example without SpaceMouseLinux class:
    # try:
    #     # open the connection
    #     print("Opening connection to SpaceNav driver ...")
    #     spacenav.open()
    #     print("... connection established.")
    #     # register the close function if no exception was raised
    #     atexit.register(spacenav.close)
    # except spacenav.ConnectionError:
    #     # give some user advice if the connection failed
    #     print("No connection to the SpaceNav driver. Is spacenavd running?")
    #     sys.exit(-1)
    #
    # stop = False
    # while not stop:
    #     event = spacenav.wait()
    #
    #     # if event signals the release of the first button
    #     if type(event) is spacenav.ButtonEvent \
    #             and event.button == 0 and event.pressed == 0:
    #         stop = True
    #
    #     print(event)
