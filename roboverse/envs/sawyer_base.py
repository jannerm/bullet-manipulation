import roboverse.bullet as bullet
from roboverse.envs.robot_base import RobotBaseEnv


class SawyerBaseEnv(RobotBaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._id = 'SawyerBaseEnv'
        self._robot_name = 'sawyer'
        self._gripper_joint_name = ('right_gripper_l_finger_joint', 'right_gripper_r_finger_joint')
        self._gripper_range = range(20, 25)


        self._load_meshes()
        self._end_effector = self._end_effector = bullet.get_index_by_attribute(
            self._robot_id, 'link_name', 'gripper_site')
        self._setup_environment()


    def _load_meshes(self):
        self._robot_id = bullet.objects.sawyer()
        self._table = bullet.objects.table()
        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._robot_id,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
