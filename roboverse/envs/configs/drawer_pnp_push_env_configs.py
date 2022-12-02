import collections.abc
import copy

def update(d, order):
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), order)
        else:
            d[k] = [d[k][i] for i in order] + [d[k][i] for i in range(len(order), len(d[k]))]
    return d

default_config = {
    'camera_angle': {
        'yaw': 90,
        'pitch': -27,
    },
    'object_rgbs': {
        'large_object': [.93, .294, .169, 1.],
        'small_object': [.5, 1., 0., 1],
        'tray': [0.0, .502, .502, 1.],
        'drawer': {
            'frame': [.1, .25, .6, 1.],
            'bottom_frame': [.68, .85, .90, 1.],
            'bottom': [.5, .5, .5, 1.],
            'handle': [.59, .29, 0.0, 1.],
        },
    }
}

camera_angle_delta_sweep = [
    {
        'yaw': 0,
        'pitch': 0,
    },
    {
        'yaw': -10,
        'pitch': 8,
    },
    {
        'yaw': 10,
        'pitch': -8,
    },
]
object_rgbs_order_sweep = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

drawer_pnp_push_env_configs = []
for camera_angle_id, camera_angle_delta in enumerate(camera_angle_delta_sweep):
    for object_rgbs_order_id, object_rgbs_order in enumerate(object_rgbs_order_sweep):
        new_config = copy.deepcopy(default_config)
        new_config['camera_angle']['yaw'] += camera_angle_delta['yaw']
        new_config['camera_angle']['pitch'] += camera_angle_delta['pitch']
        new_config['camera_angle']['id'] = camera_angle_id
        update(new_config['object_rgbs'], object_rgbs_order)
        new_config['object_rgbs']['id'] = object_rgbs_order_id
        drawer_pnp_push_env_configs.append(new_config)