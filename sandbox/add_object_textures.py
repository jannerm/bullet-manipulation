import time
import numpy as np
import pdb
import os

import roboverse.bullet as bullet
import roboverse.devices as devices

from roboverse.utils.shapenet_utils import load_shapenet_object, \
    import_shapenet_metadata
obj_path_map, path_scaling_map = import_shapenet_metadata()

SHAPENET_PATH = "/home/albert/dev/bullet-objects/ShapeNetCore/ShapeNetCore.v2"

object_names = [
    "colunnade_top",
    "stalagcite_chunk",
    "bongo_drum_bowl",
    "pacifier_vase",
    "ringed_cup_oversized_base",
    "goblet",
    "beehive_funnel",
    "crooked_lid_trash_can",
    "double_l_faucet",
    "toilet_bowl",
    "square_rod_embellishment",
    "semi_golf_ball_bowl",
    "pepsi_bottle",
    "two_handled_vase",
    "tongue_chair",
    "oil_tanker",
    "elliptical_capsule",
    "rabbit_lamp",
    "thick_wood_chair",
    "modern_canoe",
    "pear_ringed_vase",
    "short_handle_cup",
    "curved_handle_cup",
    "box_crank",
    "bullet_vase",
    "glass_half_gallon",
    "flat_bottom_sack_vase",
    "teepee",
    "aero_cylinder",
    "keyhole",
    "trapezoidal_bin",
    "vintage_canoe",
    "bathtub",
    "flowery_half_donut",
    "grill_trash_can",
    "pitchfork_shelf",
    "t_cup",
    "cookie_circular_lidless_tin",
    "box_sofa",
    "baseball_cap",
    "two_layered_lampshade",
]

object_path_dict = dict(
    [(obj, path) for obj, path in obj_path_map.items() if obj in object_names])
scaling = dict(
    [(path, path_scaling_map[path]) for _, path in object_path_dict.items()])

def add_image_as_texture(obj_shapenet_path, image_url):
    image_url = image_url.strip()
    obj_path = os.path.join(SHAPENET_PATH, obj_shapenet_path)
    # mkdir images
    mkdir_command = "mkdir {}/images".format(obj_path)
    print(mkdir_command)
    os.system(mkdir_command)

    # wget the image url as texture.{jpg}/{png} into images/
    image_type = image_url.split(".")[-1]
    assert image_type in ['jpg', 'png']
    texture_image_name = "texture0.{}".format(image_type)
    wget_command = "wget -O {}/images/{} {}".format(obj_path, texture_image_name, image_url)
    print(wget_command)
    os.system(wget_command)

    # add map_Kd ../images/texture0.{jpg}/{png} to the end of models/model_normalized.mtl
    model_mtl_file = os.path.join(obj_path, "models", "model_normalized.mtl")
    with open(model_mtl_file, "a") as f_in:
        f_in.write("map_Kd ../images/{}\n".format(texture_image_name))

if __name__ == "__main__":
    bullet.connect()
    bullet.setup()

    obj_idx = 0
    
    while obj_idx < len(object_names):
        object_name = object_names[obj_idx]
        print("object_name", object_name)
 
        table = bullet.objects.table()
        load_shapenet_object(obj_path_map[object_name], scaling, np.array([.8, 0, -.2]), scale_local=0.5)
    
        for i in range(10):
            time.sleep(0.01)
            bullet.step()
    
        image_url = input('Enter image_url for object texture.' + 
            ' Else s if not changing image texture.')
    
        if image_url != "s":
            print("object_name", object_name)
            add_image_as_texture(obj_path_map[object_name], image_url)
        else:
           print("skipping adding image as texture.")
    
        # input to see if we stay on the same object or move on to the next.
        command = ""
        while command not in ['n', 'c']:
            command = input('n for next object, c for current.')
            if command == 'n':
                obj_idx += 1 
 
        bullet.reset()

