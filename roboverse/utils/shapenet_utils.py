import os
import json
import math
import random
import roboverse.bullet as bullet
from roboverse.bullet.misc import load_obj
import os.path as osp
import importlib.util

PATH = '/media/avi/data/Work/github/jannerm/bullet-manipulation/roboverse/envs/assets/ShapeNetCore'
SHAPENET_ASSET_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)), '../envs/assets/bullet-objects/ShapeNetCore')
# SHAPENET_ASSET_PATH = '/home/albert/dev/bullet-objects/ShapeNetCore'

def import_shapenet_metadata():
    metadata_spec = importlib.util.spec_from_file_location(
        "metadata", osp.join(SHAPENET_ASSET_PATH, "metadata.py"))
    shapenet_metadata = importlib.util.module_from_spec(metadata_spec)
    metadata_spec.loader.exec_module(shapenet_metadata)
    return shapenet_metadata.obj_path_map, shapenet_metadata.path_scaling_map


def load_shapenet_object(object_path, scaling, object_position, scale_local=0.5, quat=[1, -1, 0, 0]):
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    obj = load_obj(
        SHAPENET_ASSET_PATH + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(
            dir_name, object_name),
        SHAPENET_ASSET_PATH + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
            dir_name, object_name),
        object_position,
        quat, # this rotates objects 90 degrees. Originally: [0, 0, 1, 0]
        scale=scale_local*scaling[
            '{0}/{1}'.format(dir_name, object_name)])
    return obj


def load_random_objects(file_path, number):
    # if number > 5:
    # print("Don't load more than 4 objects!")
    # return None

    objects = []
    chosen_objects = []
    print(file_path)
    for root, dirs, files in os.walk(file_path + '/ShapeNetCore.v2'):
        for d in dirs:
            for modelroot, modeldirs, modelfiles in os.walk(
                    os.path.join(root, d)):
                for md in modeldirs:
                    objects.append(os.path.join(modelroot, md))
                break
        break
    try:
        chosen_objects = random.sample(range(len(objects)), number)
    except ValueError:
        print('Sample size exceeded population size')

    with open('{0}/scaling.json'.format(file_path), 'r') as fp:
        scaling = json.load(fp)

    def valid_positioning(pos, offset):
        if len(pos) <= 1:
            return True
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if math.sqrt(sum([(a - b) ** 2 for a, b in
                                  zip(pos[i], pos[j])])) < offset:
                    return False
        return True

    while True:
        positions = []
        for i in range(number):
            positions.append(
                (random.uniform(0.6, 0.85), random.uniform(-0.6, 0.3)))
        if valid_positioning(positions, 1 / number):
            break

    object_ids = []
    count = 0
    for i in chosen_objects:
        path = objects[i].split('/')
        dir_name = path[-2]
        object_name = path[-1]
        obj = load_obj(
            file_path + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dir_name,
                                                                      object_name),
            file_path + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
                dir_name, object_name),
            [positions[count][0], positions[count][1], 0], [0, 0, 1, 0],
            scale=random.uniform(0.5, 1) * scaling[
                '{0}/{1}'.format(dir_name, object_name)])
        object_ids.append(obj)
        count += 1
    return object_ids


def get_shapenet_object_list():
    objects = []
    for root, dirs, files in os.walk(PATH + '/ShapeNetCore.v2'):
        for d in dirs:
            for modelroot, modeldirs, modelfiles in os.walk(
                    os.path.join(root, d)):
                for md in modeldirs:
                    objects.append(os.path.join(modelroot, md))
                break
        break
    with open('{0}/scaling.json'.format(PATH), 'r') as fp:
        scaling = json.load(fp)
    return objects, scaling


def load_shapenet_objects(object_positions,
                          bullet,
                          object_ids=[0, 1, 25, 29, 30],
                          # object_ids=[33],

                          ):
    object_list, scaling = get_shapenet_object_list()
    object_dict = {}
    for num, i in enumerate(object_ids):
        path = object_list[i].split('/')
        dir_name = path[-2]
        object_name = path[-1]
        obj = load_obj(
            PATH + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dir_name,
                                                                  object_name),
            PATH + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
                dir_name, object_name),
            object_positions[num],
            [0, 0, 1, 0],
            scale=0.5 * scaling[
                '{0}/{1}'.format(dir_name, object_name)])
        object_dict[object_name] = obj
        for _ in range(200):
            bullet.step()

    assert  len(object_dict.keys()) == len(object_ids)
    return object_dict

def load_single_object(id, pos, quat=[0, 0, 1, 0], scale=1.0):
    file_path = SHAPENET_PATH
    objects = []

    for root, dirs, files in os.walk(file_path + '/ShapeNetCore.v2'):
        for d in dirs:
            for modelroot, modeldirs, modelfiles in os.walk(
                    os.path.join(root, d)):
                for md in modeldirs:
                    if md == id:
                        objects.append(os.path.join(modelroot, md))
                break
        break

    with open('{0}/scaling.json'.format(file_path), 'r') as fp:
        scaling = json.load(fp)

    object_ids = []
    count = 0
    for i in range(len(objects)):
        path = objects[i].split('/')
        dir_name = path[-2]
        object_name = path[-1]
        obj = load_obj(
            file_path + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dir_name,
                                                                       object_name),
            file_path + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
                dir_name, object_name),
            pos, quat,
            scale=scale * scaling[
                '{0}/{1}'.format(dir_name, object_name)])
        object_ids.append(obj)
        count += 1
    return object_ids
