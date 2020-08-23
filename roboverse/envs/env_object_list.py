"""File for storing lists of objects used in registration.py"""
POSSIBLE_TRAIN_OBJECTS = [
    'conic_cup',
    'sack_vase',
    'fountain_vase',
    'shed',
    'circular_table',
    'hex_deep_bowl',
    'square_prism_bin',
    'narrow_tray',
    # New objects:
    'colunnade_top',
    'stalagcite_chunk',
    'bongo_drum_bowl',
    'pacifier_vase',
    'beehive_funnel',
    'crooked_lid_trash_can',
    'double_l_faucet',
    'toilet_bowl',
    'pepsi_bottle',
    'two_handled_vase',
    'tongue_chair',
    'oil_tanker',
    'thick_wood_chair',
    'modern_canoe',
    'pear_ringed_vase',
    'short_handle_cup',
    'curved_handle_cup',
    'bullet_vase',
    'glass_half_gallon',
    'flat_bottom_sack_vase',
    'trapezoidal_bin',
    'vintage_canoe',
]

POSSIBLE_TRAIN_SCALINGS = [
    0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3, 0.2,
    # New objects:
    0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.5, 0.2, 0.3, 0.2,
    0.2, 0.5, 0.2, 0.4, 0.3, 0.3, 0.2, 0.3, 0.3, 0.2,
    0.2, 0.4,
]

POSSIBLE_TRAIN_DICT = dict(zip(POSSIBLE_TRAIN_OBJECTS, POSSIBLE_TRAIN_SCALINGS))

POSSIBLE_TEST_OBJECTS = [
    'conic_bin',
    'jar',
    'gatorade',
    'box_sofa',
    'bathtub',
    # New objects:
    'ringed_cup_oversized_base',
    'square_rod_embellishment',
    't_cup',
    'aero_cylinder',
    'grill_trash_can',
]

POSSIBLE_TEST_SCALINGS = [
    0.2, 0.35, 0.35, 0.2, 0.2,
    # New objects:
    0.3, 0.3, 0.2, 0.2, 0.3,
]

POSSIBLE_TEST_DICT = dict(zip(POSSIBLE_TEST_OBJECTS, POSSIBLE_TEST_SCALINGS))