from roboverse.envs.env_object_list import (
    POSSIBLE_TRAIN_OBJECTS, POSSIBLE_TRAIN_SCALINGS,
    POSSIBLE_TEST_OBJECTS, POSSIBLE_TEST_SCALINGS)
import numpy as np

class RandObjEnv:
    """
    Generalization env. Randomly samples one of the following objects
    every time the env resets.
    """
    def __init__(self,
                 *args,
                 in_eval=False,
                 deterministic_idx=[],
                 train_scaling_list=[0.3]*10,
                 test_scaling_list=[0.3]*10,
                 possible_train_objects="default",
                 possible_test_objects="default",
                 num_objects=2,
                 **kwargs):
        self.in_eval = in_eval # True when doing evaluation
        self.deterministic_idx = deterministic_idx # Contain desired idx when doing replay
        self._num_objects = num_objects
        print("self num objects in init: ", self._num_objects)
        # so that we use novel test_objects.

        if possible_train_objects == "default":
            # Decided based on object_success_best_scaling.csv
            self.possible_train_objects = POSSIBLE_TRAIN_OBJECTS[:10]
            self.possible_train_scaling_local_list = POSSIBLE_TRAIN_SCALINGS[:10]
        else:
            assert isinstance(possible_train_objects, list)
            self.possible_train_objects = possible_train_objects
            self.possible_train_scaling_local_list = train_scaling_list

        if possible_test_objects == "default":
            self.possible_test_objects = POSSIBLE_TEST_OBJECTS[:10]
            self.possible_test_scaling_local_list = POSSIBLE_TEST_SCALINGS[:10]
        else:
            assert isinstance(possible_test_objects, list)
            self.possible_test_objects = possible_test_objects
            self.possible_test_scaling_local_list = test_scaling_list

        if self.in_eval:
            self.possible_objects = self.possible_test_objects
            self.possible_scaling_local_list = \
                self.possible_test_scaling_local_list
        else:
            self.possible_objects = self.possible_train_objects
            self.possible_scaling_local_list = \
                self.possible_train_scaling_local_list

        super().__init__(*args,
            object_names=self.possible_objects,
            scaling_local_list=self.possible_scaling_local_list,
            **kwargs)

    def reset(self):
        if len(self.deterministic_idx) > 0:
            self.object_names = []
            self.scaling_local_list = []
            for i in range(self._num_objects):
                chosen_obj_idx = self.deterministic_idx[i]
                self.object_names.append(self.possible_objects[chosen_obj_idx])
                self.scaling_local_list.append(self.possible_scaling_local_list[chosen_obj_idx])
        else:
            """Change implementation to multiple objects"""
            self.object_names = []
            self.scaling_local_list = []
            self.object_names_idx = []
            for _ in range(self._num_objects):
                chosen_obj_idx = np.random.randint(0, len(self.possible_objects))
                while chosen_obj_idx in self.object_names_idx:
                    chosen_obj_idx = np.random.randint(0, len(self.possible_objects))
                self.object_names_idx.append(chosen_obj_idx)
                self.object_names.append(self.possible_objects[chosen_obj_idx])
                self.scaling_local_list.append(self.possible_scaling_local_list[chosen_obj_idx])
            #print("self.object_names", self.object_names, self.scaling_local_list)
        return super().reset()
