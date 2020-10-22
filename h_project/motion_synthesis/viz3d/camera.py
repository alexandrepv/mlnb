import pyrr
import numpy as np
from . import default

class Camera:

    def __init__(self, params):

        self.view_matrix = None
        self.view_projection_matrix = None
        self.mvp = None

        if params[default.CAMERA_KEY_TYPE] == default.CAMERA_KEY_PERSPECTIVE:
            self.projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(fovy=params[default.CAMERA_KEY_FOV_VERTICAL],
                                                                                        aspect=params[default.CAMERA_KEY_ASPECT],
                                                                                        near=params[default.CAMERA_KEY_NEAR],
                                                                                        far=params[default.CAMERA_KEY_FAR])
        elif params[default.CAMERA_KEY_TYPE] == default.CAMERA_KEY_ORTHOGRAPHIC:
            pass
        else:
            raise Exception(f"[ERROR] Camera type '{params[default.CAMERA_KEY_TYPE]}' not supported")

        self.update(camera_pos=np.array([4, 3, 3]),
                    look_at_target=np.array([0, 0, 0]))


    def update(self, camera_pos, look_at_target, up=np.array([0, 1, 0], dtype=np.float32)):

        # It seems like the look at function already inverts the matrix
        self.view_matrix = pyrr.matrix44.create_look_at(eye=camera_pos, target=look_at_target, up=up)
        self.view_projection_matrix = self.view_matrix @ self.projection_matrix

