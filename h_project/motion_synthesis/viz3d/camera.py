import pyrr
from . import default

class Camera:

    def __init__(self, params):

        if params[default.CAMERA_KEY_TYPE] == default.CAMERA_KEY_PERSPECTIVE:
            self.matrix = pyrr.matrix44.create_perspective_projection_matrix(fovy=params[default.CAMERA_KEY_FOV_VERTICAL],
                                                                             aspect=params[default.CAMERA_KEY_ASPECT],
                                                                             near=params[default.CAMERA_KEY_NEAR],
                                                                             far=params[default.CAMERA_KEY_FAR])
        elif params[default.CAMERA_KEY_TYPE] == default.CAMERA_KEY_ORTHOGRAPHIC:
            pass
        else:
            raise Exception(f"[ERROR] Camera type '{params[default.CAMERA_KEY_TYPE]}' not supported")

