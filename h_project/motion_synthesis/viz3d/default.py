import numpy as np


MESH_DEBUG_SOLID_NUM_VERTICES = int(1E6)

CAMERA_KEY_TYPE = 'type'
CAMERA_KEY_PERSPECTIVE = 'perspective'
CAMERA_KEY_ORTHOGRAPHIC = 'orthographic'
CAMERA_KEY_ASPECT = 'aspect'
CAMERA_KEY_FAR = 'far'
CAMERA_KEY_NEAR = 'near'
CAMERA_KEY_FOV_VERTICAL = 'fov_vertical'

MAIN_CAMERA_PARAMS = {CAMERA_KEY_TYPE: CAMERA_KEY_PERSPECTIVE,
                      CAMERA_KEY_ASPECT: 500/400,
                      CAMERA_KEY_FAR: 100,
                      CAMERA_KEY_NEAR: 0.1,
                      CAMERA_KEY_FOV_VERTICAL: 45.0}  # In degrees?


CUBE_VERTICES_LIST=[[-1, -1, -1, 1],
                    [-1, -1,  1, 1],
                    [1, -1, -1, 1],
                    [1, -1,  1, 1],
                    [1,  1, -1, 1],
                    [1,  1,  1, 1],
                    [-1,  1, -1, 1],
                    [-1,  1,  1, 1]]
CUBE_NORMALS_LIST=[[0, -1,  0],
                   [0,  0,  1],
                   [0,  0,  0],
                   [1,  0,  0],
                   [0,  1,  0],
                   [0,  0,  0],
                   [0,  0, -1],
                   [-1,  0,  0]]
CUBE_INDICES_LIST=[[0, 2, 3],
                   [0, 3, 1],
                   [4, 6, 7],
                   [4, 7, 5],
                   [3, 2, 4],
                   [3, 4, 5],
                   [7, 6, 0],
                   [7, 0, 1],
                   [6, 4, 2],
                   [6, 2, 0],
                   [1, 3, 5],
                   [1, 5, 7]]
# Meshes
"""
CUBE_VERTICES_LIST = [[0.5, -0.5, 0.5],
                      [0.5, 0.5, 0.5],
                      [0.5, 0.5, -0.5],
                      [0.5, -0.5, -0.5],
                      [-0.5, -0.5, 0.5],
                      [-0.5, 0.5, 0.5],
                      [-0.5, 0.5, -0.5],
                      [-0.5, -0.5, -0.5]]
CUBE_NORMALS_LIST = [[0.5, -0.5, 0.5],
                      [0.5, 0.5, 0.5],
                      [0.5, 0.5, -0.5],
                      [0.5, -0.5, -0.5],
                      [-0.5, -0.5, 0.5],
                      [-0.5, 0.5, 0.5],
                      [-0.5, 0.5, -0.5],
                      [-0.5, -0.5, -0.5]]
CUBE_INDICES_LIST = [[0, 3, 1],
                     [1, 3, 2],
                     [6, 5, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],
                     [0, 3, 1],]
#CUBE_INDICES_LIST = [[0,3,2,1],
#                     [4,5,6,7],
#                     [1,2,6,5],
#                     [0,4,7,3],
#                     [1,5,4,0],
#                     [2,3,7,6]]"""