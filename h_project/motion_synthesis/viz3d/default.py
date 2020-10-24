import numpy as np

MESH_DEBUG_SOLID_NUM_VERTICES = int(1E6)

CAMERA_FAR_FIELD = 0.1
CAMERA_NEAR_FIELD = 1000
CAMERA_FOV_DEGREES = 45.0

CAMERA_POSITION = np.array([0, 0, 2], dtype=np.float32)
CAMERA_WORLD_UP = np.array([0, 1, 0], dtype=np.float32)
CAMERA_UP = np.array([0, 1, 0], dtype=np.float32)
CAMERA_FRONT = np.array([0, 0, -1], dtype=np.float32)
CAMERA_PITCH_RADIANS = 0
CAMERA_YAW_RADIANS = 0
CAMERA_MOUSE_SENSITIVITY = 0.001
CAMERA_ZOOM = 45.0
CAMERA_MOVEMENT_SPEED = 2.5
CAMERA_MAX_PITCH_RADIANS = 1.55334  # ~89 degrees

TEMPLATE_CUBE_VERTICES = np.array([[0.5, -0.5, 0.5, 1],
                                   [0.5, 0.5, 0.5, 1],
                                   [0.5, 0.5, -0.5, 1],
                                   [0.5, -0.5, -0.5, 1],
                                   [-0.5, -0.5, 0.5, 1],
                                   [-0.5, 0.5, 0.5, 1],
                                   [-0.5, 0.5, -0.5, 1],
                                   [-0.5, -0.5, -0.5, 1]], dtype=np.float32)
TEMPLATE_CUBE_NORMALS = np.array([[1, 0,  0],
                                  [0,  1,  0],
                                  [0,  0,  1],
                                  [-1,  0,  0],
                                  [0,  -1,  0],
                                  [0,  0,  -1]], dtype=np.float32)
TEMPLATE_CUBE_COMBO = np.array([[0, 2, 1, 0], # Triangle indices and normal indices
                                [0, 3, 2, 0],
                                [1, 2, 6, 1],
                                [1, 6, 5, 1],
                                [4, 1, 5, 2],
                                [4, 0, 1, 2],
                                [5, 7, 4, 3],
                                [5, 6, 7, 3],
                                [4, 3, 0, 4],
                                [4, 7, 3, 4],
                                [3, 6, 2, 5],
                                [3, 7, 6, 5]], dtype=np.int32)
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