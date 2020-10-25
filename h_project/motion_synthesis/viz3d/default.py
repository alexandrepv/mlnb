import numpy as np

MESH_DEBUG_SOLID_NUM_VERTICES = int(1E6)

CAMERA_FAR_FIELD = 1000
CAMERA_NEAR_FIELD = 0.1
CAMERA_FOV_DEGREES = 45.0

CAMERA_POSITION = np.array([0, 0, 2], dtype=np.float32)
CAMERA_WORLD_UP = np.array([0, 1, 0], dtype=np.float32)
CAMERA_UP = np.array([0, 1, 0], dtype=np.float32)
CAMERA_FRONT = np.array([0, 0, -1], dtype=np.float32)
CAMERA_PITCH_RADIANS = 0
CAMERA_YAW_RADIANS = -np.pi/2  # TODO: This is a workaround for the camera 90degree turn at the begining. FIX THIS PROPERLY!
CAMERA_MOUSE_SENSITIVITY = 0.001
CAMERA_ZOOM = 45.0
CAMERA_MOVEMENT_SPEED = 2.5
CAMERA_MAX_PITCH_RADIANS = 1.55334  # ~89 degrees

# Environment variables
BACKGROUND_COLOR = np.array([60, 60, 60, 255], dtype=np.float32) / 255.0
GRID_LINE_COLOR = np.array([95, 95, 95, 255], dtype=np.float32) / 255.0
AXIS_X_COLOR_RGBA = np.array([255, 54, 83, 255], dtype=np.float32) / 255.0
AXIS_Y_COLOR_RGBA = np.array([136, 216, 13, 255], dtype=np.float32) / 255.0
AXIS_Z_COLOR_RGBA = np.array([44, 142, 254, 255], dtype=np.float32) / 255.0
STUDIO_LIGHTS = [{'direction': (-0.892, 0.3, 0.9),
                  'diffuse_color': (0.8, 0.8, 0.8),
                  'specular_color': (0.5, 0.5, 0.5)},
                 {'direction': (0.588, 0.46, 0.248),
                  'diffuse_color': (0.498, 0.5, 0.6),
                  'specular_color': (0.2, 0.2, 0.2)},
                 {'direction': (0.216, -0.392, -0.216),
                  'diffuse_color': (0.798, 0.838, 1.0),
                  'specular_color': (0.066, 0.0, 0.0)}]

TEMPLATE_GRID_CELL_SIZE = 1.0
TEMPLATE_GRID_NUM_CELLS = 25  # Per axis per quadrant. 25 means 2500 cells


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