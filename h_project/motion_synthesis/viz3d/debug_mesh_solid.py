import numpy as np
import copy
from . import default

class DebugMeshSolid:

    def __init__(self, max_num_vertices=default.MESH_DEBUG_SOLID_NUM_VERTICES):

        # Setup
        self.vertices = np.ones((max_num_vertices, 4), dtype=np.float32)  # XYZ1
        self.normals = np.ndarray((max_num_vertices, 4), dtype=np.float32)  # XYZ1
        self.colors = np.ndarray((max_num_vertices, 4), dtype=np.float32)  # RGBA
        self.indices = np.ndarray((max_num_vertices,), dtype=np.uint32)  # UINT32
        self.num_vertices = 0
        self.num_indices = 0

        # OpenGL variables
        self.gl_vbo = None
        self.gl_ebo = None

        # Template meshes
        self.template_cube_vertices = np.array(default.CUBE_VERTICES_LIST, dtype=np.float32)
        self.template_cube_normals = np.array(default.CUBE_NORMALS_LIST, dtype=np.float32)
        self.template_cube_indices = np.array(default.CUBE_INDICES_LIST, dtype=np.uint32).flatten()
        self.template_cube_num_vertices = self.template_cube_vertices.shape[0]


    def add_cuboid(self, transform, width, height, depth, colorRGBA):

        """

        :param transform: Row major, Translation on the last column
        :param width:
        :param height:
        :param depth:
        :param colorRGBA:
        :return:
        """

        # ===== Vertices ====
        a = self.num_vertices
        b = a + self.template_cube_num_vertices
        cuboid_vertices = self.template_cube_vertices * np.array([width, height, depth, 1.0], dtype=np.float32)
        self.vertices[a:b, :] = np.matmul(cuboid_vertices, np.transpose(transform))
        self.num_vertices = copy.copy(b)

        # ===== Normals ====

        # ===== Colors =====
        self.colors[a:b, :] = colorRGBA

        # ===== Indices =====
        c = self.num_indices
        d = c + self.template_cube_indices.size
        self.indices[c:d] = self.template_cube_indices + c
        self.num_indices = copy.copy(d)

    def clear(self):
        self.num_vertices = 0
        self.num_indices = 0