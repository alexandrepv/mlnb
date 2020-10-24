import numpy as np
import copy
from . import default

class MeshWorkshopSolid:

    """
    Only create one of these as this is designed to hold a large number of template solids and debug solids in memory
    """

    def __init__(self, max_num_vertices=default.MESH_DEBUG_SOLID_NUM_VERTICES):

        # Main Memory Allocation
        self.vertices = np.ones((max_num_vertices, 4), dtype=np.float32)  # XYZ1
        self.normals = np.ndarray((max_num_vertices, 3), dtype=np.float32)  # XYZ
        self.colors = np.ndarray((max_num_vertices, 4), dtype=np.float32)  # RGBA
        self.num_vertices = 0
        self.num_indices = 0

        # ===================================
        #           Template Solids
        # ===================================

        # Create cube template
        cube_num_vertices = default.TEMPLATE_CUBE_COMBO.shape[0] * 3
        self.template_cube_vertices = np.ndarray((cube_num_vertices, 4), dtype=np.float32)
        self.template_cube_normals = np.ndarray((cube_num_vertices, 3), dtype=np.float32)
        for i in range(default.TEMPLATE_CUBE_COMBO.shape[0]):  # Per triangle
            a = self.num_vertices
            b = a + 3
            triangle_indices = default.TEMPLATE_CUBE_COMBO[i, 0:3]
            triangle_normal = default.TEMPLATE_CUBE_NORMALS[default.TEMPLATE_CUBE_COMBO[i, 3], :]
            self.template_cube_vertices[a:b, :] = default.TEMPLATE_CUBE_VERTICES[triangle_indices, :]
            self.template_cube_normals[a:b, :] = triangle_normal
            self.num_vertices += 3

    def add_cuboid(self, transform, width, height, depth, colorRGBA):

        """
        TODO: Change width, height and depth to be a 3D vector

        :param transform: Row major, Translation on the last column
        :param width:
        :param height:
        :param depth:
        :param colorRGBA:
        :return:
        """

        # For simplicity
        a = self.num_vertices
        b = a + self.template_cube_vertices.shape[0]

        # ===== Vertices ====
        cuboid_vertices = self.template_cube_vertices * np.array([width, height, depth, 1.0], dtype=np.float32)
        self.vertices[a:b, :] = np.matmul(cuboid_vertices, np.transpose(transform))
        self.num_vertices = copy.copy(b)

        # ===== Normals ====
        self.normals[a:b, :] = np.matmul(self.template_cube_normals, np.transpose(transform[0:3, 0:3]))

        # ===== Colors =====
        self.colors[a:b, :] = colorRGBA