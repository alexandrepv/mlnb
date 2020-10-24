import numpy as np
import copy
from . import default

class MeshWorkshopWireframe:

    """
    Only create one of these as this is designed to hold a large number of template solids and debug solids in memory
    """

    def __init__(self, max_num_vertices=default.MESH_DEBUG_SOLID_NUM_VERTICES):

        # Main Memory Allocation
        self.vertices = np.ones((max_num_vertices, 4), dtype=np.float32)  # XYZ1
        self.colors = np.ndarray((max_num_vertices, 4), dtype=np.float32)  # RGBA
        self.num_vertices = 0

        # ===================================
        #           Template Wireframes
        # ===================================

        # Create cube template
        self.template_transform_vertices = np.array([[0, 0, 0, 1],
                                                     [1, 0, 0, 1],
                                                     [0, 0, 0, 1],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 0, 1],
                                                     [0, 0, 1, 1]], dtype=np.float32)
        self.template_transform_colors = np.array([[1, 0, 0, 1],
                                                   [1, 0, 0, 1],
                                                   [0, 1, 0, 1],
                                                   [0, 1, 0, 1],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 1, 1]], dtype=np.float32)


    def add_lines(self, transform, vertices, colorRGBA):

        """
        Vertices are organize as [V0_a, V0_b, V1_a, V1_b, ... Vn_a, Vn_b] in an N x 3 array
        :param transform: numpy ndarray (4, 4)
        :param vertices: numpy ndarray (N, 3)
        :return:
        """
        pass

    def add_axes(self, transform, axis_size=0.25):

        a = self.num_vertices
        b = a + 6
        self.vertices[a:b, :] = np.matmul(self.template_transform_vertices, np.transpose(transform))
        self.vertices[a:b, :] *= np.array([axis_size, axis_size, axis_size, 1], dtype=np.float32)
        self.colors[a:b, :] = self.template_transform_colors
        self.num_vertices = copy.copy(b)

    def add_xz_grid(self, transform, colorRGBA, cell_size=1, num_cells_per_axis=10):

        pass