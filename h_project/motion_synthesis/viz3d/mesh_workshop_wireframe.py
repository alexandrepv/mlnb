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
        self.template_axes_vertices = np.array([[0, 0, 0, 1],
                                                [1, 0, 0, 1],
                                                [0, 0, 0, 1],
                                                [0, 1, 0, 1],
                                                [0, 0, 0, 1],
                                                [0, 0, 1, 1]], dtype=np.float32)
        self.template_axes_colors = np.array([[1, 0, 0, 1],
                                              [1, 0, 0, 1],
                                              [0, 1, 0, 1],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 1],
                                              [0, 0, 1, 1]], dtype=np.float32)

        self.template_grid_vertices = np.ndarray((default.TEMPLATE_GRID_NUM_CELLS * 8 + 4, 4), dtype=np.float32)
        self.template_grid_colors = np.ndarray((default.TEMPLATE_GRID_NUM_CELLS * 8 + 4, 4), dtype=np.float32)

        # Create grid template
        max_dist = default.TEMPLATE_GRID_CELL_SIZE * default.TEMPLATE_GRID_NUM_CELLS
        self.template_grid_vertices[0, :] = [-max_dist, 0, 0, 1]
        self.template_grid_vertices[1, :] = [max_dist, 0, 0, 1]
        self.template_grid_vertices[2, :] = [0, 0, -max_dist, 1]
        self.template_grid_vertices[3, :] = [0, 0, max_dist, 1]
        self.template_grid_colors[0:2, :] = default.AXIS_X_COLOR_RGBA
        self.template_grid_colors[2:4, :] = default.AXIS_Z_COLOR_RGBA
        self.template_grid_colors[4:, :] = default.GRID_LINE_COLOR
        for i in range(default.TEMPLATE_GRID_NUM_CELLS):
            dist = max_dist * ((i+1) / default.TEMPLATE_GRID_NUM_CELLS)
            index = i * 8 + 4  # skip first 4 lines - they are the axes

            self.template_grid_vertices[index, :] = [-max_dist, 0, dist, 1]
            self.template_grid_vertices[index + 1, :] = [max_dist, 0, dist, 1]
            self.template_grid_vertices[index + 2, :] = [-max_dist, 0, -dist, 1]
            self.template_grid_vertices[index + 3, :] = [max_dist, 0, -dist, 1]

            self.template_grid_vertices[index + 4, :] = [dist, 0, -max_dist, 1]
            self.template_grid_vertices[index + 5, :] = [dist, 0, max_dist, 1]
            self.template_grid_vertices[index + 6, :] = [-dist, 0, -max_dist, 1]
            self.template_grid_vertices[index + 7, :] = [-dist, 0, max_dist, 1]


    def add_lines(self, transform, vertices, colorRGBA):

        """
        Vertices are organize as [V0_a, V0_b, V1_a, V1_b, ... Vn_a, Vn_b] in an N x 3 array
        :param transform: numpy ndarray (4, 4)
        :param vertices: numpy ndarray (N, 3)
        :return:
        """

        a = self.num_vertices
        b = a + vertices.shape[0]
        self.vertices[a:b, :] = np.matmul(vertices, np.transpose(transform))
        self.colors[a:b, :] = colorRGBA
        self.num_vertices = copy.copy(b)

    def add_axes(self, transform, axis_size=0.25):

        a = self.num_vertices
        b = a + 6
        self.vertices[a:b, :] = np.matmul(self.template_axes_vertices, np.transpose(transform))
        self.vertices[a:b, :] *= np.array([axis_size, axis_size, axis_size, 1], dtype=np.float32)
        self.colors[a:b, :] = self.template_axes_colors
        self.num_vertices = copy.copy(b)

    def add_xz_grid(self):
        a = self.num_vertices
        b = a + self.template_grid_vertices.shape[0]
        self.vertices[a:b, :] = self.template_grid_vertices
        self.colors[a:b, :] = self.template_grid_colors
        self.num_vertices = copy.copy(b)