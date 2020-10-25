import numpy as np
import copy
from . import default
from . import template_geometry_solid as templates

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

        # Template solids
        self.vertices_cube, self.normals_cube = templates.create_cube(0.5, 1, 0.5)
        self.vertices_cylinder_8, self.normals_cylinder_8 = templates.create_cylinder(num_sides=8)
        self.vertices_cylinder_16, self.normals_cylinder_16 = templates.create_cylinder(num_sides=16)
        self.vertices_cylinder_32, self.normals_cylinder_32 = templates.create_cylinder(num_sides=32)

        # Create sphere templates
        #self.template_sphere_vertices, \
        #self.template_cylinder8_normals = self._create_cylinder_template(num_sides=8)

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
        b = a + self.vertices_cube.shape[0]

        cuboid_vertices = self.vertices_cube * np.array([width, height, depth, 1.0], dtype=np.float32)
        self.vertices[a:b, :] = np.matmul(cuboid_vertices, np.transpose(transform))
        self.normals[a:b, :] = np.matmul(self.normals_cube, np.transpose(transform[0:3, 0:3]))
        self.colors[a:b, :] = colorRGBA
        self.num_vertices = copy.copy(b)

    def add_cylinder8(self, transform, height, radius, colorRGBA):

        # For simplicity
        a = self.num_vertices
        b = a + self.vertices_cylinder_8.shape[0]

        cylinder_vertices = self.vertices_cylinder_8 * np.array([radius, height, radius, 1.0], dtype=np.float32)
        self.vertices[a:b, :] = np.matmul(cylinder_vertices, np.transpose(transform))
        self.normals[a:b, :] = np.matmul(self.normals_cylinder_8, np.transpose(transform[0:3, 0:3]))
        self.colors[a:b, :] = colorRGBA
        self.num_vertices = copy.copy(b)

    def add_cylinder16(self, transform, height, radius, colorRGBA):

        # For simplicity
        a = self.num_vertices
        b = a + self.vertices_cylinder_16.shape[0]

        cylinder_vertices = self.vertices_cylinder_16 * np.array([radius, height, radius, 1.0], dtype=np.float32)
        self.vertices[a:b, :] = np.matmul(cylinder_vertices, np.transpose(transform))
        self.normals[a:b, :] = np.matmul(self.normals_cylinder_16, np.transpose(transform[0:3, 0:3]))
        self.colors[a:b, :] = colorRGBA
        self.num_vertices = copy.copy(b)

    def add_cylinder32(self, transform, height, radius, colorRGBA):

        # For simplicity
        a = self.num_vertices
        b = a + self.vertices_cylinder_32.shape[0]

        cylinder_vertices = self.vertices_cylinder_32 * np.array([radius, height, radius, 1.0], dtype=np.float32)
        self.vertices[a:b, :] = np.matmul(cylinder_vertices, np.transpose(transform))
        self.normals[a:b, :] = np.matmul(self.normals_cylinder_32, np.transpose(transform[0:3, 0:3]))
        self.colors[a:b, :] = colorRGBA
        self.num_vertices = copy.copy(b)