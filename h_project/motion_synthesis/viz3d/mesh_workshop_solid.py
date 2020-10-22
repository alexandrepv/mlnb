import numpy as np
import copy
from . import default

from OpenGL.GL import *
import OpenGL.GL.shaders

VERTEX_SHADER = """
#version 330 core
in vec4 position_in;
in vec4 color_in;
//in vec3 normal_in;

out vec4 position_out;
//out vec3 normal_out;
out vec4 frag_color;

uniform mat4 view_projection;

void main()
{
    gl_Position = view_projection * position_in;
    frag_color = color_in;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec4 frag_color;
out vec4 color_out;

void main()
{
    color_out = frag_color;
} 
"""

class MeshWorkshopSolid:

    def __init__(self, max_num_vertices=default.MESH_DEBUG_SOLID_NUM_VERTICES):

        # Setup
        self.vertices = np.ones((max_num_vertices, 4), dtype=np.float32)  # XYZ1
        self.normals = np.ndarray((max_num_vertices, 4), dtype=np.float32)  # XYZ1
        self.colors = np.ndarray((max_num_vertices, 4), dtype=np.float32)  # RGBA
        self.indices = np.ndarray((max_num_vertices,), dtype=np.uint32)  # UINT32
        self.num_vertices = 0
        self.num_indices = 0

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
