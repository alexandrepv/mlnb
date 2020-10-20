import numpy as np
from . import default

class MeshDebugSolid:

    def __init__(self, max_num_vertices=default.MESH_DEBUG_SOLID_NUM_VERTICES):

        # Setup
        self.vertices = np.ndarray((max_num_vertices, 4), dtype=np.float32)
        self.num_vertices = 0

        # Create template solids
        self.template_mesh_cube = np.array(default.CUBE_VERTICES_LIST, dtype=np.float32)



    def add_cuboid(self, transform, width, height, depth):


        pass
