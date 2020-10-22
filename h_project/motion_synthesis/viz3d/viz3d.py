import numpy as np
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import pyrr
import time

# Local modules
from . import shaders
from . import default
from . import camera
from . import debug_mesh_solid

class Viz3D:

    def __init__(self):

        self.window_glfw = None
        self.shader_program = None
        self.main_camera = camera.Camera(params=default.MAIN_CAMERA_PARAMS)
        self.debug_mesh_solid = debug_mesh_solid.DebugMeshSolid()

        self.shader_var_id_position_in = None
        self.shader_var_id_normal_in = None
        self.shader_var_id_color_in = None

        self.gl_vertex_vbo = None
        self.gl_normal_vbo = None
        self.gl_color_vbo = None
        self.gl_indices_ebo = None

        self.initialised = False
        self.okay = True

    def initialise(self, window_size=(1000, 800), window_title='Vis3D'):

        # Initialize GLFW
        if not glfw.init():
            return False

        self.window_glfw = glfw.create_window(window_size[0], window_size[1], window_title, None, None)
        if not self.window_glfw:
            glfw.terminate()
            return False

        # Before you do anything, make you MUST make the drawing context active!
        glfw.make_context_current(self.window_glfw)

        self.shader_program = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(shaders.LIGHT_CASTER_VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(shaders.LIGHT_CASTER_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

        self.gl_vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.gl_vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.debug_mesh_solid.num_vertices * 16,  # 4 floats * 4 bytes per float
                     None, GL_STATIC_DRAW)

        self.gl_color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.gl_color_vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.debug_mesh_solid.num_vertices * 16,  # 4 floats * 4 bytes per float
                     None, GL_STATIC_DRAW)

        # Create EBO
        self.gl_indices_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gl_indices_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     self.debug_mesh_solid.num_indices * 4,
                     None, GL_STATIC_DRAW)

        # get the position from  shader
        self.shader_var_id_position_in = glGetAttribLocation(self.shader_program, 'position_in')
        self.shader_var_id_color_in = glGetAttribLocation(self.shader_program, 'color_in')

        self.transformLoc = glGetUniformLocation(self.shader_program, 'view_projection')

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE);

        self.initialised = True
        return True

    def _initialize_debug_mesh_solid(self):

        pass

    def render(self):

        # Render 3D solids
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader_program)

        glEnableVertexAttribArray(self.shader_var_id_position_in)
        glBindBuffer(GL_ARRAY_BUFFER, self.gl_vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.debug_mesh_solid.num_vertices * 16,  # 4 floats * 4 bytes per float
                     self.debug_mesh_solid.vertices, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(self.shader_var_id_position_in, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glEnableVertexAttribArray(self.shader_var_id_color_in)
        glBindBuffer(GL_ARRAY_BUFFER, self.gl_color_vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.debug_mesh_solid.num_vertices * 16,  # 4 floats * 4 bytes per float
                     self.debug_mesh_solid.colors, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(self.shader_var_id_color_in, 4, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gl_indices_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     self.debug_mesh_solid.num_indices * 4,
                     self.debug_mesh_solid.indices, GL_DYNAMIC_DRAW)

        glUniformMatrix4fv(self.transformLoc, 1, GL_FALSE, self.main_camera.view_projection_matrix)
        glDrawElements(GL_TRIANGLES, self.debug_mesh_solid.num_indices, GL_UNSIGNED_INT, None)

        # Do I need to do this?
        glDisableVertexAttribArray(self.shader_var_id_position_in)
        glDisableVertexAttribArray(self.shader_var_id_color_in)


        # Render 3D wireframe

        # Swap buffers and acquire inputs for next frame
        glfw.swap_buffers(self.window_glfw)
        glfw.poll_events()

        self.okay = not glfw.window_should_close(self.window_glfw)


        # Clear all immediate mode meshes
        self.debug_mesh_solid.clear()
        return self.okay

    def shutdown(self):

        glfw.terminate()



