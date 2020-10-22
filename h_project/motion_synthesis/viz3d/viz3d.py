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
from . import mesh_workshop_solid

class Viz3D:

    def __init__(self):

        self.window_glfw = None
        self.shader_program = None
        self.main_camera = camera.Camera(params=default.MAIN_CAMERA_PARAMS)

        # Mesh Workshop Solid
        self.mws = mesh_workshop_solid.MeshWorkshopSolid()
        self.mws_gl_vbo_vertices = None
        self.mws_gl_vbo_normals = None
        self.mws_gl_vbo_colors = None
        self.mws_gl_ebo_indices = None
        self.mws_gl_program = None
        self.mws_gl_shader_world_position = None
        self.mws_gl_shader_world_normal = None
        self.mws_gl_shader_color = None
        self.mws_gl_uniform_view_projection = None

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

        self._initialize_mesh_workshop_solid()

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE);

        self.initialised = True
        return True

    def render(self):

        # Render 3D solids
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._render_mesh_workshop_solid()


        # Render 3D wireframe

        # Swap buffers and acquire inputs for next frame
        glfw.swap_buffers(self.window_glfw)
        glfw.poll_events()

        self.okay = not glfw.window_should_close(self.window_glfw)
        return self.okay


    def _initialize_mesh_workshop_solid(self):

        self.mws_gl_program = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(shaders.MWS_VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(shaders.MWS_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

        glUseProgram(self.mws_gl_program)

        self.mws_gl_shader_world_position = glGetAttribLocation(self.mws_gl_program, 'world_position')
        self.mws_gl_shader_color = glGetAttribLocation(self.mws_gl_program, 'color')
        self.mws_gl_uniform_view_projection = glGetUniformLocation(self.mws_gl_program, 'view_projection')

        self.mws_gl_vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mws_gl_vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.mws.num_vertices * 16, None, GL_STATIC_DRAW)

        self.mws_gl_vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mws_gl_vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, self.mws.num_vertices * 16, None, GL_STATIC_DRAW)

        self.mws_gl_ebo_indices = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.mws_gl_ebo_indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.mws.num_indices * 4, None, GL_STATIC_DRAW)

    def _render_mesh_workshop_solid(self):

        glUseProgram(self.mws_gl_program)

        self._upload_data_to_gpu(data=self.mws.vertices,
                                 gl_vbo=self.mws_gl_vbo_vertices,
                                 gl_shader_variable=self.mws_gl_shader_world_position,
                                 num_elements=self.mws.num_vertices)

        self._upload_data_to_gpu(data=self.mws.colors,
                                 gl_vbo=self.mws_gl_vbo_colors,
                                 gl_shader_variable=self.mws_gl_shader_color,
                                 num_elements=self.mws.num_vertices)

        self._upload_indices_to_gpu(indices=self.mws.indices,
                                    gl_ebo=self.mws_gl_ebo_indices,
                                    num_elements=self.mws.num_indices)

        glUniformMatrix4fv(self.mws_gl_uniform_view_projection, 1, GL_FALSE, self.main_camera.view_projection_matrix)
        glDrawElements(GL_TRIANGLES, self.mws.num_indices, GL_UNSIGNED_INT, None)

        self.mws.num_vertices = 0
        self.mws.num_indices = 0

    def _upload_data_to_gpu(self, data, gl_vbo, gl_shader_variable, num_elements):
        glEnableVertexAttribArray(gl_shader_variable)
        glBindBuffer(GL_ARRAY_BUFFER, gl_vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     num_elements * data.shape[1] * data.itemsize,
                     data,
                     GL_DYNAMIC_DRAW)
        glVertexAttribPointer(gl_shader_variable, data.shape[1], GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    def _upload_indices_to_gpu(self, indices, gl_ebo, num_elements):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_elements * 4, indices, GL_DYNAMIC_DRAW)

    def shutdown(self):

        glfw.terminate()



