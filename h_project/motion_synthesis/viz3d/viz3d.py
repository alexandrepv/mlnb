import numpy as np
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import pyrr
import time
import copy

# Local modules
from . import shaders
from . import default
from . import camera
from . import mesh_workshop_solid
from . import mesh_workshop_wireframe

class Viz3D:

    def __init__(self):

        self.window_glfw = None
        self.shader_program = None

        # Input variables
        self.first_mouse = True

        # Camera variables
        self.main_camera = None
        self.main_camera_speed = 1.0

        # Input variables
        self.mouse_pos_past = np.array([0, 0], dtype=np.float32)
        self.mouse_left_down = False
        self.mouse_right_down = False

        # Mesh Workshop Solid
        self.mws = mesh_workshop_solid.MeshWorkshopSolid()
        self.mws_gl_program = None
        self.mws_gl_vbo_vertices = None
        self.mws_gl_vbo_normals = None
        self.mws_gl_vbo_colors = None
        self.mws_gl_shader_world_position = None
        self.mws_gl_shader_world_normal = None
        self.mws_gl_shader_color = None
        self.mws_gl_uniform_view_projection = None
        self.mws_gl_uniform_light_position = None
        self.mws_gl_uniform_light_diffuse = None
        self.mws_gl_uniform_light_ambient = None

        # Mesh Workshop Wireframe
        self.mww = mesh_workshop_wireframe.MeshWorkshopWireframe()
        self.mww_gl_program = None
        self.mww_gl_vbo_vertices = None
        self.mww_gl_vbo_colors = None
        self.mww_gl_shader_world_position = None
        self.mww_gl_shader_color = None
        self.mww_gl_uniform_view_projection = None

        # Frame update variables
        self.time_past = time.time()
        self.elapsed_time = 0

        self.initialised = False
        self.okay = True

    def initialise(self, window_width=1280, window_height=720, window_title='Vis3D'):

        if not glfw.init():
            return False

        glfw.window_hint(glfw.SAMPLES, 4)

        self.window_glfw = glfw.create_window(window_width, window_height, window_title, None, None)
        if not self.window_glfw:
            glfw.terminate()
            return False

        self.main_camera = camera.Camera(view_width=window_width, view_height=window_height)

        # Before you do anything, make you MUST make the drawing context active!
        glfw.make_context_current(self.window_glfw)

        glfw.set_key_callback(self.window_glfw, self.key_callback)
        glfw.set_cursor_pos_callback(self.window_glfw, self.mouse_movement_callback)
        glfw.set_mouse_button_callback(self.window_glfw, self.mouse_button_callback)
        #glfw.set_cursor_enter_callback(self.window_glfw, self.mouse_enter_window_callback)

        self._init_mesh_workshop_solid()
        self._init_mesh_workshop_wireframe()

        # Depth test
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE);
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glClearColor(default.BACKGROUND_COLOR[0],
                     default.BACKGROUND_COLOR[1],
                     default.BACKGROUND_COLOR[2], 1)
        glClearDepth(1.0)

        #glEnable(GL_BLEND)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)

        self.initialised = True
        return True

    def render(self):

        time_present = time.time()
        self.elapsed_time = time_present - self.time_past
        self.time_past = time_present

        # Render 3D solids
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.main_camera.update(self.elapsed_time)

        self._render_mesh_workshop_wireframe()
        self._render_mesh_workshop_solid()

        # Swap buffers and acquire inputs for next frame
        glfw.swap_buffers(self.window_glfw)
        glfw.poll_events()

        self.okay = not glfw.window_should_close(self.window_glfw)
        return self.okay

    def _init_mesh_workshop_solid(self):

        self.mws_gl_program = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(shaders.MWS_VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(shaders.MWS_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

        glUseProgram(self.mws_gl_program)

        # Get VBO locations
        self.mws_gl_shader_world_position = glGetAttribLocation(self.mws_gl_program, 'world_position')
        self.mws_gl_shader_world_normal = glGetAttribLocation(self.mws_gl_program, 'world_normal')
        self.mws_gl_shader_color = glGetAttribLocation(self.mws_gl_program, 'color')

        # Get Uniform location
        self.mws_gl_uniform_view_projection = glGetUniformLocation(self.mws_gl_program, 'view_projection')
        self.mws_gl_uniform_light_position = glGetUniformLocation(self.mws_gl_program, 'light_position')
        self.mws_gl_uniform_light_diffuse = glGetUniformLocation(self.mws_gl_program, 'light_diffuse')
        # self.mws_gl_uniform_light_ambient = glGetUniformLocation(self.mws_gl_program, 'light_ambient')

        self.mws_gl_vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mws_gl_vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.mws.num_vertices * 16, None, GL_DYNAMIC_DRAW)

        self.mws_gl_vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mws_gl_vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, self.mws.num_vertices * 12, None, GL_DYNAMIC_DRAW)

        self.mws_gl_vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mws_gl_vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, self.mws.num_vertices * 16, None, GL_DYNAMIC_DRAW)

    def _init_mesh_workshop_wireframe(self):

        self.mww_gl_program = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(shaders.MWW_VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(shaders.MWW_FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

        glUseProgram(self.mww_gl_program)

        # Get VBO locations
        self.mww_gl_shader_world_position = glGetAttribLocation(self.mww_gl_program, 'world_position')
        self.mww_gl_shader_color = glGetAttribLocation(self.mww_gl_program, 'color')

        # Get Uniform location
        self.mww_gl_uniform_view_projection = glGetUniformLocation(self.mww_gl_program, 'view_projection')

        self.mww_gl_vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mww_gl_vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, self.mww.num_vertices * 16, None, GL_DYNAMIC_DRAW)

        self.mww_gl_vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mww_gl_vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, self.mww.num_vertices * 16, None, GL_DYNAMIC_DRAW)

    def _render_mesh_workshop_solid(self):

        if self.mws.num_vertices == 0:
            return

        glUseProgram(self.mws_gl_program)

        # VBOs
        self._upload_data_to_gpu(data=self.mws.vertices,
                                 gl_vbo=self.mws_gl_vbo_vertices,
                                 gl_shader_variable=self.mws_gl_shader_world_position,
                                 num_elements=self.mws.num_vertices)

        self._upload_data_to_gpu(data=self.mws.normals,
                                 gl_vbo=self.mws_gl_vbo_normals,
                                 gl_shader_variable=self.mws_gl_shader_world_normal,
                                 num_elements=self.mws.num_vertices)

        self._upload_data_to_gpu(data=self.mws.colors,
                                 gl_vbo=self.mws_gl_vbo_colors,
                                 gl_shader_variable=self.mws_gl_shader_color,
                                 num_elements=self.mws.num_vertices)

        # Uniforms
        glUniformMatrix4fv(self.mws_gl_uniform_view_projection, 1, GL_FALSE, self.main_camera.view_projection_matrix)

        # DEBUG Light
        light_pos = np.array([1.2, 1, 2], dtype=np.float32)
        light_diffuse = np.array([1, 1, 1], dtype=np.float32)
        light_ambient = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        glUniform3fv(self.mws_gl_uniform_light_position, 1, light_pos)
        glUniform3fv(self.mws_gl_uniform_light_diffuse, 1, light_diffuse)
        #glUniform3fv(self.mws_gl_uniform_light_ambient, 1, GL_FALSE, light_ambient)

        # Render Time!
        glDrawArrays(GL_TRIANGLES, 0, self.mws.num_vertices)

        # Clear all meshes
        self.mws.num_vertices = 0

    def _render_mesh_workshop_wireframe(self):

        if self.mww.num_vertices == 0:
            return

        glUseProgram(self.mww_gl_program)

        # VBOs
        self._upload_data_to_gpu(data=self.mww.vertices,
                                 gl_vbo=self.mww_gl_vbo_vertices,
                                 gl_shader_variable=self.mww_gl_shader_world_position,
                                 num_elements=self.mww.num_vertices)

        self._upload_data_to_gpu(data=self.mww.colors,
                                 gl_vbo=self.mww_gl_vbo_colors,
                                 gl_shader_variable=self.mww_gl_shader_color,
                                 num_elements=self.mww.num_vertices)

        # Uniforms
        glUniformMatrix4fv(self.mww_gl_uniform_view_projection, 1, GL_FALSE, self.main_camera.view_projection_matrix)

        # Render Time!
        glDrawArrays(GL_LINES, 0, self.mww.num_vertices)

        # Clear all meshes
        self.mww.num_vertices = 0

    def _init_mesh(self):

        #self.mws_gl_ebo_indices = glGenBuffers(1)
        #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.mws_gl_ebo_indices)
        #glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.mws.num_indices * 4, None, GL_STATIC_DRAW)
        pass

    def _render_mesh(self):

        #self._upload_data_to_gpu(data=self.mws.colors,
        #                         gl_vbo=self.mws_gl_vbo_colors,
        #                         gl_shader_variable=self.mws_gl_shader_color,
        #                         num_elements=self.mws.num_vertices)

        #self._upload_indices_to_gpu(indices=self.mws.indices,
        #                            gl_ebo=self.mws_gl_ebo_indices,
        #                            num_elements=self.mws.num_indices)

        # glDrawElements(GL_TRIANGLES, self.mws.num_indices, GL_UNSIGNED_INT, None)
        pass


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

    # =============================================================================
    #                               Input Callbacks
    # =============================================================================

    def key_callback(self, window, key, scancode, action, mods):

        if key == glfw.KEY_W:
            if action == glfw.PRESS:
                self.main_camera.move_forward = True
            elif action == glfw.RELEASE:
                self.main_camera.move_forward = False

        if key == glfw.KEY_S:
            if action == glfw.PRESS:
                self.main_camera.move_back = True
            elif action == glfw.RELEASE:
                self.main_camera.move_back = False

        if key == glfw.KEY_A:
            if action == glfw.PRESS:
                self.main_camera.move_left = True
            elif action == glfw.RELEASE:
                self.main_camera.move_left = False

        if key == glfw.KEY_D:
            if action == glfw.PRESS:
                self.main_camera.move_right = True
            elif action == glfw.RELEASE:
                self.main_camera.move_right = False

        if key == glfw.KEY_E:
            if action == glfw.PRESS:
                self.main_camera.move_up = True
            elif action == glfw.RELEASE:
                self.main_camera.move_up = False

        if key == glfw.KEY_Q:
            if action == glfw.PRESS:
                self.main_camera.move_down = True
            elif action == glfw.RELEASE:
                self.main_camera.move_down = False

        # Run!
        if key == glfw.KEY_LEFT_SHIFT:
            if action == glfw.PRESS:
                self.main_camera.movement_speed = default.CAMERA_MOVEMENT_SPEED * 4
            elif action == glfw.RELEASE:
                self.main_camera.movement_speed = default.CAMERA_MOVEMENT_SPEED

    def mouse_enter_window_callback(self, window, enter):

        #if enter == glfw.ENT:
        #    self.mouse_pos_past[:] = glfw.get_cursor_pos(window)
        pass

    def mouse_button_callback(self, window,  button, action, mods):

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_left_down = True
            elif action == glfw.RELEASE:
                self.mouse_left_down = False

        if button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.mouse_right_down = True
                self.first_mouse = True
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            elif action == glfw.RELEASE:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                self.mouse_right_down = False

    def mouse_movement_callback(self, window, xpos, ypos):

        if self.mouse_right_down:

            if self.first_mouse:
                self.mouse_pos_past[0] = xpos
                self.mouse_pos_past[1] = ypos
                self.first_mouse = False

            delta_x = xpos - self.mouse_pos_past[0]
            delta_y = self.mouse_pos_past[1] - ypos
            self.mouse_pos_past[0] = xpos
            self.mouse_pos_past[1] = ypos
            self.main_camera.process_mouse_movement(delta_x=delta_x, delta_y=delta_y)

