import numpy as np
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import pyrr

# Local modules
from . import shaders

class Viz3D:

    def __init__(self):

        self.window_glfw = None

        self.shader_program = None

        self.initialised = False



    def initialise(self, window_size=(1280, 720), window_title='Vis3D'):

        # Initialize GLFW
        if not glfw.init():
            return False

        self.window_glfw = glfw.create_window(window_size[0], window_size[1], window_title, None, None)
        if not self.window_glfw:
            glfw.terminate()
            return False

        # Before you do anything, make you MUST make the drawing context active!
        glfw.make_context_current(self.window_glfw)

        # Initialize Shaders
        self.shader_program = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(shaders.VERTEX_SHADER, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(shaders.FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

        cube = [-0.5, -0.5, 0.5, 1.0, 0.0, 0.0,
                0.5, -0.5, 0.5, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                -0.5, 0.5, 0.5, 1.0, 1.0, 1.0,

                -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
                0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
                0.5, 0.5, -0.5, 0.0, 0.0, 1.0,
                -0.5, 0.5, -0.5, 1.0, 1.0, 1.0]

        # convert to 32bit float

        cube = np.array(cube, dtype=np.float32)

        indices = [0, 1, 2, 2, 3, 0,
                   4, 5, 6, 6, 7, 4,
                   4, 5, 1, 1, 0, 4,
                   6, 7, 3, 3, 2, 6,
                   5, 6, 2, 2, 1, 5,
                   7, 4, 0, 0, 3, 7]

        indices = np.array(indices, dtype=np.uint32)

        # Create Buffer object in gpu
        VBO = glGenBuffers(1)
        # Bind the buffer
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 192, cube, GL_STATIC_DRAW)

        # Create EBO
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 144, indices, GL_STATIC_DRAW)

        # get the position from  shader
        position = glGetAttribLocation(self.shader_program, 'position')
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)

        # get the color from  shader
        color = glGetAttribLocation(self.shader_program, 'color')
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        glUseProgram(self.shader_program)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Make the window's context current

        self.initialised = True

        return True

    def render(self):

        if not self.initialised:
            raise Exception('[ERROR] You need to initialise Viz3D first.')

        # Render 3D solids

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
        rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())

        transformLoc = glGetUniformLocation(self.shader_program, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

        # Render 3D wireframe

        # Swap buffers and acquire inputs for next frame
        glfw.swap_buffers(self.window_glfw)
        glfw.poll_events()

        return not glfw.window_should_close(self.window_glfw)

    def shutdown(self):

        glfw.terminate()



