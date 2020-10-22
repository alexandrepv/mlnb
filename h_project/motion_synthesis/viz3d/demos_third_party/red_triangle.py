# From: https://gitlab.com/metamost/learning-opengl-with-python/blob/master/tutorial-3-matrices/main.py
import contextlib, ctypes, logging, sys
import numpy as np
from scipy import linalg
from OpenGL import GL as gl
import glfw

log = logging.getLogger(__name__)

@contextlib.contextmanager
def create_main_window(width, height):
    if not glfw.init():
        log.error('failed to initialize GLFW')
        sys.exit(1)
    try:
        log.debug('requiring modern OpenGL without any legacy features')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        log.debug('opening window')
        title = 'Tutorial 3: Matrices'
        window = glfw.create_window(width, height, title, None, None)
        if not window:
            log.error('failed to open GLFW window.')
            sys.exit(2)
        glfw.make_context_current(window)

        log.debug('set background to dark blue')
        gl.glClearColor(0, 0, 0.4, 0)

        yield window

    finally:
        log.debug('terminating window context')
        glfw.terminate()

@contextlib.contextmanager
def create_vertex_array_object():
    log.debug('creating and binding the vertex array (VAO)')
    vertex_array_id = gl.glGenVertexArrays(1)
    try:
        gl.glBindVertexArray(vertex_array_id)
        yield
    finally:
        log.debug('cleaning up vertex array')
        gl.glDeleteVertexArrays(1, [vertex_array_id])

@contextlib.contextmanager
def create_vertex_buffer():
    with create_vertex_array_object():
        # A triangle
        vertex_data = [-1, -1, 0,
                        1, -1, 0,
                        0,  1, 0]
        attr_id = 0  # No particular reason for 0,
                     # but must match layout location in the shader.

        log.debug('creating and binding the vertex buffer (VBO)')
        vertex_buffer = gl.glGenBuffers(1)
        try:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)

            array_type = (gl.GLfloat * len(vertex_data))
            sizeof_float = ctypes.sizeof(ctypes.c_float)
            gl.glBufferData(gl.GL_ARRAY_BUFFER,
                            len(vertex_data) * sizeof_float,
                            array_type(*vertex_data),
                            gl.GL_STATIC_DRAW)

            log.debug('setting the vertex attributes')
            gl.glVertexAttribPointer(
               attr_id,            # attribute 0.
               3,                  # components per vertex attribute
               gl.GL_FLOAT,        # type
               False,              # to be normalized?
               0,                  # stride
               None                # array buffer offset
            )

            # use currently bound VAO
            gl.glEnableVertexAttribArray(attr_id)

            yield
        finally:
            log.debug('cleaning up buffer')
            gl.glDisableVertexAttribArray(attr_id)
            gl.glDeleteBuffers(1, [vertex_buffer])

@contextlib.contextmanager
def load_shaders():
    shaders = {
    gl.GL_VERTEX_SHADER: '''\
        #version 330 core
        layout(location = 0) in vec3 vertexPosition_modelspace;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(vertexPosition_modelspace, 1);
        }
    ''',
    gl.GL_FRAGMENT_SHADER: '''\
        #version 330 core
        out vec3 color;
        void main() {
          color = vec3(1,0,0);
        }
    '''}
    log.debug('creating the shader program')
    program_id = gl.glCreateProgram()
    try:
        shader_ids = []
        for shader_type, shader_src in shaders.items():
            shader_id = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader_id, shader_src)

            log.debug(f'compiling the {shader_type} shader')
            gl.glCompileShader(shader_id)

            # check if compilation was successful
            result = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
            nlog = gl.glGetShaderiv(shader_id, gl.GL_INFO_LOG_LENGTH)
            if nlog:
                logmsg = gl.glGetShaderInfoLog(shader_id)
                log.error(logmsg)
                sys.exit(10)

            gl.glAttachShader(program_id, shader_id)
            shader_ids.append(shader_id)

        log.debug('linking shader program')
        gl.glLinkProgram(program_id)

        # check if linking was successful
        result = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
        nlog = gl.glGetProgramiv(program_id, gl.GL_INFO_LOG_LENGTH)
        if nlog:
            logmsg = gl.glGetProgramInfoLog(program_id)
            log.error(logmsg)
            sys.exit(11)

        log.debug('installing shader program into rendering state')
        gl.glUseProgram(program_id)
        yield program_id
    finally:
        log.debug('cleaning up shader program')
        for shader_id in shader_ids:
            gl.glDetachShader(program_id, shader_id)
            gl.glDeleteShader(shader_id)
        gl.glUseProgram(0)
        gl.glDeleteProgram(program_id)

def normalized(v):
    norm = linalg.norm(v)
    return v / norm if norm > 0 else v

def perspective(fov, aspect, near, far):
    n, f = near, far
    t = np.tan((fov * np.pi / 180) / 2) * near
    b = - t
    r = t * aspect
    l = b * aspect
    assert abs(r - l) > 0
    assert abs(t - b) > 0
    assert abs(n - f) > 0
    return np.array((
        ((2*n)/(r-l),           0,           0,  0),
        (          0, (2*n)/(t-b),           0,  0),
        ((r+l)/(r-l), (t+b)/(t-b), (f+n)/(n-f), -1),
        (          0,           0, 2*f*n/(n-f),  0)))

def look_at(eye, target, up):
    zax = normalized(eye - target)
    xax = normalized(np.cross(up, zax))
    yax = np.cross(zax, xax)
    x = - xax.dot(eye)
    y = - yax.dot(eye)
    z = - zax.dot(eye)
    return np.array(((xax[0], yax[0], zax[0], 0),
                     (xax[1], yax[1], zax[1], 0),
                     (xax[2], yax[2], zax[2], 0),
                     (     x,      y,      z, 1)))

def create_mvp(program_id, width, height):
    fov, near, far = 45, 0.1, 100
    eye = np.array((4,3,3))
    target, up = np.array((0,0,0)), np.array((0,1,0))
    projection = perspective(fov, width / height, near, far)
    view = look_at(eye, target, up)
    model = np.identity(4)
    mvp = model @ view @ projection
    matrix_id = gl.glGetUniformLocation(program_id, 'MVP')
    return matrix_id, mvp.astype(np.float32)

def main_loop(window, mvp_matrix_id, mvp):
    while (
        glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and
        not glfw.window_should_close(window)
    ):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set the view matrix
        gl.glUniformMatrix4fv(mvp_matrix_id, 1, False, mvp)

        # Draw the triangle
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

        glfw.swap_buffers(window)
        glfw.poll_events()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    width, height = 500, 400
    with create_main_window(width, height) as window:
        with create_vertex_buffer():
            with load_shaders() as prog_id:
                mvp_matrix_id, mvp = create_mvp(prog_id, width, height)
                main_loop(window, mvp_matrix_id, mvp)
