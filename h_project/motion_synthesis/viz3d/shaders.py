VERTEX_SHADER = """

    #version 330

    in vec3 position;
    in vec3 color;
    out vec3 newColor;

    uniform mat4 transform; 

    void main() {

     gl_Position = transform * vec4(position, 1.0f);
     newColor = color;

      }


"""

FRAGMENT_SHADER = """
    #version 330

    in vec3 newColor;
    out vec4 outColor;

    void main() {

      outColor = vec4(newColor, 1.0f);

    }

"""