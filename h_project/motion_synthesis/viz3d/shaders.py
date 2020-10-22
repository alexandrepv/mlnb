MWS_VERTEX_SHADER = """
#version 330 core
in vec4 world_position;
in vec4 color;
//in vec3 world_normal;

out vec4 position_out;
//out vec3 normal_out;
out vec4 frag_color;

uniform mat4 view_projection;

void main()
{
    gl_Position = view_projection * world_position;
    frag_color = color;
}
"""

MWS_FRAGMENT_SHADER = """
#version 330 core
in vec4 frag_color;
out vec4 color_out;

void main()
{
    color_out = frag_color;
} 
"""

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