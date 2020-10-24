MWS_VERTEX_SHADER = """
#version 330 core
in vec4 world_position; 
in vec3 world_normal; 
in vec4 color;

out vec3 frag_position;
out vec3 frag_normal;
out vec4 frag_color;

uniform mat4 view_projection;


void main()
{
    frag_color = color;
    frag_normal = world_normal;
    frag_position = world_position.xyz;
    gl_Position = view_projection * world_position;
}
"""

MWS_FRAGMENT_SHADER = """
#version 330 core
in vec3 frag_position;
in vec3 frag_normal;
in vec4 frag_color;
out vec4 final_color;

uniform vec3 light_position;
uniform vec3 light_diffuse;
//uniform vec3 light_ambient;

void main()
{
    // Fucked up ONE
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * light_diffuse;
    vec3 norm = normalize(frag_normal);
    vec3 lightDir = normalize(light_position - frag_position);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * light_diffuse;
    vec3 result = (ambient + diffuse) * frag_color.rgb;

    final_color = vec4(result, 1.0);
    //final_color = vec4(result*0.001 + vec3(gl_FragCoord.z), 1.0);
} 
"""

MWW_VERTEX_SHADER = """
#version 330 core
in vec4 world_position; 
in vec4 color;

out vec3 frag_position;
out vec4 frag_color;

uniform mat4 view_projection;

void main()
{
    frag_color = color;
    gl_Position = view_projection * world_position;
}
"""

MWW_FRAGMENT_SHADER = """
#version 330 core
in vec3 frag_position;
in vec4 frag_color;
out vec4 final_color;

void main()
{
    final_color = frag_color;
    //final_color = frag_color*0.001 + vec4(vec3(gl_FragCoord.z), 1.0);
} 
"""