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

    // Something fucking else
    vec3 light_position2 = vec3(1.2, 1, 2);
    vec3 light_color2 = vec3(1, 1, 1);
    vec3 ambient2 = 0.1 * light_color2;
    vec3 norm2 = normalize(frag_normal);
    vec3 lightDir2 = normalize(light_position2 - frag_position);
    float diff2 = max(dot(norm2, lightDir2), 0.0);
    vec3 diffuse2 = diff2 * light_color2;
    vec3 result2 = (ambient2 + diffuse2) * frag_color.rgb;
    
    final_color = vec4(result, 1.0);
} 
"""