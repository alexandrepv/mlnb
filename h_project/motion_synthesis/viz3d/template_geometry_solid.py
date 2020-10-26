import numpy as np
import itertools

def create_cube(width=0.5, height=1.0, depth=0.5):

    vertices, normals = create_cylinder(num_sides=4)
    radius = np.sqrt(2)
    vertices *= np.array([width * radius, height, depth * radius, 1], dtype=np.float32)
    return vertices, normals

def create_cylinder(radius=1, height=1, num_sides=16):

    vertices = np.ndarray((num_sides * 12, 4), dtype=np.float32)
    normals = np.ndarray((num_sides * 12, 3), dtype=np.float32)

    offset_angle = np.pi / num_sides
    start_angle = offset_angle
    stop_angle = 2 * np.pi - offset_angle
    angles = np.linspace(start_angle, stop_angle, num_sides)  # minus the offset
    s = np.sin(angles)
    c = np.cos(angles)
    panel = np.ndarray((4, 4), dtype=np.float32)

    # Top and Bottom
    offset = 0
    for i in range(angles.size):
        i_next = (i + 1) % num_sides
        index = i * 6

        # Bottom
        vertices[index, :] = [0, 0, 0, 1]
        vertices[index + 1, :] = [s[i_next], 0, c[i_next], 1]
        vertices[index + 2, :] = [s[i], 0, c[i], 1]
        normals[index:(index + 3), :] = [0, -1, 0]

        # Top
        vertices[index + 3, :] = [0, 1, 0, 1]
        vertices[index + 4, :] = [s[i], 1, c[i], 1]
        vertices[index + 5, :] = [s[i_next], 1, c[i_next], 1]
        normals[(index + 3):(index + 6), :] = [0, 1, 0]
        offset += 6

    # Side Faces
    for i in range(angles.size):

        i_next = (i + 1) % num_sides
        panel[0, :] = [s[i], 0, c[i], 1]
        panel[1, :] = [s[i_next], 0, c[i_next], 1]
        panel[2, :] = [s[i], 1, c[i], 1]
        panel[3, :] = [s[i_next], 1, c[i_next], 1]

        index = i * 6 + offset
        vertices[index, :] = panel[0]
        vertices[index + 1, :] = panel[1]
        vertices[index + 2, :] = panel[3]
        vertices[index + 3, :] = panel[0]
        vertices[index + 4, :] = panel[3]
        vertices[index + 5, :] = panel[2]

        mean_vector = panel[0, 0:3] + panel[1, 0:3]
        mean_vector /= np.linalg.norm(mean_vector)
        normals[index:(index + 6), :] = mean_vector

    vertices *= np.array([radius, height, radius, 1], dtype=np.float32)

    return vertices, normals


def create_sphere(num_subdivisions=4, radius=1.0):

    if num_subdivisions < 1:
        num_subdivisions = 1

    num_face_vertices = num_subdivisions ** 2 * 24
    mark = np.arange(7) * num_face_vertices
    swap_commands = [[0, 1, 2, 1, 1, 1],  # index 0:3, sign 3:6
                     [0, 1, 2, -1, 1, -1],
                     [2, 1, 0, -1, 1, 1],
                     [2, 1, 0, 1, 1, -1],
                     [0, 2, 1, 1, -1, 1],
                     [0, 2, 1, 1, 1, -1]]

    total_num_vertices = num_face_vertices * len(swap_commands)
    face_vertices = np.ndarray((num_face_vertices, 3), dtype=np.float32)
    vertices = np.ndarray((total_num_vertices, 3), dtype=np.float32)

    index = 0
    dist = np.linspace(-1, 1, 2 * num_subdivisions + 1)
    for i in range(num_subdivisions*2):
        for j in range(num_subdivisions*2):
            face_vertices[index, :] = [dist[i], dist[j], 1]
            face_vertices[index + 1, :] = [dist[i+1], dist[j+1], 1]
            face_vertices[index + 2, :] = [dist[i], dist[j+1], 1]
            face_vertices[index + 3, :] = [dist[i], dist[j], 1]
            face_vertices[index + 4, :] = [dist[i + 1], dist[j], 1]
            face_vertices[index + 5, :] = [dist[i + 1], dist[j + 1], 1]
            index += 6

    for i, cmd in enumerate(swap_commands):
        a = mark[i]
        b = mark[i+1]
        for axis in range(3):
            vertices[a:b, axis] = cmd[axis+ 3] * face_vertices[:, cmd[axis]]

    vertices = (vertices.T / np.linalg.norm(vertices, axis=1).T).T
    normals = vertices.copy()

    return vertices, normals