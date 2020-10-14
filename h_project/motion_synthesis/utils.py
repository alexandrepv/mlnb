import numpy as np

def rotate_euler(x_rad, y_rad, z_rad, order='xyz'):

    """
    Returns a 3x3 rotation matrix based on the angles and the rotation order
    :param x_rad: Angle in radians
    :param y_rad: Angle in radians
    :param z_rad: Angle in radians
    :param order: string with the order of axes
    :return: numpy ndarray (3, 3) <float32>
    """

    cx = np.cos(x_rad)
    sx = np.sin(x_rad)
    cy = np.cos(y_rad)
    sy = np.sin(y_rad)
    cz = np.cos(z_rad)
    sz = np.sin(z_rad)

    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    rotation = np.eye(3, dtype=np.float32)

    for axis in order.lower():
        if axis == 'x':
            rotation = np.matmul(rotation, rx)
        elif axis == 'y':
            rotation = np.matmul(rotation, ry)
        else:
            rotation = np.matmul(rotation, rz)

    return rotation

def transform_euler(x_rad, y_rad, z_rad, pos_vector, rot_order='xyz'):

    tr = np.eye(4)
    tr[:3, :3] = rotate_euler(x_rad=x_rad,
                              y_rad=y_rad,
                              z_rad=z_rad,
                              order=rot_order)
    tr[:3, 3] = pos_vector
    return tr
