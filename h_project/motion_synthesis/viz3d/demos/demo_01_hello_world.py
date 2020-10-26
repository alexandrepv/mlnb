import h_project.motion_synthesis.viz3d.viz3d as viz3d
import numpy as np
import pyrr
import time

if __name__ == "__main__":

    app = viz3d.Viz3D()

    app.initialise(window_width=1920, window_height=1080)

    identity = np.eye(4, dtype=np.float32)

    points = np.random.rand(100, 4)
    points[:, 3] = 1

    color = np.array([1, 0.5, 0.31, 1], dtype=np.float32)
    color2 = np.array([0.3, .9, .2, 1], dtype=np.float32)

    t0 = time.time()
    while app.render():
        time_elapsed = time.time() - t0
        transform = pyrr.matrix44.create_from_y_rotation(0)

        #app.mws.add_cuboid(transform, 1.0, 1.0, 1.0, color)
        app.mww.add_axes(transform=identity, axis_size=3)
        #app.mww.add_lines(transform=identity, vertices=points, colorRGBA=color2)
        #app.mws.add_cylinder32(transform=transform, height=0.25, radius=1, colorRGBA=color)
        app.mws.add_sphere_3(transform=transform, radius=0.5, colorRGBA=color)
        app.mww.add_xz_grid()

    app.shutdown()