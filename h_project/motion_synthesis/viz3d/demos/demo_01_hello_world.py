import h_project.motion_synthesis.viz3d.viz3d as viz3d
import numpy as np
import pyrr
import time

if __name__ == "__main__":

    app = viz3d.Viz3D()

    app.initialise(window_width=1920, window_height=1080)

    t0 = time.time()
    while app.render():
        time_elapsed = time.time() - t0
        transform = pyrr.matrix44.create_from_y_rotation(time_elapsed)
        color = np.array([1, 0.5, 0.31, 1], dtype=np.float32)
        app.mws.add_cuboid(transform, 1.0, 1.0, 1.0, color)


    app.shutdown()