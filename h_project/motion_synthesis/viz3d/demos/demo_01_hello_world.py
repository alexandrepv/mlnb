import h_project.motion_synthesis.viz3d.viz3d as viz3d
import numpy as np

if __name__ == "__main__":

    app = viz3d.Viz3D()

    app.initialise()

    while app.render():
        identity = np.eye(4, dtype=np.float32)
        color = np.array([0, 1, 0, 0.1], dtype=np.float32)
        app.debug_mesh_solid.add_cuboid(identity, 1.0, 1.0, 1.0, color)
        pass

    app.shutdown()