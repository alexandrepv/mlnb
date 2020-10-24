import pywavefront

class Mesh:

    def __init__(self):

        self.vertices = None
        self.normals = None
        self.indices = None
        self.uv = None

    def load_obj(self, obj_fpath):

        scene = pywavefront.Wavefront(obj_fpath)