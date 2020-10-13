import numpy as np
import pandas as pd
import copy
import re

from h_project.motion_synthesis.quaternion_old import Quaternions

SKELETON_DF_KEY_BONE = 'bone'
SKELETON_DF_KEY_PARENT_INDEX = 'parent_index'
SKELETON_DF_KEY_OFFSET_X = 'offset_x'
SKELETON_DF_KEY_OFFSET_Y = 'offset_y'
SKELETON_DF_KEY_OFFSET_Z = 'offset_z'
SKELETON_DF_KEY_ROT_ORDER = 'rot_order'

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x' : 0,
    'y' : 1,
    'z' : 2,
}

# My dictionary maps
channel_map = {
    'Xposition': 'pos_x',
    'Yposition': 'pos_y',
    'Zposition': 'pos_z',
    'Xrotation': 'rot_x',
    'Yrotation': 'rot_y',
    'Zrotation': 'rot_z'
}

rotation_map = {
    'Xposition': '',
    'Yposition': '',
    'Zposition': '',
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

class SkeletonBVH():

    def __init__(self):

        self.num_frames = 0
        self.animation_df = None
        self.skeleton_df = None
        self.end_effectors_df = None
        self.num_frames = 0
        self.frame_time = 0

    def load_bvh_new(self, fpath):

        # Dataframe setup
        columns = [SKELETON_DF_KEY_BONE,
                   SKELETON_DF_KEY_PARENT_INDEX,
                   SKELETON_DF_KEY_OFFSET_X,
                   SKELETON_DF_KEY_OFFSET_Y,
                   SKELETON_DF_KEY_OFFSET_Z,
                   SKELETON_DF_KEY_ROT_ORDER]
        self.skeleton_df = pd.DataFrame(columns=columns)
        self.end_effectors_df = pd.DataFrame(columns=['bone', 'length'])

        # Temporarty storage
        parents_index_list = []
        bone_name_list = []
        offset_list = []
        rot_order_list = []
        end_effector_index_list = []
        end_effector_length_list = []

        # Initializers
        current_index = -1
        end_site = False
        animation_columns = []
        animation_data = None
        frame_index = 0

        file = open(fpath, "r")
        for line in file:

            # Skip non-important lines
            if any(key in line for key in ['HIERARCHY', 'MOTION', '{']):
                continue

            # ============== Skeleton ================
            if any(key in line for key in ['ROOT', 'JOINT']):
                bone_name_list.append(line.split()[1])
                parents_index_list.append(current_index)
                current_index = len(parents_index_list) - 1
                continue

            match_offset = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if match_offset:
                if end_site:
                    vector = np.array(list(map(np.float32, match_offset.groups())))
                    end_effector_index_list.append(current_index)
                    end_effector_length_list.append(np.linalg.norm(vector))
                else:
                    offset_list.append(list(map(np.float32, match_offset.groups())))
                continue

            match_channels = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if match_channels:
                channels_list = line.split()[2:]
                animation_columns.extend([f'{bone_name_list[-1]}_{channel_map[key]}' for key in channels_list])
                rotation_order = ''.join([rotation_map[key] for key in channels_list])
                rot_order_list.append(rotation_order)
                continue

            if "End Site" in line:
                end_site = True
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    current_index = parents_index_list[current_index]
                continue

            # =============== Frames and Frame time ================

            match_frames = re.match("\s*Frames:\s+(\d+)", line)
            if match_frames:
                self.num_frames = int(match_frames.group(1))
                animation_data = np.ndarray((self.num_frames, len(animation_columns)), dtype=np.float32)
                continue

            match_frame_time = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if match_frame_time:
                self.num_frames = np.float32(match_frame_time.group(1))
                continue

            # If you got here, it means you finished with the skeleton
            self.skeleton_df[SKELETON_DF_KEY_BONE] = bone_name_list
            self.skeleton_df[SKELETON_DF_KEY_PARENT_INDEX] = parents_index_list
            offsets = np.array(offset_list).astype(np.float32)
            self.skeleton_df[SKELETON_DF_KEY_OFFSET_X] = offsets[:, 0]
            self.skeleton_df[SKELETON_DF_KEY_OFFSET_Y] = offsets[:, 1]
            self.skeleton_df[SKELETON_DF_KEY_OFFSET_Z] = offsets[:, 2]
            self.skeleton_df[SKELETON_DF_KEY_ROT_ORDER] = rot_order_list

            # =============== Motion ================

            # If you got here, it means this is the motion data part of the file
            dmatch = line.strip().split()
            if dmatch:
                animation_data[frame_index, :] = np.array(list(map(np.float32, dmatch)))
                frame_index += 1

        self.animation_df = pd.DataFrame(columns=animation_columns,
                                         data=animation_data)
        self.end_effectors_df['bone'] = np.array(end_effector_index_list, dtype=np.int)
        self.end_effectors_df['length'] = np.array(end_effector_length_list, dtype=np.float32)

    def load(self, filename, start=None, end=None, order=None, world=False):
        """
        Reads a BVH file and constructs an animation

        Parameters
        ----------
        filename: str
            File to be opened

        start : int
            Optional Starting Frame

        end : int
            Optional Ending Frame

        order : str
            Optional Specifier for joint order.
            Given as string E.G 'xyz', 'zxy'

        world : bool
            If set to true euler angles are applied
            together in world space rather than local
            space

        Returns
        -------

        (animation, joint_names, frametime)
            Tuple of loaded animation and joint names
        """

        f = open(filename, "r")

        i = 0
        active = -1
        end_site = False

        names = []
        orients = Quaternions.id(0)
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)

        for line in f:

            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "{" in line: continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offmatch.groups()))])
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channelis:2 + channelie]
                    if any([p not in channelmap for p in parts]):
                        continue
                    order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match("\s*JOINT\s+(\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                if start and end:
                    fnum = (end - start) - 1
                else:
                    fnum = int(fmatch.group(1))
                jnum = len(parents)
                # result: [fnum, J, 3]
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                # result: [fnum, len(orients), 3]
                rotations = np.zeros((fnum, len(orients), 3))
                continue

            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                continue

            if (start and end) and (i < start or i >= end - 1):
                i += 1
                continue

            dmatch = line.strip().split()
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i - start if start else i
                if channels == 3:
                    # This should be root positions[0:1] & all rotations
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    # fill in all positions
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1

        f.close()

        rotations = Quaternions.from_euler(np.radians(rotations), order=order, world=world)

        return (Animation(rotations, positions, orients, offsets, parents), names, frametime)

    def _read_skeleton(self, file_object):

        pass