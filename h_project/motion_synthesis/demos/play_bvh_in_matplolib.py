import numpy as np
import re


from h_project.motion_synthesis.preprocessing import MocapParameterizer
from h_project.motion_synthesis.parsers import BVHParser
from h_project.motion_synthesis.skeleton_bvh import SkeletonBVH


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

fpath = r'D:\data\mocap\mixamo_full_bvh_animations\catwalk_walk.bvh'

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# ====================== Parse BVH file ========================

bvh_parser = BVHParser()
mocap = [bvh_parser.parse(filename=fpath)]
skeleton_bvh = SkeletonBVH()
skeleton_bvh.load_bvh_new(fpath)

fig = plt.figure()
ax = p3.Axes3D(fig)

lines = skeleton_bvh.generate_skeleton_line_representation(frame=0)
line_plot_list = [ax.plot(lines[i, :, 0]*0.5,
                          lines[i, :, 1],
                          lines[i, :, 2]*0.5)[0] for i in range(lines.shape[0])]
axisEqual3D(ax)

def update_plot(frame):
    lines = skeleton_bvh.generate_skeleton_line_representation(frame=frame)
    for i, line_plot in enumerate(line_plot_list):
        line_plot.set_xdata(lines[i, :, 0] * 0.5)
        line_plot.set_ydata(lines[i, :, 1])
        line_plot.set_3d_properties(lines[i, :, 2] * 0.5, zdir='y')  # z axis

ax.set_xlim3d([-50, 50])
ax.set_xlabel('X')
ax.set_ylim3d([-50, 50])
ax.set_ylabel('Y')
ax.set_zlim3d([-100, 100])
ax.set_zlabel('Z')
ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_plot, skeleton_bvh.animation_df.index.size, interval=1, blit=False)
plt.show()

BVH2Pos = MocapParameterizer('position')
data_pos = BVH2Pos.fit_transform(mocap)
num_frames = data_pos[0].values.index.size

# ================ Get bone positions ready for display ============
joints_df = data_pos[0].values
columns = joints_df.columns
positions_str = ['Xposition', 'Zposition', 'Yposition']
all_relevant_bones = []
for pos_str in positions_str:
    for column in columns:
        pattern = r'(.*)_' + f'{pos_str}'
        matches = re.match(pattern, column)
        if matches is not None:
            all_relevant_bones.append(matches[1])

unique_bones = np.unique(all_relevant_bones)
positions = np.ndarray((3, unique_bones.size, num_frames))

for j, pos_str in enumerate(positions_str):
    for i, bone in enumerate(unique_bones):
        positions[j, i, :] = data_pos[0].values[f'{bone}_{pos_str}']

# TODO: This was added because of the 3D plot bloody wrong ratio!
positions[0, :, :] *= 0.5
positions[1, :, :] *= 0.5

# Zero all positions based on hip (ROOT) position
#positions_zeroed = positions.copy()
#positions_zeroed[0, :] = positions_zeroed

# ========================= Animation ========================


area_radius = 50
area_height = 200

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)
axisEqual3D(ax)

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(positions[0, i, 0],
                 positions[1, i, 0],
                 positions[2, i, 0], 'o')[0] for i in range(unique_bones.size)]

def update_lines(frame) :
    for i, line in enumerate(lines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(positions[0:2, i, frame])
        print(positions[2, i, frame].shape)
        line.set_3d_properties(positions[2, i, frame])

        x = positions[0, i, frame]
        y = positions[1, i, frame]

        # Setting the axes properties
        ax.set_xlim3d([x-area_radius, x+area_radius])
        ax.set_ylim3d([y-area_radius, y+area_radius])
    return lines

# Setting the axes properties
ax.set_xlim3d([-15, 20])
ax.set_xlabel('X')

ax.set_ylim3d([-25, 40])
ax.set_ylabel('Y')

ax.set_zlim3d([0, area_height])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, num_frames, interval=10, blit=False)
plt.show()