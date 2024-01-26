import pickle
import time
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


path_skeleton = "/home/hamid/Documents/classiccv/footefield/pose_estimation/skeleton_numpypkl/a2_azure_kinect3_4_calib_snap_skeleton_3D_smooth.npy"
path_joint = "fit/output/HALPE/a2_azure_kinect3_4_calib_snap_skeleton_3D_smooth_params.pkl"


def init_graph(poses, ax):
    lines = []
    graphs = []

    # axis = [0, 1, 0]
    # theta = math.radians(45)
    # axis = axis / norm([axis])
    # rot = Rotation.from_rotvec(theta * axis)
    # keypoints = rot.apply(poses[0])
    keypoints = poses[0]
    
    x = [point[0] for point in keypoints]
    y = [point[2] for point in keypoints]
    z = [point[1] for point in keypoints]

    graph = ax.scatter(x, y, z, c='r', marker='o')
    graphs.append(graph)
    
    return graphs, lines


def update_graph(idx, poses, graphs, lines, title):
    # axis = [0, 1, 0]
    # theta = math.radians(45)
    # axis = axis / norm([axis])
    # rot = Rotation.from_rotvec(theta * axis)
    # keypoints = rot.apply(poses[idx])
    keypoints = poses[idx]

    # Define the data for the scatter plot
    x = [point[0] for point in keypoints]
    y = [point[2] for point in keypoints]
    z = [point[1] for point in keypoints]

    graphs[0]._offsets3d = (x, y, z)

    title.set_text('3D Test, time={}'.format(idx))

# Its too long
# Make it also more robust
def visualize_poses(poses, elev=1, azim=-89, roll=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    graphs, lines = init_graph(poses, ax)

    ax.view_init(elev=elev, azim=azim, roll=roll)

    # # Remove the grid background
    # ax.grid(False)

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    RANGE = 1
    ax.axes.set_xlim3d(-RANGE, RANGE)
    ax.axes.set_zlim3d(-RANGE, RANGE)
    ax.axes.set_ylim3d(-RANGE, RANGE)

    ani = matplotlib.animation.FuncAnimation(
        fig, update_graph, len(poses), fargs=(poses, graphs, lines, title),
        interval=100, blit=False)
    ani.save(f'./anim_{time.time()}.gif', fps=5)

    plt.show()


if __name__ == "__main__":
    with open(path_joint, "rb") as handle:
        joints = pickle.load(handle)

    print("Pose to SMPL:", type(joints), joints.keys())

    poses = np.array(joints['Jtr'])
    visualize_poses(poses, 10, 141, 0)

    poses = np.load(path_skeleton)
    print(poses.shape)
    visualize_poses(poses, 10, 141, 0)
