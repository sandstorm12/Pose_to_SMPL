import cv2
import torch
import random
import numpy as np

import matplotlib.pyplot as plt

from mmpose.apis import MMPoseInferencer

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def add_mmpose_keypoints():
    img_path = "/tmp/smpl.png"
    
    plt.savefig("/tmp/smpl.png", bbox_inches='tight', pad_inches=0)

    inferencer = MMPoseInferencer('rtmpose-x_8xb256-700e_body8-halpe26-384x288')

    image = cv2.imread(img_path)

    result_generator = inferencer(image)
    for result in result_generator:
        poeple_keypoints = result['predictions'][0]
        for idx_person, predictions in enumerate(poeple_keypoints):
            print(predictions.keys())
            keypoints = np.array(predictions['keypoints'])

            scale = np.sqrt(np.sum((keypoints[8] - keypoints[10]) ** 2))
            print(scale)

            for idx, point in enumerate(keypoints):
                x = int(point[0])
                y = int(point[1])

                # if idx == 6:
                #     y = int(y + .05 * scale)

                # if idx == 5:
                #     y = int(y + .05 * scale)

                if idx == 11:
                    x = int(x - .15 * scale)
                    y = int(y + .1 * scale)

                if idx == 12:
                    x = int(x + .15 * scale)
                    y = int(y + .1 * scale)
                
                if idx == 18:
                    y = int(y + .1 * scale)

                if idx == 19:
                    y = int(y - .1 * scale)

                cv2.circle(
                    image, (x, y), 4, (0, 0, 0),
                    thickness=-1, lineType=8)
                
    cv2.imwrite("./coco_smpl_correspondance.png", image)


def display_model(
        model_info,
        model_faces=None,
        with_joints=False,
        kintree_table=None,
        ax=None,
        batch_idx=0,
        show=True,
        savepath=None,
        only_joint=False):
    """
    Displays mesh batch_idx in batch of model_info, model_info as returned by
    generate_random_model
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts = model_info['verts'][batch_idx]
    joints = model_info['joints'][batch_idx]
    if model_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
    elif not only_joint:
        mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if with_joints:
        draw_skeleton(joints, kintree_table=kintree_table, ax=ax)

    def on_press(event):
        print(event.key)
        if event.key == 'shift+right':
            # ax.elev+=5
            # ax.azim += 5
            ax.roll += 5
        if event.key == 'shift+left':
            # ax.elev-=5
            # ax.azim -= 5
            ax.roll -= 5
        if event.key == 'shift+up':
            ax.elev += 5
        if event.key == 'shift+down':
            ax.elev -= 5
        if event.key == 'right':
            ax.azim += 5
        if event.key == 'left':
            ax.azim -= 5
        if event.key == 'd':
            add_mmpose_keypoints()
        fig.canvas.draw_idle()
        

    fig.canvas.mpl_connect('key_press_event', on_press)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax.view_init(azim=-90, elev=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if savepath:
        # print('Saving figure at {}.'.format(savepath))
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

    return fig


def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=True):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    colors = []
    left_right_mid = ['r', 'g', 'b']
    kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
    for c in kintree_colors:
        colors += left_right_mid[c]
    # For each 24 joint
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                [joints3D[j1, 1], joints3D[j2, 1]],
                [joints3D[j1, 2], joints3D[j2, 2]],
                color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_numbers:
            ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    return ax


if __name__ == '__main__':
    cuda = True
    batch_size = 1

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='male',
        model_root='smplpytorch/native/models')

    # Generate random pose and shape parameters
    pose_params = torch.rand(batch_size, 72) * 0.01
    shape_params = torch.rand(batch_size, 10) * 0.03

    # # GPU mode
    # if cuda:
    #     pose_params = pose_params.cuda()
    #     shape_params = shape_params.cuda()
    #     smpl_layer.cuda()

    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

    # Draw output vertices and joints
    display_model(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        savepath='image.png',
        show=True)
