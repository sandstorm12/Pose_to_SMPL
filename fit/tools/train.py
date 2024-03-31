import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

from tqdm import tqdm
sys.path.append(os.getcwd())
from save import save_single_pic



def init(smpl_layer, target, device, cfg):
    params = {}
    params["pose_params"] = torch.zeros(target.shape[0], 72)
    params["shape_params"] = torch.rand(target.shape[0], 10) * 0.03
    params["scale"] = torch.ones([1])
    # params["translation"] = torch.zeros([1, 3])
    # params["rotation"] = torch.zeros([3, 3])
    params["transformation"] = torch.eye(4)

    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    params["scale"] = params["scale"].to(device)
    # params["translation"] = params["translation"].to(device)
    # params["rotation"] = params["rotation"].to(device)
    params["transformation"] = params["transformation"].to(device)

    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)
    # params["translation"].requires_grad = True
    # params["rotation"].requires_grad = True
    params["transformation"].requires_grad = True

    optim_params = [{'params': params["pose_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["shape_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["scale"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    # {'params': params["translation"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    # {'params': params["rotation"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["transformation"], 'lr': cfg.TRAIN.LEARNING_RATE},]
    optimizer = optim.Adam(optim_params)

    index = {}
    smpl_index = []
    dataset_index = []
    for tp in cfg.DATASET.DATA_MAP:
        smpl_index.append(tp[0])
        dataset_index.append(tp[1])

    index["smpl_index"] = torch.tensor(smpl_index).to(device)
    index["dataset_index"] = torch.tensor(dataset_index).to(device)

    return smpl_layer, params, target, optimizer, index


def train(smpl_layer, target,
          logger, writer, device,
          args, cfg, meters):
    res = []
    smpl_layer, params, target, optimizer, index = \
        init(smpl_layer, target, device, cfg)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]
    # translation = params["translation"]
    # rotation = params["rotation"]
    transformation = params["transformation"]
    
    with torch.no_grad():
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        params["scale"] *= (torch.max(torch.abs(target.index_select(1, index["dataset_index"])))/torch.max(Jtr.index_select(1, index["smpl_index"])))
        # print("Scale", target.index_select(1, index["dataset_index"]).shape)
        # params["scale"] *= (
        #     (torch.max(target.index_select(1, index["dataset_index"])[:,:,1]) - torch.min(target.index_select(1, index["dataset_index"])[:,:,1])) \
        #         / (torch.max(Jtr.index_select(1, index["smpl_index"])[:,:,1]) - torch.min(Jtr.index_select(1, index["smpl_index"])[:,:,1]))
        # )

        # params["scale"] *= 1.02 # Durect or inverse with the SMPL

    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        # Add rotation term
        # Invert SMPL relation with the coefficient
        # print(Jtr.index_select(1, index["smpl_index"]).shape)
        # print(torch.bmm(Jtr.index_select(1, index["smpl_index"]), rotation).shape)
        # print(torch.matmul(Jtr.index_select(1, index["smpl_index"]), rotation).shape)
        # print(torch.matmul(rotation, Jtr.index_select(1, index["smpl_index"])).shape)
        # print(torch.bmm(rotation, Jtr.index_select(1, index["smpl_index"])).shape)
        points_tensor = Jtr.index_select(1, index["smpl_index"])
        points_tensor = torch.cat((
            points_tensor,
            torch.ones(points_tensor.shape[0], points_tensor.shape[1], 1)), dim=2)
        points_tensor = (torch.matmul(points_tensor, transformation))
        points_tensor = points_tensor[:, :, :3] / points_tensor[:, :, -1:]
        points_tensor = points_tensor * scale
        loss = F.smooth_l1_loss(points_tensor,
                                target.index_select(1, index["dataset_index"])) + torch.abs(1 - torch.det(transformation))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters.update_early_stop(float(loss))
        if meters.update_res:
            res = [pose_params, shape_params, verts, Jtr, scale.detach().numpy(),
                   transformation.detach().numpy()]
                #    , rotation.detach().numpy(), translation.detach().numpy()]
        if meters.early_stop:
            logger.info("Early stop at epoch {} !".format(epoch))
            break

        if epoch % cfg.TRAIN.WRITE == 0 or epoch<10:
            # logger.info("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
            #         epoch, float(loss),float(scale)))
            print("Epoch {}, lossPerBatch={:.6f}, scale={:.4f} transformation={}".format(
                     epoch, float(loss),float(scale), str(transformation)))
            print("Deter:", torch.det(transformation))
            writer.add_scalar('loss', float(loss), epoch)
            writer.add_scalar('learning_rate', float(
                optimizer.state_dict()['param_groups'][0]['lr']), epoch)
            # save_single_pic(res,smpl_layer,epoch,logger,args.dataset_name,target)

    logger.info('Train ended, min_loss = {:.4f}'.format(
        float(meters.min_loss)))
    return res
