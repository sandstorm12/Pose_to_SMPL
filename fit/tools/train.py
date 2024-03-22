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
    params["shape_params"] = torch.zeros(target.shape[0], 10)
    params["scale"] = torch.ones([1])
    params["translation"] = torch.zeros([1, 3])

    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    params["scale"] = params["scale"].to(device)
    params["translation"] = params["translation"].to(device)

    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)
    params["translation"].requires_grad = True

    optim_params = [{'params': params["pose_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["shape_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["scale"], 'lr': cfg.TRAIN.LEARNING_RATE*10},
                    {'params': params["translation"], 'lr': cfg.TRAIN.LEARNING_RATE*10}]
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
    translation = params["translation"]
    
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
        loss = F.smooth_l1_loss(scale*Jtr.index_select(1, index["smpl_index"]) + translation,
                                target.index_select(1, index["dataset_index"]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters.update_early_stop(float(loss))
        if meters.update_res:
            res = [pose_params, shape_params, verts, Jtr, scale.detach().numpy(), translation.detach().numpy()]
        if meters.early_stop:
            logger.info("Early stop at epoch {} !".format(epoch))
            break

        if epoch % cfg.TRAIN.WRITE == 0 or epoch<10:
            # logger.info("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
            #         epoch, float(loss),float(scale)))
            print("Epoch {}, lossPerBatch={:.6f}, scale={:.4f} translation={}".format(
                     epoch, float(loss),float(scale), str(translation)))
            writer.add_scalar('loss', float(loss), epoch)
            writer.add_scalar('learning_rate', float(
                optimizer.state_dict()['param_groups'][0]['lr']), epoch)
            # save_single_pic(res,smpl_layer,epoch,logger,args.dataset_name,target)

    logger.info('Train ended, min_loss = {:.4f}'.format(
        float(meters.min_loss)))
    return res
