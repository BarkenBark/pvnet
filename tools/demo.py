import sys

sys.path.append('.')
sys.path.append('..')
from lib.networks.model_repository import *
from lib.utils.arg_utils import args
from lib.utils.net_utils import smooth_l1_loss, load_model, compute_precision_recall
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from lib.utils.evaluation_utils import pnp, Evaluator
from lib.utils.draw_utils import imagenet_to_uint8, visualize_bounding_box, visualize_mask,visualize_vertex_field, visualize_overlap_mask, visualize_points
from lib.utils.base_utils import Projector
from lib.utils.extend_utils.extend_utils import uncertainty_pnp
import json

from lib.utils.config import cfg

from torch.nn import DataParallel
from torch import nn, optim
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy

with open(args.cfg_file, 'r') as f:
    train_cfg = json.load(f)
train_cfg['model_name'] = '{}_{}'.format(args.linemod_cls, train_cfg['model_name'])

vote_num = 9


class NetWrapper(nn.Module):
    def __init__(self, net):
        super(NetWrapper, self).__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, image, mask, vertex, vertex_weights):
        seg_pred, vertex_pred = self.net(image)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)
        return seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall


class EvalWrapper(nn.Module):
    def forward(self, seg_pred, vertex_pred, use_argmax=True):
        vertex_pred = vertex_pred.permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex_pred.shape
        vertex_pred = vertex_pred.view(b, h, w, vn_2 // 2, 2)
        if use_argmax:
            mask = torch.argmax(seg_pred, 1)
        else:
            mask = seg_pred
        return ransac_voting_layer_v3(mask, vertex_pred, 512, inlier_thresh=0.99)

class EvalWrapper2(nn.Module):
    def forward(self, seg_pred, vertex_pred, use_argmax=True):
        vertex_pred = vertex_pred.permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex_pred.shape
        vertex_pred = vertex_pred.view(b, h, w, vn_2 // 2, 2)
        if use_argmax:
            mask = torch.argmax(seg_pred, 1)
        else:
            mask = seg_pred
        mean = ransac_voting_layer_v3(mask, vertex_pred, 512, inlier_thresh=0.99)
        mean, covar = estimate_voting_distribution_with_mean(mask, vertex_pred, mean) # [b,vn,2], [b,vn,2,2]
        return mean, covar

def compute_vertex(mask, points_2d):
    num_keypoints = points_2d.shape[0]
    h, w = mask.shape
    m = points_2d.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = xy[:, None, :] * np.ones(shape=[1, num_keypoints, 1])
    vertex = points_2d[None, :, :2] - vertex
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm < 1e-3] += 1e-3
    vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    return np.reshape(vertex_out, [h, w, m * 2])


def read_data():
    import torchvision.transforms as transforms

    demo_dir_path = os.path.join(cfg.DATA_DIR, 'demo')
    rgb = Image.open(os.path.join(demo_dir_path, 'cat.jpg'))
    mask = np.array(Image.open(os.path.join(demo_dir_path, 'cat_mask.png')).convert('RGB')).astype(np.int32)[..., 0]
    mask[mask != 0] = 1
    print(mask.shape)
    points_3d = np.loadtxt(os.path.join(demo_dir_path, 'cat_points_3d.txt'))
    bb8_3d = np.loadtxt(os.path.join(demo_dir_path, 'cat_bb8_3d.txt'))
    pose = np.load(os.path.join(demo_dir_path, 'cat_pose.npy'))

    projector = Projector()
    points_2d = projector.project(points_3d, pose, 'linemod')
    vertex = compute_vertex(mask, points_2d)

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    rgb = transformer(rgb)
    vertex = torch.tensor(vertex, dtype=torch.float32).permute(2, 0, 1)
    mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.int64)
    vertex_weight = mask.unsqueeze(0).float()
    pose = torch.tensor(pose.astype(np.float32))
    points_2d = torch.tensor(points_2d.astype(np.float32))
    data = (rgb, mask, vertex, vertex_weight, pose, points_2d)

    return data, points_3d, bb8_3d


def demo():
    net = Resnet18_8s(ver_dim=vote_num * 2, seg_dim=2)
    net = NetWrapper(net).cuda()
    net = DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
    model_dir = os.path.join(cfg.MODEL_DIR, 'cat_demo')
    load_model(net.module.net, optimizer, model_dir, args.load_epoch)
    data, points_3d, bb8_3d = read_data()
    image, mask, vertex, vertex_weights, pose, corner_target = [d.unsqueeze(0).cuda() for d in data]

    # Run the net
    seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = net(image, mask, vertex, vertex_weights)

    print('vertex_pred.shape')
    print(vertex_pred.shape)
    print(' ')

    print('vertex_pred[0]')
    print(vertex_pred)
    print(' ')

    # Various visualizations
    #visualize_vertex_field(vertex_pred,vertex_weights, keypointIdx=3)
    print('asdasdsadas')
    print(seg_pred.shape, mask.shape)
    visualize_mask(np.squeeze(seg_pred.cpu().detach().numpy()), mask.cpu().detach().numpy())
    rgb = Image.open('data/demo/cat.jpg')
    img = np.array(rgb)
    #visualize_overlap_mask(img, np.squeeze(seg_pred.cpu().detach().numpy()), None)

    # Run the ransac voting
    eval_net = DataParallel(EvalWrapper2().cuda())
    #corner_pred = eval_net(seg_pred, vertex_pred).cpu().detach().numpy()[0]
    corner_pred, covar = [x.cpu().detach().numpy()[0] for x in eval_net(seg_pred, vertex_pred)]
    print('Keypoint predictions:')
    print(corner_pred)
    print(' ')
    print('covar: ', covar)
    print(' ')
    camera_matrix = np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]])


    # Fit pose to points
    #pose_pred = pnp(points_3d, corner_pred, camera_matrix)
    #evaluator = Evaluator()
    #pose_pred = evaluator.evaluate_uncertainty(corner_pred,covar,pose,'cat',intri_matrix=camera_matrix)

    def getWeights(covar):
        cov_invs = []
        for vi in range(covar.shape[0]): # For every keypoint
            if covar[vi,0,0]<1e-6 or np.sum(np.isnan(covar)[vi])>0:
                cov_invs.append(np.zeros([2,2]).astype(np.float32))
                continue
            cov_inv = np.linalg.inv(scipy.linalg.sqrtm(covar[vi]))
            cov_invs.append(cov_inv)
        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]
        return weights

    weights = getWeights(covar)
    pose_pred = uncertainty_pnp(corner_pred, weights, points_3d, camera_matrix)

    print('Predicted pose: \n', pose_pred)
    print('Ground truth pose: \n', pose[0].detach().cpu().numpy())
    print(' ')

    projector = Projector()
    bb8_2d_pred = projector.project(bb8_3d, pose_pred, 'linemod')
    bb8_2d_gt = projector.project(bb8_3d, pose[0].detach().cpu().numpy(), 'linemod')
    image = imagenet_to_uint8(image.detach().cpu().numpy())[0]
    visualize_points(image[None, ...], corner_target.detach().cpu().numpy(), pts_pred=corner_pred[None,:,:])
    visualize_bounding_box(image[None, ...], bb8_2d_pred[None, None, ...], bb8_2d_gt[None, None, ...])


if __name__ == "__main__":
    demo()
