#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import glob
from pathlib import Path
import open3d as o3d
from visual_utils import open3d_vis_utils as V
from lidar_detect.msg import lidar_3dbox_array
from lidar_detect.msg import lidar_3dbox
import vis3dbox
import time
import warnings
warnings.filterwarnings('ignore')


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class PointCloudSubscriber(object):
    def __init__(self) -> None:
        self.detector = vis3dbox.Detector()
        self.a = 0
        # self.sub = rospy.Subscriber("/points_raw",PointCloud2,self.callback, queue_size=5)
        self.sub = rospy.Subscriber('/velodyne_points',PointCloud2,self.callback, queue_size=1)
        self.pub = rospy.Publisher('/lidarobj', lidar_3dbox_array, queue_size=1)
        self.points = o3d.utility.Vector3dVector()
        self.lines = o3d.utility.Vector2iVector()
        self.lpoints = o3d.utility.Vector3dVector()

    def callback(self, msg):
        
        # print('time:')
        # print('{}.{}'.format(msg.header.stamp.secs, msg.header.stamp.nsecs))
        # print(time.time())
        assert isinstance(msg, PointCloud2)
        # gen=point_cloud2.read_points(msg,field_names=("x","y","z"))
        points = point_cloud2.read_points_list(
            msg, field_names=("x", "y", "z"))
        ourpoints = []
        # print(time.time()-float('{}.{}'.format(msg.header.stamp.secs, msg.header.stamp.nsecs)))
        self.a = self.a+1
        if self.a%3!=1:
            return
        for point in points:
            ourpoints.append([point.x,point.y,point.z,0])
        ourpoints = np.array(ourpoints)
        # np.save('points.npy', ourpoints[:,:-1])
        ourpoints = ourpoints[ourpoints[:,0]<10]
        ourpoints = ourpoints[ourpoints[:,1]<5]
        ourpoints = ourpoints[ourpoints[:,1]>-5]
        ourpoints = ourpoints[ourpoints[:,0]>0.1]
        self.points = o3d.utility.Vector3dVector(ourpoints[:,:-1])
        # print(time.time())
        # print(time.time()-float('{}.{}'.format(msg.header.stamp.secs, msg.header.stamp.nsecs)))
        ourpoints = torch.from_numpy(ourpoints).cuda().float()
        t1 = time.time()
        with torch.no_grad():
            input_dict = {'points':ourpoints, 'frame_id':self.a}
            data_dict = demo_dataset.prepare_data(data_dict=input_dict)
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
        
            # print(pred_dicts[0]['pred_boxes'].shape)
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            score = pred_dicts[0]['pred_scores'].cpu().numpy()
            print(score)
            if score.shape[0]>0:
                # print(score.shape, pred_boxes.shape)
                gtbox = pred_boxes[score[:]==np.max(score)]
                gtbox = gtbox[0]
                # print(gtbox.shape)
                lines, lpoints = self.translate_boxes_to_open3d_instance(gtbox)
                
                self.lines = lines
                self.lpoints = lpoints
            else:
                self.lines = o3d.utility.Vector2iVector()
                self.lpoints = o3d.utility.Vector3dVector()
                

            # np.save('pred_boxes.npy', pred_boxes)
            '''
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            '''
            boxes = []
            for x,y,z,w,l,h,heading in pred_boxes:
                box = self.detector.get_3d_box((x,y,z),(l,w,h),heading)
                box=box.transpose(1,0).ravel()
                boxes.append(box)
            if len(boxes)>0:
                self.detector.display(boxes)
            center = pred_dicts[0]['pred_boxes'][:,0:3]
            lhw = pred_dicts[0]['pred_boxes'][:,3:6]
            # print(score)
            prelable = pred_dicts[0]['pred_labels']
            boxarray = lidar_3dbox_array()
            for i in range(center.shape[0]):
                box = lidar_3dbox()
                # box.header.frame_id='velodyne'
                position = center.cpu().numpy()[i]
                box.position = '{},{},{}'.format(position[0],position[1],position[2])
                box.size = str(lhw.cpu().numpy()[i])
                box.score = score[i]
                box.label = prelable.cpu().numpy()[i]
                boxarray.boxes.append(box)
            self.pub.publish(boxarray)
        
        


    def translate_boxes_to_open3d_instance(self, gt_boxes):
        """
                4-------- 6
            /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1
            |/         |/
            2 -------- 0
        """
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set1 = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set1.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
        line_set1.lines = o3d.utility.Vector2iVector(lines)
        points = line_set1.points
        lines = o3d.utility.Vector2iVector(lines)
        return lines, points

logger = common_utils.create_logger()
ckpt = 'ckpt/pvrcnn.pth'
cfg_from_yaml_file('ckpt/pvrcnn.yaml', cfg)
demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path('data/000000.bin'), ext='.bin', logger=logger
    )

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
model.cuda()
model.eval()
point_list = o3d.geometry.PointCloud()
points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]
points = o3d.utility.Vector3dVector(points)


if __name__ =='__main__':
    # time.sleep(5)
    rospy.init_node("pointcloud_subscriber")
    pcds = PointCloudSubscriber()
    # rospy.spin()
    # exit()
    time.sleep(5)
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Openpcd Lidar',
        width=960,
        height=540,
        left=480,
        top=270)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    
    point_list.points = pcds.points
    lines = pcds.lines
    lpoints = pcds.lpoints
    line_set = o3d.geometry.LineSet(points=lpoints, lines=lines)
    line_set.lines = pcds.lines
    color = [[0,0.8,0.1]for i in range(14)]
    line_set.colors = o3d.utility.Vector3dVector(color)
    vis.add_geometry(line_set)
    vis.add_geometry(point_list)
    
    while not rospy.is_shutdown():
        if not pcds.lpoints is (None):
            # print('update')
            line_set.lines = pcds.lines
            line_set.points = pcds.lpoints
            vis.update_geometry(line_set)
            vis.poll_events()
            vis.update_renderer()
        # o3d.visualization.draw_geometries(pcds.line_set)
        if not pcds.points is None:
            point_list.points = pcds.points
            vis.update_geometry(point_list)
            vis.poll_events()
            vis.update_renderer()
        # # This can fix Open3D jittering issues:
        time.sleep(0.08)

