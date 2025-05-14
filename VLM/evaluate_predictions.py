import json
import os
import pickle
import time
import datetime
import re
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ProcessPoolExecutor
import ast
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import re
import os
import py3d
import trimesh
import json
import pickle
import json
import open3d as o3d
import torch
import trimesh
from shapely.geometry import Polygon
import math



#Evaluation Area Level IoU





def create_rotated_bounding_box(center, size, rotation_angle, rotation_axis=[0, 0, 1]):
    #print("calling....")
    #print(center, size, rotation_angle)
    """
    创建带旋转的 3D 盒子
    :param center: 盒子的中心点 (x, y, z)
    :param size: 盒子的大小 (width, height, depth)
    :param rotation_angle: 旋转角度 (单位：度)
    :param rotation_axis: 旋转轴，默认绕 z 轴
    :return: 旋转后的 trimesh box
    """
    box = trimesh.creation.box(extents=size)

    # 计算旋转矩阵 (单位转换为弧度)
    R = trimesh.transformations.rotation_matrix(np.radians(rotation_angle), rotation_axis)

    # 应用旋转
    box.apply_transform(R)

    # 移动到指定中心
    box.apply_translation(center)
    # if not box.is_volume:
    #     print("Box is not a valid 3D volume!")
    #     print(center, size, rotation_angle)

    return box


def is_valid_volume(mesh):
    """ 判断网格是否是有效的体积 """
    return mesh.is_volume and mesh.is_watertight

def compute_3d_iou(box1, box2):
    """ 计算两个旋转 bounding box 的 3D IoU """
    # if not box1.is_watertight:
    #     box1 = trimesh.repair.fill_holes(box1)
    # if not box2.is_watertight:
    #     box2 = trimesh.repair.fill_holes(box2)
    # if not box1.is_volume:
    #     print("no box1")
    # if not box2.is_volume:
    #     print("no box2")

    intersection = box1.intersection(box2)
    intersection_volume = intersection.volume
    # if isinstance(intersection, trimesh.Trimesh) and intersection.is_volume:
    #     # 如果交集不是封闭的网格，则进行修复
    #     if not intersection.is_watertight:
    #         intersection = trimesh.repair.fill_holes(intersection)

    #     # 计算交集体积
    #     intersection_volume = intersection.volume
    # else:
    #     # 如果没有交集，或者交集不是一个有效的网格
    #     intersection_volume = 0.0

    volume1 = box1.volume
    volume2 = box2.volume

    union_volume = volume1 + volume2 - intersection_volume
    iou = intersection_volume / union_volume if union_volume > 0 else 0.0
    return iou





def compute_2d_iou(box1, box2, drop_axis):
    """ 使用 Shapely 计算 2D IoU """
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[drop_axis]
    
    # 提取投影到 2D 平面的顶点坐标
    vertices1 = np.delete(box1.vertices, axis_idx, axis=1)
    vertices2 = np.delete(box2.vertices, axis_idx, axis=1)

    # 获取 2D 矩形的外轮廓（凸包）
    poly1 = Polygon(vertices1).convex_hull
    poly2 = Polygon(vertices2).convex_hull

    # if not poly1.is_valid:
    #     poly1 = poly1.buffer(0)
    # if not poly2.is_valid:
    #     poly2 = poly2.buffer(0)

    # 计算交集/并集面积
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - intersection_area
    #print(intersection_area, union_area)

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou



def check_center_distance(pred_center_coor, gt_center_coor, pred_side_length, gt_side_length, threshold = 0.05):
    if abs(pred_center_coor - gt_center_coor) < threshold and pred_side_length < threshold and gt_side_length < threshold:
        return 1
    else:
        return 0


def compute_iou_auto(pred_box, gt_box, pred_center_size, gt_center_size, threshold = 0.05):
    """
    自动判断计算 3D 或 2D IoU
    :return: IoU 值
    """
    #threshold = 0.05  # 判定阈值

    # 确定最小的轴

    if gt_center_size[1][0] < threshold:
        #print("x")
        return  check_center_distance(pred_center_size[0][0], gt_center_size[0][0], pred_center_size[1][0], gt_center_size[1][0]) * compute_2d_iou(pred_box, gt_box, drop_axis = 'x')  # 计算 YZ 平面的 IoU
    elif gt_center_size[1][1] < threshold:
        #print("y")
        return check_center_distance(pred_center_size[0][1], gt_center_size[0][1], pred_center_size[1][1], gt_center_size[1][1]) * compute_2d_iou(pred_box, gt_box, drop_axis = 'y')  # 计算 XZ 平面的 IoU
    elif gt_center_size[1][2] < threshold:
        #print("z")
        return check_center_distance(pred_center_size[0][2], gt_center_size[0][2], pred_center_size[1][2], gt_center_size[1][2]) * compute_2d_iou(pred_box, gt_box, drop_axis = 'z')  # 计算 XY 平面的 IoU

    # 否则计算 3D IoU
    return compute_3d_iou(pred_box, gt_box)



def main(model, evaluation_level):

    if evaluation_level == "human_evaluation":
        different_level_valid_cnt = {"area": 0,
                                     "space": 0,
                                     "object": 0,
                                     "part": 0}
        IOU25 = {"area": 0,
                "space": 0,
                "object": 0,
                "part": 0}
        
        IOU50 = {"area": 0,
                "space": 0,
                "object": 0,
                "part": 0}
        
        IOU75 = {"area": 0,
                "space": 0,
                "object": 0,
                "part": 0}
        
        human_evaluation_information_path = "../anywhere3d_datasets/detailed_total_chosen_information.json"
        with open(human_evaluation_information_path, "r") as f:
            human_evaluation_info = json.load(f)

        
        for sample in human_evaluation_info:
            exp_level = sample["evaluation_level"].replace("_level", "")
            exp_file_index = sample["exp_file_index"]
            dataset = sample["datasetname"]
            scene_id = sample["scene_id"]

            path_gt = "../anywhere3d_datasets/anywhere3d_" + exp_level + "_level_aligned.xlsx"
            
            file_gt = pd.read_excel(path_gt, header = 0, index_col = 0)
            db_id = file_gt.iloc[exp_file_index]['_id']

             
            path_pred = f'./{model}/{exp_level + "_level"}/{dataset}-{scene_id}-{db_id}.json'

            # path_pred = f'./{model}/human_evaluation/{exp_level + "_level"}-{dataset}-{scene_id}-{db_id}.json'

            print(path_pred)

            with open(path_pred, "r") as f:
                file_pred = json.load(f)


            x_offset = y_offset = z_offset = 0

            gt_center_size = [[file_gt.iloc[exp_file_index]['box_x'] - x_offset, file_gt.iloc[exp_file_index]['box_y'] - y_offset, file_gt.iloc[exp_file_index]['box_z'] - z_offset],
                            [file_gt.iloc[exp_file_index]['box_width'], file_gt.iloc[exp_file_index]['box_length'], file_gt.iloc[exp_file_index]['box_height']]]
            gt_center_size = np.array(gt_center_size)
            
            gt_cube = create_rotated_bounding_box(gt_center_size[0], gt_center_size[1], file_gt.iloc[exp_file_index]['box_rot_angle'])

            pred_center_size = [[file_pred['pred_box_x'], file_pred['pred_box_y'], file_pred['pred_box_z']],
                                [file_pred['pred_box_width'], file_pred['pred_box_length'], file_pred['pred_box_height']]
                                ]
            pred_center_size = np.array(pred_center_size)

            
            different_level_valid_cnt[exp_level] += 1
            
            if not np.isnan(pred_center_size).any():
                
                pred_cube = create_rotated_bounding_box(pred_center_size[0], pred_center_size[1], 0)
                if exp_level == "area":
                    iou = compute_2d_iou(pred_cube, gt_cube, drop_axis = 'z')
                else:
                    iou = compute_iou_auto(pred_cube, gt_cube, pred_center_size, gt_center_size)
                
                file_pred["iou"] = iou
                if iou >= 0.25:
                    IOU25[exp_level] += 1
                if iou >= 0.5:
                    IOU50[exp_level] += 1
                if iou >= 0.75:
                    IOU75[exp_level] += 1
            else:
                file_pred["iou"] = 0
            
            # with open(path_pred, "w") as f:
            #     json.dump(file_pred, f, indent = 4)

            
                    
        print("valid prediction number")
        print(different_level_valid_cnt)
        print("Acc@0.25IoU")
        for gran_level in IOU25:
            print(gran_level, IOU25[gran_level] / different_level_valid_cnt[gran_level])
        print(sum([IOU25[gran_level] for gran_level in IOU25]) / 200)

        print("Acc@0.5IoU")
        for gran_level in IOU50:
            print(gran_level, IOU50[gran_level] / different_level_valid_cnt[gran_level])
        print(sum([IOU50[gran_level] for gran_level in IOU50]) / 200)

        print("Acc@0.75IoU")
        for gran_level in IOU75:
            print(gran_level, IOU75[gran_level] / different_level_valid_cnt[gran_level])
        print(sum([IOU75[gran_level] for gran_level in IOU75]) / 200)
            

    elif evaluation_level == "all":

        all_evaluation_lis = ["area_level", "space_level", "object_level", "part_level"]
        valid_cnt = 0
        IoU_25_cnt = 0
        IoU_50_cnt = 0
        IoU_75_cnt = 0
        x_offset = y_offset = z_offset = 0

        for eva_lvl in all_evaluation_lis:
            path_gt = f"../anywhere3d_datasets/anywhere3d_{eva_lvl}_aligned.xlsx"
            file_gt = pd.read_excel(path_gt, header = 0, index_col = 0)

            for index, row in file_gt.iterrows():
                print(index, row['datasetname'], row["scene_id"], row["_id"])
                path_pred = f'./{model}/{eva_lvl}/{row["datasetname"]}-{row["scene_id"]}-{row["_id"]}.json'
                with open(path_pred, "r") as f:
                    file_pred = json.load(f)
                
                valid_cnt += 1

                gt_center_size = [[row['box_x'] - x_offset, row['box_y'] - y_offset, row['box_z'] - z_offset],
                                [row['box_width'], row['box_length'], row['box_height']]
                                ]
                gt_center_size = np.array(gt_center_size)
                gt_cube = create_rotated_bounding_box(gt_center_size[0], gt_center_size[1], row['box_rot_angle'])

                pred_center_size = [[file_pred['pred_box_x'], file_pred['pred_box_y'], file_pred['pred_box_z']],
                                    [file_pred['pred_box_width'], file_pred['pred_box_length'], file_pred['pred_box_height']]
                                    ]
                pred_center_size = np.array(pred_center_size)
                
                if not np.isnan(pred_center_size).any():
                    
                    pred_cube = create_rotated_bounding_box(pred_center_size[0], pred_center_size[1], 0)

                    if eva_lvl == "area_level":
                        iou = compute_2d_iou(pred_cube, gt_cube, drop_axis = "z")
                        #iou = compute_iou_auto(pred_cube, gt_cube, gt_center_size[1])
                    else:
                        iou = compute_iou_auto(pred_cube, gt_cube, pred_center_size, gt_center_size)
                    if iou >= 0.25:
                        IoU_25_cnt += 1
                    if iou >= 0.5:
                        IoU_50_cnt += 1
                    if iou >= 0.75:
                        IoU_75_cnt += 1


        print("valid prediction number")
        print(valid_cnt) 
        print("Acc@0.25IoU")
        print(IoU_25_cnt / valid_cnt) 
        print("Acc@0.5IoU")
        print(IoU_50_cnt / valid_cnt) 
        print("Acc@0.75IoU")
        print(IoU_75_cnt / valid_cnt) 

    else:

        path_gt = "../anywhere3d_datasets/anywhere3d_" + evaluation_level + "_aligned.xlsx"
        file_gt = pd.read_excel(path_gt, header = 0, index_col = 0)


        valid_cnt = 0
        IoU_25_cnt = 0
        IoU_50_cnt = 0
        IoU_75_cnt = 0

        for index, row in file_gt.iterrows():
            #flag = 1
            print(index, row['datasetname'], row["scene_id"], row["_id"])


            path_pred = f'./{model}/{evaluation_level}/{row["datasetname"]}-{row["scene_id"]}-{row["_id"]}.json'
            with open(path_pred, "r") as f:
                file_pred = json.load(f)

            if row['datasetname'] == "scannet":
                x_offset = 0
                y_offset = 0
                z_offset = 0
            else:
                # continue
                # if row['datasetname'] == "multiscan" or row['datasetname'] == "3RScan":
                #     ply_path = os.path.join(os.path.join('/home/wangtianxu/Viewer', row['datasetname'], 'scans', row['scene_id']), row['scene_id'] + '_vh_clean_2.ply')
                # elif row['datasetname'] == "arkitscene_valid":
                #     ply_path = os.path.join(os.path.join('/home/wangtianxu/Viewer/ARKitScene/validation/scans', row['scene_id']), row['scene_id'] + '_vh_clean_2.ply')    
                
                # ply_point_cloud = o3d.io.read_point_cloud(ply_path)
                # ply_points = np.asarray(ply_point_cloud.points)
                # mean_ply = np.mean(ply_points, axis = 0)
                # zmin_ply = np.min(ply_points[:, 2])
                # x_offset = round(mean_ply[0], 2)
                # y_offset = round(mean_ply[1], 2)
                # z_offset = round(zmin_ply, 2) 
                x_offset = y_offset = z_offset = 0

            gt_center_size = [[row['box_x'] - x_offset, row['box_y'] - y_offset, row['box_z'] - z_offset],
                            [row['box_width'], row['box_length'], row['box_height']]
                            ]
            gt_center_size = np.array(gt_center_size)
            gt_cube = create_rotated_bounding_box(gt_center_size[0], gt_center_size[1], row['box_rot_angle'])
            
            # print(file_pred['pred_box_x'], file_pred['pred_box_y'], file_pred['pred_box_z'], file_pred['pred_box_width'], file_pred['pred_box_length'], file_pred['pred_box_height'])
            pred_center_size = [[file_pred['pred_box_x'], file_pred['pred_box_y'], file_pred['pred_box_z']],
                                [file_pred['pred_box_width'], file_pred['pred_box_length'], file_pred['pred_box_height']]
                                ]

            pred_center_size = np.array(pred_center_size)
            
            # if np.isnan(pred_center_size).any():
            #     flag = 0

            #print(index)
            valid_cnt += 1
            if not np.isnan(pred_center_size).any():
                
                pred_cube = create_rotated_bounding_box(pred_center_size[0], pred_center_size[1], 0)

                if evaluation_level == "area_level":
                    iou = compute_2d_iou(pred_cube, gt_cube, drop_axis = "z")
                    #iou = compute_iou_auto(pred_cube, gt_cube, gt_center_size[1])
                else:
                    iou = compute_iou_auto(pred_cube, gt_cube, pred_center_size, gt_center_size)
                if iou >= 0.25:
                    IoU_25_cnt += 1
                if iou >= 0.5:
                    IoU_50_cnt += 1
                if iou >= 0.75:
                    IoU_75_cnt += 1

        print("valid prediction number")
        print(valid_cnt) 
        print("Acc@0.25IoU")
        print(IoU_25_cnt / valid_cnt) 
        print("Acc@0.5IoU")
        print(IoU_50_cnt / valid_cnt) 
        print("Acc@0.75IoU")
        print(IoU_75_cnt / valid_cnt) 


if __name__ == "__main__":
    main(model = "gpt-4.1-2025-04-14", evaluation_level = "object_level")
   


