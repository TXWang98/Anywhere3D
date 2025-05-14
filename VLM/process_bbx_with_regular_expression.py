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



#Prediction processing, using regular expressions to extract bounding box center x, y, z coordinates 
#as well as bounding box len x, y, z


def extract_coor_length(file_path, text):
    position_pattern = r"xcoordinate:\s*([-+]?\d*\.\d+|\d+),\s*ycoordinate:\s*([-+]?\d*\.\d+|\d+),\s*zcoordinate:\s*([-+]?\d*\.\d+|\d+)"
    size_pattern = r"xlength:\s*([-+]?\d*\.\d+|\d+),\s*ylength:\s*([-+]?\d*\.\d+|\d+),\s*zlength:\s*([-+]?\d*\.\d+|\d+)"

    position_pattern2 = r"\"xcoordinate\":\s*([-+]?\d*\.\d+|\d+),\s*\"ycoordinate\":\s*([-+]?\d*\.\d+|\d+),\s*\"zcoordinate\":\s*([-+]?\d*\.\d+|\d+)"
    size_pattern2 = r"\"xlength\":\s*([-+]?\d*\.\d+|\d+),\s*\"ylength\":\s*([-+]?\d*\.\d+|\d+),\s*\"zlength\":\s*([-+]?\d*\.\d+|\d+)"


        
    position_matches = list(re.finditer(position_pattern, text))
    size_matches = list(re.finditer(size_pattern, text))

    position_matches2 = list(re.finditer(position_pattern2, text))
    size_matches2 = list(re.finditer(size_pattern2, text))

    if position_matches:

        position_last_match = position_matches[-1]
        
        x_coor = float(position_last_match.group(1))
        y_coor = float(position_last_match.group(2))
        z_coor = float(position_last_match.group(3))
            
    else:
        if position_matches2:
            position_last_match2 = position_matches2[-1]
            x_coor = float(position_last_match2.group(1))
            y_coor = float(position_last_match2.group(2))
            z_coor = float(position_last_match2.group(3))
        else:
            print("----------------------------------------------------------------------------------------------------------------")
            print(file_path, "position np.nan")
            print("----------------------------------------------------------------------------------------------------------------")
            print(text)
            x_coor = y_coor = z_coor = np.nan
        
    if size_matches:
        size_last_match = size_matches[-1]
        
        x_len = float(size_last_match.group(1))
        y_len = float(size_last_match.group(2))
        z_len = float(size_last_match.group(3))
    else:
        if size_matches2:
            size_last_match2 = size_matches2[-1]
            x_len = float(size_last_match2.group(1))
            y_len = float(size_last_match2.group(2))
            z_len = float(size_last_match2.group(3))
        else:
            print("----------------------------------------------------------------------------------------------------------------")
            print(file_path, "size np.nan")
            print("----------------------------------------------------------------------------------------------------------------")
            print(text)
            x_len = y_len = z_len = np.nan
    
    return x_coor, y_coor, z_coor, x_len, y_len, z_len


def main(model, evaluation_level):


    pred_file_dir = f"./{model}/{evaluation_level}"
    all_pred_file = sorted(os.listdir(pred_file_dir))
    for ele, pred_file in enumerate(all_pred_file):
        with open(os.path.join(pred_file_dir, pred_file), "r") as f:
            pred_data = json.load(f)

        x_coor, y_coor, z_coor, x_len, y_len, z_len = extract_coor_length(os.path.join(pred_file_dir, pred_file), pred_data["pred_bbx_reasoning"])
        
        pred_data["pred_box_x"] = x_coor
        pred_data["pred_box_y"] = y_coor
        pred_data["pred_box_z"] = z_coor
        pred_data["pred_box_width"] = x_len
        pred_data["pred_box_length"] = y_len
        pred_data["pred_box_height"] = z_len
        with open(os.path.join(pred_file_dir, pred_file), "w") as f:
            json.dump(pred_data, f, indent = 4)






if __name__ == "__main__":
    main(model = "gpt-4.1-2025-04-14", evaluation_level = "object_level")
