import json
import os
import pickle
import time
import datetime
import re
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from concurrent.futures import ProcessPoolExecutor
import ast
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import base64
import math



class SceneInfo:
    def __init__(self, scene_type, scene_id, scene_name):
        assert scene_type in ["scannet", "multiscan", "3RScan", "arkitscene_valid"]
        self.scene_type = scene_type
        self.scene_id = scene_id
        self.scene_name = scene_name
        self.scene_graph_file = ""

        if self.scene_type == "scannet":
            # self.scene_graph_file = f"/home/wangtianxu/SceneVerse_Dataset/scene_graphs_w_obj_cap_anywhere3D/ScanNet/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_pred_labels_structural_object_labels_wo_captions.json"
            self.scene_graph_file = f"../scene_graphs_anywhere3D/ScanNet/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_wo_labels_wo_captions.json"
            self.gpt4scene_img_dir = f"../{self.scene_type}_gpt4scene_data/{self.scene_id}"
            self.caption_dir = f"../qwen_captions/scannet/{self.scene_id}"

        elif self.scene_type == "multiscan":
            # self.scene_graph_file = f"/home/wangtianxu/SceneVerse_Dataset/scene_graphs_w_obj_cap_anywhere3D/MultiScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_pred_labels_structural_object_labels_wo_captions.json"
            self.scene_graph_file = f"../scene_graphs_anywhere3D/MultiScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_wo_labels_wo_captions.json"
            self.gpt4scene_img_dir = f"../{self.scene_type}_gpt4scene_data/{self.scene_id}"
            self.caption_dir = f"../qwen_captions/multiscan/{self.scene_name}"
        
        elif self.scene_type == "arkitscene_valid":
            # self.scene_graph_file = f"/home/wangtianxu/SceneVerse_Dataset/scene_graphs_w_obj_cap_anywhere3D/ARKitScenes_validation/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_pred_labels_structural_object_labels_wo_captions.json"
            self.scene_graph_file = f"../scene_graphs_anywhere3D/ARKitScenes_validation/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_wo_labels_wo_captions.json"
            self.gpt4scene_img_dir = f"../arkitscene_gpt4scene_data/{self.scene_name}"
            self.caption_dir = f"../qwen_captions/arkitscene_valid/{self.scene_name}"

        elif self.scene_type == "3RScan":
            # self.scene_graph_file = f"/home/wangtianxu/SceneVerse_Dataset/scene_graphs_w_obj_cap_anywhere3D/3RScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_pred_labels_structural_object_labels_wo_captions.json"
            self.scene_graph_file = f"../scene_graphs_anywhere3D/3RScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_wo_labels_wo_captions.json"
            self.gpt4scene_img_dir = f"../{self.scene_type}_gpt4scene_data/{self.scene_name}"
            self.caption_dir = f"../qwen_captions/3RScan/{self.scene_name}"


        with open(self.scene_graph_file, "r") as f:
            self.scene_graph = json.load(f)

    
    def get_object_qwen_caption(self, object_id):
        if os.path.exists(os.path.join(self.caption_dir, str(object_id))):
            with open(os.path.join(self.caption_dir, str(object_id), "caption.json"), "r") as f:
                object_qwen_caption_file = json.load(f)
            if "caption" in object_qwen_caption_file:
                return object_qwen_caption_file["caption"]
            else:
                return "None"

        else:
            return "None"   

    
    def get_gpt4scene_imgs(self):

        self.bev_with_marker_path = f"{self.gpt4scene_img_dir}/bev.png"
        if os.path.exists(self.bev_with_marker_path):
            with open(self.bev_with_marker_path, "rb") as bev_file:
                bev = bev_file.read()
            self.bev_img_base64 = base64.b64encode(bev).decode('utf-8')
        else:
            print("bev path not exists")
        
        self.frame_imgs_base64 = []
        all_img_list = os.listdir(self.gpt4scene_img_dir)
        for f_img in all_img_list:
            if "bev" not in f_img and "concat" not in f_img and f_img.endswith(".png"):
                #print(os.path.join(self.gpt4scene_img_dir, f_img))
                with open(os.path.join(self.gpt4scene_img_dir, f_img), "rb") as single_img_file:
                    single_img_data = single_img_file.read()
                self.frame_imgs_base64.append(base64.b64encode(single_img_data).decode('utf-8'))
        #print("len frame imgs", len(self.frame_imgs_base64))
        
        return self.bev_img_base64, self.frame_imgs_base64



class VLMBaseline():
    def __init__(self, model):
        self.model = model
        if "gpt" in self.model:
            self.API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
            self.REGION = "eastus2"
            self.model_client = AzureOpenAI(
                api_key = os.environ.get("OPENAI_API_KEY_ANYWHERE3D"),
                api_version = "2025-03-01-preview",
                azure_endpoint = f"{self.API_BASE}/{self.REGION}"
            )
        
        elif "o4-mini" in self.model:
            self.API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
            self.REGION = "eastus2"
            self.model_client = AzureOpenAI(
                api_key = os.environ.get("OPENAI_API_KEY_ANYWHERE3D"),
                api_version = "2024-12-01-preview",
                azure_endpoint = f"{self.API_BASE}/{self.REGION}"
            )

        elif "qwen" in self.model:
            self.model_client = OpenAI(
                api_key = os.getenv("ALI_API_KEY"),
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif "intern" in self.model:
            self.model_client = OpenAI(
                api_key = os.getenv("internvl_API_KEY"),
                base_url = "https://chat.intern-ai.org.cn/api/v1/"
            )

    def get_VLM_response(self, messages, temperature = 1, max_tokens = 16384):
        
        patience = 3
        if "o4-mini" not in self.model:
            while patience > 0:
                patience -= 1
                try:
                        response = self.model_client.chat.completions.create(model = self.model,
                                                                messages = messages,
                                                                max_tokens = max_tokens
                                                                )
                        
                        prediction = response.choices[0].message.content.strip()
                        if prediction != "" and prediction != None:
                            return prediction
                except Exception as e:
                    print(e)
                    time.sleep(10)
            return "None"
        else:
            while patience > 0:
                patience -= 1
                try:
                        response = self.model_client.chat.completions.create(model = self.model,
                                                                messages = messages,
                                                                max_completion_tokens = max_tokens,
                                                                # reasoning = {
                                                                #     "effort": "high",
                                                                #     "summary": "auto"
                                                                # }
                                                                reasoning_effort = "high",
                                                                # summary = "auto"
                                                                # temperature = 1
                                                                )
                        # print("response.choices[0]\n", response.choices[0])
                        # print("response.output_text \n", response.output_text)
                        prediction = response.choices[0].message.content.strip()
                        if prediction != "" and prediction != None:
                            print("prediction", prediction)
                            return prediction
                except Exception as e:
                    print(e)
                    time.sleep(5)
            return "None"

def test_a_question(evaluation_level, datasetname, scene_id, scene_name, db_id, referring_expressions, model):


    SYSTEM_PROMPT_PATH = "./system_prompt_for_" + evaluation_level + ".txt"
    with open(SYSTEM_PROMPT_PATH, 'r') as file:
        system_prompt = file.read()
    
    # For no label setting
    # system_prompt = system_prompt.replace("Note that the object category is predicted by a 3D vision-language model and may not be entirely accurate. ", "")

    scene_info = SceneInfo(scene_type = datasetname, scene_id = scene_id, scene_name = scene_name)
    scene_graph_data = scene_info.scene_graph["object_info"]

    for obj in scene_graph_data:
        obj_label, obj_id = obj.split("-")
        #print(scene_info.get_object_qwen_caption(obj_id))
        scene_graph_data[obj]["caption"] = scene_info.get_object_qwen_caption(obj_id)

    # print(scene_graph_data)

    bev_img_base64, frame_imgs_base64 = scene_info.get_gpt4scene_imgs()

    user_messages_content = [
                                {
                                    "type": "text", 
                                    "text": "Here is the scene graph: " + str(scene_graph_data) + "\n" + "Here is the referring expressions: " +  referring_expressions + "\nThe subsequent images include a Bird Eye View image as the first, followed by 8 frames extracted from the scene video. Please return the center coordinates and sizes of predicted 3D bounding box STRICTLY following the instructed format."},
                                {
                                    "type": "image_url", 
                                    "image_url":{
                                                "url": f"data:image/png;base64,{bev_img_base64}",
                                                "detail": "high",
                                    }
                                }
                            ]
    for f_img in frame_imgs_base64:
        user_messages_content.append({
                                        "type": "image_url", 
                                        "image_url": {
                                                    "url": f"data:image/png;base64,{f_img}",
                                                    "detail": "high",
                                        }
                                    })

    messages = [    
                {
                    "role": "system", 
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ]
                },
                {
                    "role": "user", 
                    "content": user_messages_content
                }
    ]
    #print("Here is the scene graph: " + str(scene_graph_data) + "\n" + "Here is the referring expressions: " +  referring_expressions + "\nThe subsequent images include a Bird Eye View image as the first, followed by 8 frames extracted from the scene video. Please return the center coordinates and sizes of predicted 3D bounding box STRICTLY following the instructed format.")

    VLM_model = VLMBaseline(model)
    response = VLM_model.get_VLM_response(messages)
    result_dic = {
                "evaluation_level": evaluation_level,
                "_id": db_id,
                "datasetname": datasetname, 
                "scene_id": scene_id,
                "scene_name": scene_name,
                "referring_expressions": referring_expressions,
                "pred_bbx_reasoning": response,
                "pred_box_x": 0,
                "pred_box_y": 0,
                "pred_box_z": 0,
                "pred_box_width": 0,
                "pred_box_length": 0,
                "pred_box_height": 0
            }
    

    #For human_evaluation:
    # save_path = f"./{model}/human_evaluation/{evaluation_level}-{datasetname}-{scene_id}-{db_id}.json"
    # print(save_path)
    # with open(save_path, "w") as f:
    #     json.dump(result_dic, f, indent = 4)

    #For single level evaluation

    save_path = f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json"
    print(save_path)
    with open(save_path, "w") as f:
        json.dump(result_dic, f, indent = 4)

def main(model, evaluation_level):

    save_dir = f"./{model}/{evaluation_level}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if "human_evaluation" in evaluation_level:
        save_dir = f"./{model}/{evaluation_level}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gt_file_path = "../anywhere3d_datasets/detailed_total_chosen_information.json"
        with open(gt_file_path, "r") as f:
            gt_file = json.load(f)
        
        questions = []
        for sample in gt_file:
            datasetname = sample["datasetname"]
            scene_id = sample["scene_id"]
            scene_name = sample["scene_name"]
            db_id = sample["db_id"]
            eva_lvl = sample["evaluation_level"]
            # print(datasetname, scene_id, db_id)
            if not os.path.exists(f"./{model}/{evaluation_level}/{eva_lvl}-{datasetname}-{scene_id}-{db_id}.json"):
                # print("yes")
                questions.append((eva_lvl, datasetname, scene_id, scene_name, db_id, sample["new_referring_expressions"], model))
            else:
                with open(f"./{model}/{evaluation_level}/{eva_lvl}-{datasetname}-{scene_id}-{db_id}.json", "r") as f:
                    prediction_file = json.load(f)
                    if prediction_file["pred_bbx_reasoning"] == "None" or prediction_file["pred_bbx_reasoning"] == "" or prediction_file["pred_bbx_reasoning"] is None:
                        questions.append((eva_lvl, datasetname, scene_id, scene_name, db_id, sample["new_referring_expressions"], model))
        
        num_workers = 8
        print(len(questions))

        with Pool(num_workers) as pool:
            pool.starmap(test_a_question, questions)

    else:
        gt_dataset_path = "../anywhere3d_datasets/anywhere3d_" + evaluation_level + ".xlsx"
        file_gt = pd.read_excel(gt_dataset_path, header = 0, index_col = 0)
        
        questions = []

        for index, row in file_gt.iterrows():
            print(index)
            if row["datasetname"] == "scannet":
                scene_name = row["scene_id"]
            elif row["datasetname"] == "multiscan":
                scene_name = "scene_0" + row["scene_id"].split("_")[0][5:] + "_00"
            elif row["datasetname"] == "3RScan":
                scan_name_id_path = "/home/wangtianxu/Viewer/3RScan/scan_name_id.pickle"
                with open(scan_name_id_path, "rb") as f:
                    scan_name_id_lis = pickle.load(f)
                scan_name_lis = [ele[0] for ele in scan_name_id_lis]
                scan_id_lis = [ele[1] for ele in scan_name_id_lis]
                scan_index = scan_id_lis.index(row["scene_id"])
                scene_name = scan_name_lis[scan_index]
            elif row["datasetname"] == "arkitscene_valid":
                scan_name_id_path = "/home/wangtianxu/Viewer/ARKitScene/validation/scan_name_id.pickle"
                with open(scan_name_id_path, "rb") as f:
                    scan_name_id_lis = pickle.load(f)
                scan_name_lis = [ele[0] for ele in scan_name_id_lis]
                scan_id_lis = [ele[1] for ele in scan_name_id_lis]
                scan_index = scan_id_lis.index(row["scene_id"])
                scene_name = scan_name_lis[scan_index]

            #print(row["datasetname"], row["scene_id"], scene_name)
            datasetname = row["datasetname"]
            scene_id = row["scene_id"]
            db_id = row["_id"]

            if not os.path.exists(f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json"):
                questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, row["new_referring_expressions"], model))
            else:
                with open(f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json", "r") as f:
                    prediction_file = json.load(f)
                    if prediction_file["pred_bbx_reasoning"] == "None" or prediction_file["pred_bbx_reasoning"] == "" or prediction_file["pred_bbx_reasoning"] is None:
                        questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, row["new_referring_expressions"], model))
                    if math.isnan(prediction_file["pred_box_x"]) or math.isnan(prediction_file["pred_box_y"]) or math.isnan(prediction_file["pred_box_z"]) \
                        or math.isnan(prediction_file["pred_box_width"]) or math.isnan(prediction_file["pred_box_length"]) or math.isnan(prediction_file["pred_box_height"]):
                        questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, row["new_referring_expressions"], model))
                    # if prediction_file["pred_box_width"] == 0 or prediction_file["pred_box_length"] == 0 or prediction_file["pred_box_height"] == 0:
                    #     questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, row["new_referring_expressions"], model))
                    
        num_workers = 8
        print(len(questions))

        with Pool(num_workers) as pool:
            pool.starmap(test_a_question, questions)
            
           

if __name__ == "__main__":
    main("gpt-4.1-2025-04-14", "object_level")

#For Qwen-vl-2.5: default temperature = 0.01, max tokens = 8192
#For GPT4o, default temperature = 1.0, max_tokens = 16384
#For o4-mini, default temperature don't know, use "max_completion_tokens" instead of max_tokens
#For internlm3-latest, 