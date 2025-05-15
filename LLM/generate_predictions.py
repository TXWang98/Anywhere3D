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
            self.caption_dir = f"../qwen_captions/scannet/{self.scene_id}"
        
        elif self.scene_type == "multiscan":
            # self.scene_graph_file = f"/home/wangtianxu/SceneVerse_Dataset/scene_graphs_w_obj_cap_anywhere3D/MultiScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_pred_labels_structural_object_labels_wo_captions.json"
            self.scene_graph_file = f"../scene_graphs_anywhere3D/MultiScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_wo_labels_wo_captions.json"
            self.caption_dir = f"../qwen_captions/multiscan/{self.scene_name}"

        elif self.scene_type == "arkitscene_valid":
            # self.scene_graph_file = f"/home/wangtianxu/SceneVerse_Dataset/scene_graphs_w_obj_cap_anywhere3D/ARKitScenes_validation/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_pred_labels_structural_object_labels_wo_captions.json"
            self.scene_graph_file = f"../scene_graphs_anywhere3D/ARKitScenes_validation/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_wo_labels_wo_captions.json"
            self.caption_dir = f"../qwen_captions/arkitscene_valid/{self.scene_name}"

        elif self.scene_type == "3RScan":
            # self.scene_graph_file = f"/home/wangtianxu/SceneVerse_Dataset/scene_graphs_w_obj_cap_anywhere3D/3RScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_pred_labels_structural_object_labels_wo_captions.json"
            self.scene_graph_file = f"../scene_graphs_anywhere3D/3RScan/{self.scene_id}/scene_graphs_for_gpt_anywhere3D_wo_relations_wo_labels_wo_captions.json"       
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

        



class LLMBaseline():
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
        elif "deepseek" in self.model:
            self.model = "deepseek-r1-250120"
            self.model_client = OpenAI(
                api_key = os.getenv("HUOSHAN_API_KEY"),
                base_url = "https://ark.cn-beijing.volces.com/api/v3",
            )

        elif "qwen" in self.model:
            self.model_client = OpenAI(
                api_key = os.getenv("ALI_API_KEY"),
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif "o4" in self.model:
            self.API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
            self.REGION = "eastus2"
            self.model_client = AzureOpenAI(
                api_key = os.environ.get("OPENAI_API_KEY_ANYWHERE3D"),
                api_version = "2025-03-01-preview",
                azure_endpoint = f"{self.API_BASE}/{self.REGION}"
            )


    def get_LLM_response(self, messages, temperature = 1, max_tokens = 16384):
        if "deepseek" in self.model:
            patience = 3
            while patience > 0:
                patience -= 1
                try:
                    response = self.model_client.chat.completions.create(model = self.model,
                                                                messages = messages,
                                                                max_tokens = max_tokens,
                                                                stream = True
                                                                )
                    reasoning_content = ""
                    prediction = ""
                    for chunk in response:
                        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            reasoning_content += chunk.choices[0].delta.reasoning_content
                        else:
                            prediction += chunk.choices[0].delta.content
                           
                    
                    if prediction != "" and prediction != None:
                        return reasoning_content, prediction
                except Exception as e:
                    print(e)
                    time.sleep(3)
            return "None", "None"

        elif "o4-mini" in self.model:
            patience = 3
            while patience > 0:
                patience -= 1
                try:
                        response = self.model_client.chat.completions.create(model = self.model,
                                                                messages = messages,
                                                                max_completion_tokens = 16384,
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
                            return prediction
                except Exception as e:
                    print(e)
                    time.sleep(5)
            return "None"

        else:
            patience = 3
            while patience > 0:
                patience -= 1
                try:
                    response = self.model_client.chat.completions.create(model = self.model,
                                                            messages = messages
                                                            )
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction != None:
                        return prediction
                except Exception as e:
                    print(e)
                    time.sleep(10)
            return "None"

def test_a_question(evaluation_level, datasetname, scene_id, scene_name, db_id, referring_expressions, model):

    print(datasetname, scene_id, scene_name)
    SYSTEM_PROMPT_PATH = "./system_prompt_for_" + evaluation_level + ".txt"
    with open(SYSTEM_PROMPT_PATH, 'r') as file:
        system_prompt = file.read()

    scene_info = SceneInfo(scene_type = datasetname, scene_id = scene_id, scene_name = scene_name)
    scene_graph_data = scene_info.scene_graph["object_info"]
    
    #Adding qwen caption to the scene graph
    for obj in scene_graph_data:
        obj_label, obj_id = obj.split("-")
        scene_graph_data[obj]["caption"] = scene_info.get_object_qwen_caption(obj_id)
    


    user_prompt = json.dumps({"object_info": scene_graph_data, "referring_expressions": referring_expressions})

    if "deepseek" in model:
        messages = [{"role": "user", "content": system_prompt + "\n" + user_prompt}]
        LLM_model = LLMBaseline(model)
        reasoning_content, answer = LLM_model.get_LLM_response(messages, max_tokens = 16384)
        result_dic = {
                    "evaluation_level": evaluation_level,
                    "_id": db_id,
                    "datasetname": datasetname, 
                    "scene_id": scene_id,
                    "scene_name": scene_name,
                    "referring_expressions": referring_expressions,
                    "pred_bbx_reasoning": reasoning_content,
                    "pred_bbx_answer": answer,
                    "pred_box_x": 0,
                    "pred_box_y": 0,
                    "pred_box_z": 0,
                    "pred_box_width": 0,
                    "pred_box_length": 0,
                    "pred_box_height": 0
                }

        # save_path = f"./{model}/human_evaluation/{evaluation_level}-{datasetname}-{scene_id}-{db_id}.json"
        # with open(save_path, "w") as f:
        #     json.dump(result_dic, f, indent = 4)

        save_path = f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json"
        with open(save_path, "w") as f:
            json.dump(result_dic, f, indent = 4)

    else:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        LLM_model = LLMBaseline(model)
        response = LLM_model.get_LLM_response(messages)
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
        
        #Generation on Huam evaluation subset

        # save_path = f"./{model}/human_evaluation/{evaluation_level}-{datasetname}-{scene_id}-{db_id}.json"
        # with open(save_path, "w") as f:
        #     json.dump(result_dic, f, indent = 4)

        # Generation on single level
            
        save_path = f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json"
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
        
        num_workers = 10
        print(len(questions))

        with Pool(num_workers) as pool:
            pool.starmap(test_a_question, questions)
    
    else:

        # gt_dataset_path = "../anywhere3d_datasets/anywhere3d_" + evaluation_level + "_aligned.xlsx"
        # file_gt = pd.read_excel(gt_dataset_path, header = 0, index_col = 0)

        gt_dataset_path = "../anywhere3d_datasets/eval_anywhere3D_aligned_total.json"
        with open(gt_dataset_path, "r") as f:
            file_gt = json.load(f)
        
        questions = []

        for sample in file_gt:
            if sample["grounding_level"] == evaluation_level:

                datasetname = sample["datasetname"]
                scene_id = sample["scene_id"]
                scene_name = sample["scene_name"]
                db_id = sample["db_id"]

                if not os.path.exists(f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json"):
                    questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, sample["referring_expressions"], model))
                else:
                    with open(f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json", "r") as f:
                        prediction_file = json.load(f)
                        if prediction_file["pred_bbx_reasoning"] == "None" or prediction_file["pred_bbx_reasoning"] == "":
                            questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, sample["referring_expressions"], model))

        
        # for index, row in file_gt.iterrows():
        #     print(index)
        #     if row["datasetname"] == "scannet":
        #         scene_name = row["scene_id"]
        #     elif row["datasetname"] == "multiscan":
        #         scene_name = "scene_0" + row["scene_id"].split("_")[0][5:] + "_00"
        #     elif row["datasetname"] == "3RScan":
        #         scan_name_id_path = "/home/wangtianxu/Viewer/3RScan/scan_name_id.pickle"
        #         with open(scan_name_id_path, "rb") as f:
        #             scan_name_id_lis = pickle.load(f)
        #         scan_name_lis = [ele[0] for ele in scan_name_id_lis]
        #         scan_id_lis = [ele[1] for ele in scan_name_id_lis]
        #         scan_index = scan_id_lis.index(row["scene_id"])
        #         scene_name = scan_name_lis[scan_index]
        #     elif row["datasetname"] == "arkitscene_valid":
        #         scan_name_id_path = "/home/wangtianxu/Viewer/ARKitScene/validation/scan_name_id.pickle"
        #         with open(scan_name_id_path, "rb") as f:
        #             scan_name_id_lis = pickle.load(f)
        #         scan_name_lis = [ele[0] for ele in scan_name_id_lis]
        #         scan_id_lis = [ele[1] for ele in scan_name_id_lis]
        #         scan_index = scan_id_lis.index(row["scene_id"])
        #         scene_name = scan_name_lis[scan_index]

        #     #print(row["datasetname"], row["scene_id"], scene_name)
        #     datasetname = row["datasetname"]
        #     scene_id = row["scene_id"]
        #     db_id = row["_id"]

        #     if not os.path.exists(f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json"):
        #         questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, row["new_referring_expressions"], model))
        #     else:
        #         with open(f"./{model}/{evaluation_level}/{datasetname}-{scene_id}-{db_id}.json", "r") as f:
        #             prediction_file = json.load(f)
        #             if prediction_file["pred_bbx_reasoning"] == "None" or prediction_file["pred_bbx_reasoning"] == "":
        #                 questions.append((evaluation_level, datasetname, scene_id, scene_name, db_id, row["new_referring_expressions"], model))

        
        num_workers = 10
        print(len(questions))

        with Pool(num_workers) as pool:
            pool.starmap(test_a_question, questions)
            
           

if __name__ == "__main__":
    main("qwen2.5-72b-instruct", "area_level")
