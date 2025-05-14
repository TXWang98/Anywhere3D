import json
import pandas as pd
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
import math


class LLMBaseline():
    def __init__(self, model):
        self.model = model
       
        self.model_client = OpenAI(
            api_key = os.getenv("ALI_API_KEY"),
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    

    def get_LLM_response(self, messages, temperature = 1, max_tokens = 8192):
        patience = 3
        while patience > 0:
            patience -= 1
            try:
                response = self.model_client.chat.completions.create(model = self.model,
                                                        messages = messages,
                                                        max_tokens = 8192,
                                                        )
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            except Exception as e:
                print(e)
                time.sleep(5)
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]



def process_single_file(file_path):
    print(file_path)

    system_prompt = "Given the following reasoning content, your task is to extract the final 3D bounding box's center coordinates and dimensions based on the whole content. Specifically, you should return the x, y, and z coordinates of the 3D bounding box center and the lengths along the x, y, z axes. Format your answer as a list of six elements, where each element is either a float or string 'none' if the information is missing. The expected format is: [x_coordinate, y_coordinate, z_coordinate, x_length, y_length, z_length]. For example: [1.0, -0.28, 0.65, 2.43, 1.2, 0.7]. or, if some values are missing: [1.0, -0.28, 'none', 2.43, 'none', 0.7]. Only return the list, without any additional explanations or text. "

    with open(file_path, "r") as f:
        pred_content = json.load(f)
    
    # print(pred_content["pred_bbx_answer"])

    #For R1 and Qwen3
    messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Here is the reasoning content:\n" + pred_content["pred_bbx_answer"]}]
    
    #For GPT4o and Qwen2.5
    # messages = [{"role": "system", "content": system_prompt},
    #             {"role": "user", "content": "Here is the reasoning content:\n" + pred_content["pred_bbx_reasoning"]}]

    
    LLM_model = LLMBaseline("qwen2.5-72b-instruct")
    try:
        response = LLM_model.get_LLM_response(messages, max_tokens = 1000)
        print(response)
        response = ast.literal_eval(response)
        pred_content["pred_box_x"] = response[0] if response[0] != "none" else np.nan
        pred_content["pred_box_y"] = response[1] if response[1] != "none" else np.nan
        pred_content["pred_box_z"] = response[2] if response[2] != "none" else np.nan
        pred_content["pred_box_width"] = response[3] if response[3] != "none" else np.nan
        pred_content["pred_box_length"] = response[4] if response[4] != "none" else np.nan
        pred_content["pred_box_height"] = response[5] if response[5] != "none" else np.nan

        with open(file_path, "w") as f:
            json.dump(pred_content, f, indent = 4)

    
    except Exception as e:
        print(e)
    


def main(folder_dir):

    target_file_lis = []

    all_file_name = os.listdir(folder_dir)
    for file_name in all_file_name:
        with open(os.path.join(folder_dir, file_name), "r") as f:
            prediction_file = json.load(f, parse_constant = lambda _: float('nan'))
            if math.isnan(prediction_file["pred_box_x"]) or math.isnan(prediction_file["pred_box_y"]) or math.isnan(prediction_file["pred_box_z"]) \
            or math.isnan(prediction_file["pred_box_width"]) or math.isnan(prediction_file["pred_box_length"]) or math.isnan(prediction_file["pred_box_height"]):
                target_file_lis.append(os.path.join(folder_dir, file_name))
    
    print(len(target_file_lis))

    for target_file in target_file_lis:
        process_single_file(target_file)

            

if __name__ == "__main__":
    main(folder_dir = "./deepseek-r1-671b/object_level")