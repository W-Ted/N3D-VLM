import yaml
from omegaconf import OmegaConf
from pathlib import Path
import torch
import sys
sys.path.append('..')
sys.path.append('.')
import gc
import json
from typing import Optional
import fire
from tqdm import tqdm, trange
import os
import numpy as np
from threading import Thread
import rerun
import uuid
from scipy.spatial.transform import Rotation as R
import copy
import re
from PIL import Image, ImageDraw, ImageFont
import cv2
import random
import math
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from transformers import TextIteratorStreamer
from transformers.image_processing_utils import BaseImageProcessor
from transformers import modeling_utils 

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

from src.llamafactory.hparams.parser import _parse_train_args
from src.llamafactory.data import get_template_and_fix_tokenizer
from src.llamafactory.data.processor.processor_utils import infer_seqlen
from src.llamafactory.extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER

from qwen2_5_vl.modeling_qwen2_5_vl_pe import Qwen2_5_VLForConditionalGeneration_pe

from pcd import parse_bbox_dict_xy
from pcd import parse_bbox_dict_uv, serialize_bboxes_uv

current_dir = os.path.dirname(os.path.abspath(__file__))
moge_path = os.path.join(current_dir, 'third_party', 'MoGe')
if moge_path not in sys.path:
    sys.path.append(moge_path)
from moge.model.v2 import MoGeModel

# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def vis_results_in_rrd(points_list, gt_bboxes_list, pred_bboxes_list, save_path):
    rerun.init("Data Visualization", spawn=False, recording_id=uuid.uuid4())

    T = len(points_list)
    assert len(gt_bboxes_list) == T
    entity_name = 'video'

    for i in range(T):
        rerun.set_time_sequence("frameid", i)

        points = points_list[i]
        gt_bboxes = gt_bboxes_list[i]

        if isinstance(points, torch.Tensor):
            points = points.numpy()
        
        rerun.log(f"/{entity_name}/world/pts_original/",
                      rerun.Points3D(points[:,:3],
                                     colors= (points[:,-3:]* 255).astype(np.uint8) 
                                     ))
        
        for n, bbox in enumerate(gt_bboxes):
            center = np.asarray([bbox.position_x, bbox.position_y, bbox.position_z])
            half = np.asarray([bbox.scale_x/2.0, bbox.scale_y/2.0, bbox.scale_z/2.0])
            rotation = R.from_rotvec([0, bbox.angle_z, 0]).as_matrix()
            safe_class_name = "_".join(bbox.class_name.split())
            name = f"object{bbox.id}_{safe_class_name}"
            rerun.log(f"/{entity_name}/world/bboxes_groundtruth/{name}",
                        rerun.Boxes3D(centers=center[None],
                                        half_sizes=half[None],
                                        colors=np.array([0,255,0]).astype(np.uint8)[None],
                                        labels=[name]),
                        rerun.InstancePoses3D(mat3x3=rotation), 
                    )

        if pred_bboxes_list is not None:
            pred_bboxes = pred_bboxes_list[i]
            for n, bbox in enumerate(pred_bboxes):
                center = np.asarray([bbox.position_x, bbox.position_y, bbox.position_z])
                half = np.asarray([bbox.scale_x/2.0, bbox.scale_y/2.0, bbox.scale_z/2.0])
                safe_class_name = "_".join(bbox.class_name.split())
                name = f"object{bbox.id}_{safe_class_name}"
                rotation = R.from_rotvec([0, bbox.angle_z, 0]).as_matrix()
                rerun.log(f"/{entity_name}/world/bboxes_predicted/{name}",
                            rerun.Boxes3D(centers=center[None],
                                            half_sizes=half[None],
                                            colors=np.array([0,0,255]).astype(np.uint8)[None],
                                            labels=[name]),
                            rerun.InstancePoses3D(mat3x3=rotation),
                        )
    rerun.save(str(save_path))


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if draw.textlength(test_line, font=font) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def visualize_qa_on_image(
    images_with_mask, 
    question_update, answer1, pred_answer1, 
    question2_update, answer2, pred_answer2,
    font_path=None, font_size=24
):
    qa_list = [
        'Q1: '+question_update,
        answer1,
        pred_answer1,
        "\n\n",
        'Q2: '+question2_update,
        answer2,
        pred_answer2
    ]
    color_list = [
        (0, 0, 0),
        (0, 128, 0),
        (255, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 128, 0),
        (255, 0, 0),
    ]
    
    vis_h, vis_w = images_with_mask.shape[:2]
    canvas_tmp = np.ones((100, vis_w, 3), dtype=np.uint8) * 255
    img_pil_tmp = Image.fromarray(canvas_tmp)
    draw_tmp = ImageDraw.Draw(img_pil_tmp)
    
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)
    
    max_text_width = vis_w - 20
    total_lines = 0
    for text in qa_list:
        lines = wrap_text(text, font, max_text_width, draw_tmp)
        total_lines += len(lines)
    boarder_height = total_lines * (font_size + 5) + 10
    
    new_vis_h = vis_h + boarder_height
    canvas = np.ones((new_vis_h, vis_w, 3), dtype=np.uint8) * 255
    canvas[:vis_h, :vis_w] = images_with_mask
    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)
    
    text_x = 10
    text_y = vis_h + 5
    for idx, text in enumerate(qa_list):
        color = color_list[idx % len(color_list)]
        lines = wrap_text(text, font, max_text_width, draw)
        for l in lines:
            draw.text((text_x, text_y), l, font=font, fill=color)
            text_y += font_size + 5
    return np.array(img_pil)


def get_first_sentence(text):
    match = re.match(r'(.+?[.?])', text)
    if match:
        return match.group(1)
    else:
        return text


def xyz_to_2d_corners(bbox_cameraxy, intrinsic, return_id=True):
    camera_x = bbox_cameraxy.position_x
    camera_y = bbox_cameraxy.position_y
    camera_z = bbox_cameraxy.position_z
    scale_x = bbox_cameraxy.scale_x
    scale_y = bbox_cameraxy.scale_y
    scale_z = bbox_cameraxy.scale_z
    class_name = bbox_cameraxy.class_name
    if ' ' in class_name:
        class_name = ' '.join(class_name.split())
    idd = bbox_cameraxy.id

    half_x = scale_x / 2.0
    half_y = scale_y / 2.0
    half_z = scale_z / 2.0
    
    corners_3d = np.array([
        [-half_x, -half_y, half_z],
        [ half_x, -half_y, half_z],
        [ half_x,  half_y, half_z],
        [-half_x,  half_y, half_z],
        [-half_x, -half_y, -half_z],
        [ half_x, -half_y, -half_z],
        [ half_x,  half_y, -half_z],
        [-half_x,  half_y, -half_z]
    ])
    
    corners_3d[:, 0] += camera_x
    corners_3d[:, 1] += camera_y
    corners_3d[:, 2] += camera_z
    
    corners_2d = []
    for corner in corners_3d:
        x, y, z = corner
        if z <= 0:
            corners_2d.append(None)
            continue
            
        point_3d_homo = np.array([-x, -y, z, 1.0])
        point_2d_homo = intrinsic @ point_3d_homo[:3]
        
        if point_2d_homo[2] != 0:
            u = point_2d_homo[0] / point_2d_homo[2]
            v = point_2d_homo[1] / point_2d_homo[2]
            corners_2d.append((u, v))
        else:
            corners_2d.append(None)

    center_3d_homo = np.array([-camera_x, -camera_y, camera_z, 1.0])
    center_2d_homo = intrinsic @ center_3d_homo[:3]
    
    if center_2d_homo[2] != 0:
        u_center = center_2d_homo[0] / center_2d_homo[2]
        v_center = center_2d_homo[1] / center_2d_homo[2]
        center_2d = (u_center, v_center)
    else:
        center_2d = None

    min_x = min([corner[0] for corner in corners_2d if corner is not None])
    max_x = max([corner[0] for corner in corners_2d if corner is not None])
    min_y = min([corner[1] for corner in corners_2d if corner is not None])
    max_y = max([corner[1] for corner in corners_2d if corner is not None])
    width_2d = max_x - min_x
    height_2d = max_y - min_y
    bbox_2d = [center_2d[0], center_2d[1], width_2d, height_2d]
    
    if return_id:
        return {
        'corners_2d': corners_2d,
        'center_2d': center_2d,
        'bbox_2d': bbox_2d,
        'class_name': class_name,
        'id': idd
    }
    return {
        'corners_2d': corners_2d,
        'center_2d': center_2d,
        'bbox_2d': bbox_2d,
        'class_name': class_name
    }


def round2_the_list(input_list, round_num=2):
    return [round(x, round_num) for x in input_list]


def uvz_to_xyz(u, v, z, intrinsic):
    pixel = np.array([u, v, 1.0])
    Kinv = np.linalg.inv(intrinsic)
    xyz_cam = z * (Kinv @ pixel)
    return xyz_cam[0], xyz_cam[1], xyz_cam[2]


def visualize_3d_bbox_on_image(image, bbox_list, box_color=None, edges=None):
    annotated_image = image.copy()
    height, width = image.shape[:2]
    
    if edges is None:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
    
    for bbox_data in bbox_list:
        corners_2d_list = bbox_data['corners_2d']
        class_name = str(bbox_data['id']) + '_' + bbox_data['class_name']
        
        pixel_corners = []
        for corner in corners_2d_list:
            if corner is not None:
                u_pixel = corner[0] * width
                v_pixel = corner[1] * height
                pixel_corners.append((u_pixel, v_pixel))
            else:
                pixel_corners.append(None)
        
        valid_corners = []
        for corner in pixel_corners:
            if corner is not None:
                valid_corners.append(corner)
        
        if len(valid_corners) >= 4:
            if box_color is None:
                box_color = [random.randint(0, 255) for _ in range(3)]
            
            for start, end in edges:
                try:
                    if start < len(pixel_corners) and end < len(pixel_corners):
                        if pixel_corners[start] is not None and pixel_corners[end] is not None:
                            pt1 = tuple([int(_pt) for _pt in pixel_corners[start]])
                            pt2 = tuple([int(_pt) for _pt in pixel_corners[end]])
                            cv2.line(annotated_image, pt1, pt2, box_color, 2)
                except Exception:
                    continue
            
            for i, corner in enumerate(pixel_corners):
                if corner is not None:
                    pt = tuple([int(_pt) for _pt in corner])
                    cv2.circle(annotated_image, pt, 4, box_color, -1)
            
            front_corners = [pixel_corners[i] for i in [0, 1, 2, 3] if pixel_corners[i] is not None]
            if front_corners:
                front_center_x = np.mean([pt[0] for pt in front_corners])
                front_center_y = np.mean([pt[1] for pt in front_corners])
                
                text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                text_x = int(front_center_x - text_size[0] // 2)
                text_y = int(front_center_y - 10)
                
                cv2.putText(annotated_image, class_name, 
                          (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    return annotated_image


def generate_scene_condition(question2, pd_list, cur_pd_list_2d, auto_split, round2_the_list):
    scene_objs_3d = pd_list[0]
    num_total_objs = len(scene_objs_3d)
    
    full_caption_list_with_id = [
        f"{item['id']}_{item['class_name']}" for item in cur_pd_list_2d
    ]
    
    target_indices = []
    
    if not auto_split:
        matches = re.findall(r'\b\d+_[a-zA-Z0-9]+(?: [a-zA-Z0-9]+)*', question2)
        for match in matches:
            try:
                idx = full_caption_list_with_id.index(match)
                target_indices.append(idx)
            except ValueError:
                continue
    else:
        target_indices = list(range(num_total_objs))

    num_matches = len(target_indices)
    if num_matches == 0:
        return ""

    use_simple_name = auto_split and (num_matches <= 3)

    captions = []
    bboxes_3d = []
    bboxes_2d = []

    for idx in target_indices:
        if use_simple_name:
            captions.append(cur_pd_list_2d[idx]['class_name'])
        else:
            captions.append(full_caption_list_with_id[idx])
        
        obj = scene_objs_3d[idx]
        raw_bbox_3d = [obj.position_x, obj.position_y, obj.position_z, 
                       obj.scale_x, obj.scale_y, obj.scale_z]
        bboxes_3d.append(round2_the_list(raw_bbox_3d))
        
        raw_bbox_2d = cur_pd_list_2d[idx]['bbox_2d']
        bboxes_2d.append(round2_the_list(raw_bbox_2d))

    cond = ""
    
    if num_matches == 1:
        cond = f" The 3D bbox of {captions[0]} is {bboxes_3d[0]}. The 2D bbox of {captions[0]} is {bboxes_2d[0]}. "
        
    elif num_matches == 2:
        cond = (f" The 3D bbox of {captions[0]} is {bboxes_3d[0]}. "
                f"The 3D bbox of {captions[1]} is {bboxes_3d[1]}. "
                f"The 2D bbox of {captions[0]} is {bboxes_2d[0]}. "
                f"The 2D bbox of {captions[1]} is {bboxes_2d[1]}. ")
                
    elif num_matches == 3:
        names_str = f"{captions[0]}, {captions[1]} and {captions[2]}"
        bbox3d_str = f"{bboxes_3d[0]}, {bboxes_3d[1]} and {bboxes_3d[2]}"
        bbox2d_str = f"{bboxes_2d[0]}, {bboxes_2d[1]} and {bboxes_2d[2]}"
        
        cond = (f" The 3D bbox of {names_str} is {bbox3d_str}. "
                f"The 2D bbox of {names_str} is {bbox2d_str}. ")
                
    elif num_matches > 3:
        cond = (f" The 3D bbox of {captions} is {bboxes_3d}. "
                f"The 2D bbox of {captions} is {bboxes_2d}. ")

    return cond


def load_moge_model(device="cuda"):
    global _MOGE_MODEL
    if _MOGE_MODEL is None:
        print("Loading MoGe-2 model...")
        _MOGE_MODEL = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        _MOGE_MODEL.eval()
    return _MOGE_MODEL


def preprocess_image_moge(image_path, target_size=640):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist")
        
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise ValueError(f"Failed to read image: {image_path}")
        
    h, w = input_image.shape[:2]
    
    # Resize logic exactly as in your script
    if max(h, w) != target_size:
        scale = target_size / max(h, w)
        new_w = round(w * scale)
        new_h = round(h * scale)
        # Ensure exactly target_size on longest side
        if max(new_w, new_h) != target_size:
            if w > h:
                new_w, new_h = target_size, round(h * target_size / w)
            else:
                new_h, new_w = target_size, round(w * target_size / h)
        input_image = cv2.resize(input_image, (new_w, new_h))
    
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    return input_image_rgb

def get_moge_data(image_path, pcd_path=None, device="cuda"):
    print(f"Extracting point cloud for {os.path.basename(image_path)}...")
    
    model = load_moge_model(device)
    img_rgb = preprocess_image_moge(image_path)
    input_tensor = torch.tensor(img_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    with torch.no_grad():
        output = model.infer(input_tensor)
    
    depth_float32 = output["depth"].cpu().numpy().astype(np.float32)
    points = output["points"].cpu().numpy()
    
    # (x right, y down, z forward) -> (x left, y up, z forward)
    points[:, :, 0:2] *= -1 
    
    mask = output["mask"].cpu().numpy()
    intr_float32 = output["intrinsics"].cpu().numpy().astype(np.float32)
    
    # concat pcd
    pcd = np.concatenate([points, mask[..., None]], axis=-1) # h,w,4
    pcd_float16 = pcd.astype(np.float16)
    
    # save
    if pcd_path:
        os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
        np.savez_compressed(pcd_path, 
                            pcd=pcd_float16,
                            depth=depth_float32,
                            mask=mask,
                            intr=intr_float32)
        print(f"Saved generated point cloud to {pcd_path}")
    
    return pcd_path

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--root_json', type=str, default='config/detection.json',
                    help='Path to the root JSON file')
parser.add_argument('--args_output', type=str, default='outputs',
                    help='Path to the output directory')
parser.add_argument('--args_path', type=str, default='config/demo.yaml',
                    help='Path to the configuration YAML file')
parser.add_argument('--template_path', type=str, default="config/code_template.txt",
                    help='Path to the template file')
parser.add_argument('--save_dir_name', type=str, default='demo',
                    help='Directory name for saving output')
infer_args = parser.parse_args()

# paths
root_json = infer_args.root_json
args_output = infer_args.args_output
args_path = infer_args.args_path
template_path = infer_args.template_path
save_dir_name = infer_args.save_dir_name
_MOGE_MODEL = None


override_config = OmegaConf.from_cli([])
dict_config = yaml.safe_load(Path(args_path).absolute().read_text())
args = OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))

callbacks = []
config={"args": args, "callbacks": callbacks}

args = config.get("args")
model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(args)
data_args.preprocessing_num_workers = 1

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
config_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
model = Qwen2_5_VLForConditionalGeneration_pe.from_pretrained(
    model_args.model_name_or_path,
    config = config_pretrained,
)
model.config.cutoff_token_len = 1024
model.to("cuda")
model.eval()

template = get_template_and_fix_tokenizer(tokenizer, data_args)

with open(template_path, "r") as f:
    CODE_TEMPLATE = f.read()
SPLIT_TEMPLATE = " First locate the bbox of the aforementioned object(s). The reference code is as followed: <code_template>"


with open(root_json, 'r') as f:
    data = json.load(f)


for nn in trange(len(data)):
    item = data[nn]

    image_path = item['images'][0]
    pcd_path = item['points'][0]
    if not os.path.exists(pcd_path): # get pcd from moge2
        get_moge_data(image_path, pcd_path)
    question_type = item['question_type']

    question = item["messages"][0]["content"]
    answer = item["messages"][1]["content"]
    # if len(item["messages"]) > 3:
    #     question2 = item["messages"][2]["content"]
    #     answer2 = item["messages"][3]["content"]
    #     auto_split = False
    # else:
    #     question2 = ""
    #     answer2 = ""
    #     auto_split = True
    
    save_dict = []

    if '<image>' not in question:
        question_update = '<image>'+question
    else:
        question_update = question
    
    # if auto_split:
        # question_update = question_update + SPLIT_TEMPLATE
    
    if "<code_template>" not in question_update:
        question_update = question_update + "<code_template>"

    sample = {
            "messages": [
        {
            "content": question_update,
            "role": "user"
        },
        {
            "content": answer,
            "role": "assistant"
        }
        ],
        "images": [
            image_path
        ],
        "points": [
            pcd_path
        ]
        }
    messages = sample["messages"]

    depth_intr = np.load(pcd_path)
    source_points = depth_intr['pcd'].reshape(-1, 4).astype(np.float32)
    points = source_points[:, :3]

    init_mask = source_points[:,-1:]
    init_mask_bool = init_mask[:,0] > 0

    shift0 = np.min(points[init_mask_bool,:], axis=0)
    points -= shift0

    distances = np.linalg.norm(points[init_mask_bool, :3], axis=1)
    scaling_init = 1 / np.mean(distances)
    shift1_init = points[init_mask_bool, :3].min(axis=0)
    points[:, :3] = ((points[:, :3] - shift1_init) * scaling_init) + shift1_init

    updated_points = np.concatenate([points, init_mask], axis=1)

    image = Image.open(image_path)
    w, h = image.size
    if w >= h:
        new_w = 640
        new_h = int(h * 640 / w)
    else:
        new_h = 640
        new_w = int(w * 640 / h)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    num_image_tokens, num_video_tokens = 0, 0
    image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
    merge_length: int = getattr(image_processor, "merge_size") ** 2

    pixel_values = image_processor([image], return_tensors="pt")
    image_grid_thw = pixel_values['image_grid_thw']
    
    image_token = "<|image_pad|>"

    for message in messages:
        content = message["content"]

        while IMAGE_PLACEHOLDER in content:
            image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length
            content = content.replace(
                IMAGE_PLACEHOLDER, f"<|vision_start|>{image_token * image_seqlen}<|vision_end|>", 1
            )
            num_image_tokens += 1

        if "<code_template>" in content:
            content = content.replace("<code_template>", CODE_TEMPLATE)


        if 'bbox_0=Bbox(' in content:
            bbox_objects = parse_bbox_dict_uv(content)
            for bbox in bbox_objects:
                bbox.normalize_and_discretize(world_max=2.0, scale_max=0.625, num_bins=1000)
            content = serialize_bboxes_uv(bbox_objects)
        message["content"] = content

    input_ids, labels = template.mm_plugin.process_token_ids(
            [], [], image_path, [], [], updated_points, tokenizer, processor
        )

    encoded_pairs = template.encode_multiturn(tokenizer, messages, None, None)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)

    if data_args.mask_history:
        encoded_pairs = encoded_pairs[::-1]

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= data_args.cutoff_len:
            break

        source_len, target_len = infer_seqlen(
            len(source_ids), len(target_ids), data_args.cutoff_len - total_length
        )
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if data_args.train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if data_args.mask_history and turn_idx != 0:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if data_args.mask_history:
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]
    
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, labels))
    input_ids = source_ids

    attention_mask = [1] * len(input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
    labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0).cuda()
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).cuda()

    streamer = TextIteratorStreamer(
        tokenizer, timeout=200.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        {"input_ids": input_ids, 
        "point_clouds": torch.from_numpy(updated_points)[None].cuda(),
        "pixel_values": pixel_values['pixel_values'].cuda(),
        "image_grid_thw": pixel_values['image_grid_thw'].cuda(),
        },
        streamer=streamer,
        max_new_tokens=4096,
        do_sample=True,
        use_cache=True,
        temperature=0.6,
        top_p=0.95,
        top_k=10,
        num_beams=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    generate_texts = []
    for text in streamer:
        generate_texts.append(text)

    layout_str = " ".join(generate_texts)

    if 'bbox_0=Bbox(' in layout_str:
        try:
            bbox_objects_pred = parse_bbox_dict_uv(layout_str)
            bbox_objects_pred_unnormed = []
            for bbox in bbox_objects_pred:
                bbox_objects_pred_unnormed.append(copy.copy(bbox).undiscretize_and_unnormalize(world_max=2.0, scale_max=0.625, num_bins=1000))
        except:
            bbox_objects_pred = []
            bbox_objects_pred_unnormed = []
        
        for bbox in bbox_objects_pred_unnormed:
            bbox.shift(-shift1_init)
            bbox.scale(1/scaling_init)
            bbox.shift(shift1_init)
            bbox.shift(shift0)
        
        pred_language_string = serialize_bboxes_uv(bbox_objects_pred_unnormed)
    else:
        pred_language_string = layout_str

    gt_item = parse_bbox_dict_xy(answer)
    pd_item = parse_bbox_dict_xy(pred_language_string)
    updated_pd_item = []
    for bbox in pd_item:
        uu, vv = bbox.position_x, bbox.position_y
        camera_x, camera_y, camera_z = uvz_to_xyz(uu, vv, bbox.position_z, depth_intr['intr'])
        bbox.position_x, bbox.position_y = -camera_x, -camera_y
        updated_pd_item.append(bbox)
    pd_list = [updated_pd_item]
    updated_gt_item = []
    for bbox in gt_item:
        uu, vv = bbox.position_x, bbox.position_y
        camera_x, camera_y, camera_z = uvz_to_xyz(uu, vv, bbox.position_z, depth_intr['intr'])
        bbox.position_x, bbox.position_y = -camera_x, -camera_y
        updated_gt_item.append(bbox)
    gt_list = [updated_gt_item]

    save_dict.append(
            {
                "point_cloud": pcd_path,
                "image_path": image_path,
                "question": question,

                "Native 3D Grounding": pred_language_string,
            }
        )

    output_filename = os.path.basename(pcd_path).replace(".npz", f".json")
    os.makedirs(os.path.join(args_output, save_dir_name), exist_ok=True)
    save_path = os.path.join(args_output, save_dir_name, output_filename)
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False)

    depth_intr = np.load(pcd_path)
    pcd = depth_intr['pcd'].reshape(-1, 4).astype(np.float32)[:,:3]

    image = Image.open(image_path)
    if w >= h:
        new_w = 640
        new_h = int(pcd.shape[0] / 640)
    else:
        new_h = 640
        new_w = int(pcd.shape[0] / 640)
    image = image.resize((new_w, new_h))
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.asarray(image).reshape(-1, 3).astype(np.float32)/255.

    pcd = np.concatenate([pcd, rgb], axis=1)

    points_list1 = [pcd[:,:6]]
    vis_results_in_rrd(points_list = points_list1, \
    gt_bboxes_list = gt_list, \
        pred_bboxes_list = pd_list, \
            save_path=f'{args_output}/{save_dir_name}/{image_path.split("/")[-1].split(".")[0]}.rrd')
