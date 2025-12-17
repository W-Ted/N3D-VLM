
import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import torch
import re
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class MyBboxes_xy:
    id: int
    class_name: str
    position_x: float
    position_y: float
    position_z: float
    angle_z: float
    scale_x: float
    scale_y: float
    scale_z: float

    def __init__(self, id, class_name, position_x, position_y, angle_z, position_z, scale_x, scale_y, scale_z):
        """Post-initialization processing to ensure correct types"""
        self.id = int(id)
        self.class_name = str(class_name)
        self.position_x = float(position_x)
        self.position_y = float(position_y)
        self.position_z = float(position_z)
        self.angle_z = float(angle_z)
        self.scale_x = abs(float(scale_x))
        self.scale_y = abs(float(scale_y))
        self.scale_z = abs(float(scale_z))
    
    def normalize_and_discretize(self, world_min=0.0, world_max=32.0, \
                                    scale_min=0.0, scale_max=20.0, num_bins=1280, \
                                        angle_min=-2*np.pi, angle_max=2*np.pi):
        self.position_x = (
            (self.position_x - world_min) / (world_max - world_min) * num_bins
        )
        self.position_y = (
            (self.position_y - world_min) / (world_max - world_min) * num_bins
        )
        self.position_z = (
            (self.position_z - world_min) / (world_max - world_min) * num_bins
        )
        self.angle_z = (self.angle_z - angle_min) / (angle_max - angle_min) * num_bins

        self.scale_x = (self.scale_x - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_y = (self.scale_y - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_z = (self.scale_z - scale_min) / (scale_max - scale_min) * num_bins

        self.position_x = np.clip(int(self.position_x), 0, num_bins - 1)
        self.position_y = np.clip(int(self.position_y), 0, num_bins - 1)
        self.position_z = np.clip(int(self.position_z), 0, num_bins - 1)
        self.angle_z = np.clip(int(self.angle_z), 0, num_bins - 1)
        self.scale_x = np.clip(int(self.scale_x), 0, num_bins - 1)
        self.scale_y = np.clip(int(self.scale_y), 0, num_bins - 1)
        self.scale_z = np.clip(int(self.scale_z), 0, num_bins - 1)

    def undiscretize_and_unnormalize(self, world_min=0.0, world_max=32.0, \
                                     scale_min=0.0, scale_max=20.0, num_bins=1280, \
                                        angle_min=-2*np.pi, angle_max=2*np.pi):
        self.position_x = self.position_x / num_bins
        self.position_y = self.position_y / num_bins
        self.position_z = self.position_z / num_bins
        self.angle_z = self.angle_z / num_bins
        self.scale_x = self.scale_x / num_bins
        self.scale_y = self.scale_y / num_bins
        self.scale_z = self.scale_z / num_bins

        self.position_x = self.position_x * (world_max - world_min) + world_min
        self.position_y = self.position_y * (world_max - world_min) + world_min
        self.position_z = self.position_z * (world_max - world_min) + world_min
        self.angle_z = self.angle_z * (angle_max - angle_min) + angle_min
        self.scale_x = self.scale_x * (scale_max - scale_min) + scale_min
        self.scale_y = self.scale_y * (scale_max - scale_min) + scale_min
        self.scale_z = self.scale_z * (scale_max - scale_min) + scale_min

        return self
    
    def shift(self, shift=[0,0,0]):
        self.position_x = self.position_x + shift[0]
        self.position_y = self.position_y + shift[1]
        self.position_z = self.position_z + shift[2]
    
    def scale(self, scale=1.0):
        self.position_x = self.position_x * scale
        self.position_y = self.position_y * scale
        self.position_z = self.position_z * scale
        self.scale_x = self.scale_x * scale
        self.scale_y = self.scale_y * scale
        self.scale_z = self.scale_z * scale


@dataclass
class MyBboxes_uv:
    id: int
    class_name: str
    position_u: float
    position_v: float
    position_z: float
    angle_z: float
    scale_x: float
    scale_y: float
    scale_z: float

    def __init__(self, id, class_name, position_u, position_v, angle_z, position_z, scale_x, scale_y, scale_z):
        """Post-initialization processing to ensure correct types"""
        self.id = int(id)
        self.class_name = str(class_name)
        self.position_u = float(position_u)
        self.position_v = float(position_v)
        self.position_z = float(position_z)
        self.angle_z = float(angle_z)
        self.scale_x = abs(float(scale_x))
        self.scale_y = abs(float(scale_y))
        self.scale_z = abs(float(scale_z))
    
    def normalize_and_discretize(self, uv_min=0.0, uv_max=1.0, \
                                    world_min=0.0, world_max=32.0, \
                                    scale_min=0.0, scale_max=20.0, num_bins=1280, \
                                        angle_min=-2*np.pi, angle_max=2*np.pi, round_uv=False, keep_scale1=False):
        self.position_u = (
            (self.position_u - uv_min) / (uv_max - uv_min) * 1000
        )
        self.position_v = (
            (self.position_v - uv_min) / (uv_max - uv_min) * 1000
        )
        self.position_z = (
            (self.position_z - world_min) / (world_max - world_min) * num_bins
        )
        self.angle_z = (self.angle_z - angle_min) / (angle_max - angle_min) * num_bins

        self.scale_x = (self.scale_x - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_y = (self.scale_y - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_z = (self.scale_z - scale_min) / (scale_max - scale_min) * num_bins

        self.position_z = np.clip(int(self.position_z), 0, num_bins - 1)
        self.angle_z = np.clip(int(self.angle_z), 0, num_bins - 1)
        self.scale_x = np.clip(int(self.scale_x), 0, num_bins - 1)
        self.scale_y = np.clip(int(self.scale_y), 0, num_bins - 1)
        self.scale_z = np.clip(int(self.scale_z), 0, num_bins - 1)

        if round_uv:
            self.position_u = np.round(self.position_u)
            self.position_v = np.round(self.position_v)
            self.position_u = np.clip(int(self.position_u), 0, num_bins - 1)
            self.position_v = np.clip(int(self.position_v), 0, num_bins - 1)
        
        if keep_scale1:
            if self.scale_x == 0:
                self.scale_x = 1
            if self.scale_y == 0:
                self.scale_y = 1
            if self.scale_z == 0:
                self.scale_z = 1

    def undiscretize_and_unnormalize(self, uv_min=0.0, uv_max=1.0, \
                                    world_min=0.0, world_max=32.0, \
                                     scale_min=0.0, scale_max=20.0, num_bins=1280, \
                                        angle_min=-2*np.pi, angle_max=2*np.pi):
        self.position_u = self.position_u / 1000
        self.position_v = self.position_v / 1000
        self.position_z = self.position_z / num_bins
        self.angle_z = self.angle_z / num_bins
        self.scale_x = self.scale_x / num_bins
        self.scale_y = self.scale_y / num_bins
        self.scale_z = self.scale_z / num_bins

        self.position_u = self.position_u * (uv_max - uv_min) + uv_min
        self.position_v = self.position_v * (uv_max - uv_min) + uv_min
        self.position_z = self.position_z * (world_max - world_min) + world_min
        self.angle_z = self.angle_z * (angle_max - angle_min) + angle_min
        self.scale_x = self.scale_x * (scale_max - scale_min) + scale_min
        self.scale_y = self.scale_y * (scale_max - scale_min) + scale_min
        self.scale_z = self.scale_z * (scale_max - scale_min) + scale_min

        return self
    
    def shift(self, shift=[0,0,0]):
        self.position_z = self.position_z + shift[2]
    
    def scale(self, scale=1.0):
        self.position_z = self.position_z * scale
        self.scale_x = self.scale_x * scale
        self.scale_y = self.scale_y * scale
        self.scale_z = self.scale_z * scale
    
    def flip(self, axis='u'):
        assert axis in ['u']
        if axis == 'u':
            self.position_u = 1.0 - self.position_u


def parse_bbox_dict_xy(content):
    """
    Parse dictionary containing bbox strings into list of Bbox objects
    
    Args:
        data_dict: Dictionary with 'content' key containing bbox strings
        
    Returns:
        List of Bbox objects
    """
    bbox_objects = []
    # Regular expression to extract bbox data
    pattern = r'bbox_(\d+)=Bbox\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'
    
    for match in re.finditer(pattern, content):
        # Extract and convert values
        bbox_id = match.group(1)
        class_name = match.group(2).strip()
        values = [float(x.strip()) for x in match.groups()[2:]]
        
        # Create Bbox object
        bbox = MyBboxes_xy(
            id=bbox_id,
            class_name=class_name,
            position_x=values[0],
            position_y=values[1],
            position_z=values[2],

            angle_z=values[3],

            scale_x=values[4],
            scale_y=values[5],
            scale_z=values[6]
        )
        bbox_objects.append(bbox)
    
    return bbox_objects




def parse_bbox_dict_uv(content):
    """
    Parse dictionary containing bbox strings into list of Bbox objects
    
    Args:
        data_dict: Dictionary with 'content' key containing bbox strings
        
    Returns:
        List of Bbox objects
    """
    bbox_objects = []
    # Regular expression to extract bbox data
    pattern = r'bbox_(\d+)=Bbox\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\)'
    
    for match in re.finditer(pattern, content):
        # Extract and convert values
        bbox_id = match.group(1)
        class_name = match.group(2).strip()
        values = [float(x.strip()) for x in match.groups()[2:]]

        bbox = MyBboxes_uv(
            id=bbox_id,
            class_name=class_name,
            position_u=values[0],
            position_v=values[1],
            position_z=values[2],

            angle_z=values[3],

            scale_x=values[4],
            scale_y=values[5],
            scale_z=values[6]
        )
        bbox_objects.append(bbox)
    
    return bbox_objects



def serialize_bboxes_uv(bbox_objects, lower_category=False):
    """
    Convert list of MyBboxes objects back to the original string format
    
    Args:
        bbox_objects: List of MyBboxes instances
        
    Returns:
        Dictionary with 'content' key containing the serialized string
    """
    lines = []
    for _, bbox in enumerate(bbox_objects): # note: idx is not bbox.id
        if lower_category:
            line = (f"bbox_{bbox.id}=Bbox({bbox.class_name.lower()},{bbox.position_u},{bbox.position_v},{bbox.position_z},{bbox.angle_z},"
                f"{bbox.scale_x},{bbox.scale_y},{bbox.scale_z})")
        else:
            line = (f"bbox_{bbox.id}=Bbox({bbox.class_name},{bbox.position_u},{bbox.position_v},{bbox.position_z},{bbox.angle_z},"
                f"{bbox.scale_x},{bbox.scale_y},{bbox.scale_z})")
        lines.append(line)
    
    # Join with newlines and add final newline to match original format
    content = '\n'.join(lines) + '\n'
    return content
