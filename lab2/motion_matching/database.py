from answer_task1 import BVHMotion
import numpy as np

class Database():
    def __init__(self, animations: list[BVHMotion], joint_names: list[str]):
        anim_frames = 0
        anim_num = len(animations)
        joint_num = len(joint_names)
        for anim in animations:
            anim_frames += anim.motion_length
            anim.adjust_joint_name(joint_names)
            
        self.bone_positions = np.zeros((anim_frames, joint_num, 3), dtype=np.float32)
        self.bone_velocities = np.zeros((anim_frames, joint_num, 3), dtype=np.float32)
        self.bone_rotations = np.zeros((anim_frames, joint_num, 4), dtype=np.float32)
        self.bone_angular_velocities = np.zeros((anim_frames, joint_num, 3), dtype=np.float32)
        
        self.range_starts = np.zeros((anim_num), dtype=np.int64)
        self.range_stops = np.zeros((anim_num), dtype=np.int64)
        
        self. 
        