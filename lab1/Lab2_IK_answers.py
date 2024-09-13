import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import functools
import os
from torch.utils.data import Dataset, DataLoader
import tqdm
from task2_inverse_kinematics import MetaData


def _axis_angle_rotation(axis: str, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == 'X':
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == 'Y':
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == 'Z':
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError("Invalid convention ({}).".format(convention))
    for letter in convention:
        if letter not in ('X', 'Y', 'Z'):
            raise ValueError(
                f"Invalid letter ({0}) in convention string".format(letter))

    matrices = map(_axis_angle_rotation, convention,
                   torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _forward_fk(path, joint_offsets, joint_rotations):
    if len(path) == 0:
        return None, None
    joint_orientations = torch.zeros_like(joint_rotations, dtype=torch.float32)
    joint_positions = torch.zeros_like(joint_offsets, dtype=torch.float32)
    # print(len(path))
    # print(joint_rotations.shape)
    start = 0
    stop = len(path)
    step = 1
    for i in range(start, stop, step):
        # print(i)
        if i == start:
            joint_orientations[i] = joint_rotations[i]
        else:
            joint_orientations[i] = torch.matmul(
                joint_orientations[i-step].clone(), joint_rotations[i])
        if i == start:
            continue
        else:
            joint_positions[i] = joint_positions[i-step] + \
                torch.matmul(joint_orientations[i-step], joint_offsets[i])
    return joint_positions, joint_orientations


def _backward_fk(path, joint_offsets, joint_rotations):
    if len(path) == 0:
        return None, None
    joint_orientations = torch.zeros_like(joint_rotations, dtype=torch.float32)
    joint_positions = torch.zeros_like(joint_offsets, dtype=torch.float32)
    # print(len(path))
    # print(joint_rotations.shape)
    start = len(path) - 1
    stop = -1
    step = -1
    for i in range(start, stop, step):
        # print(i)
        if i == start:
            joint_orientations[i] = torch.eye(3)
        else:
            joint_orientations[i] = torch.matmul(
                joint_orientations[i-step].clone(), joint_rotations[i])

        if i == start:
            continue
        else:
            joint_positions[i] = joint_positions[i-step] + \
                torch.matmul(joint_orientations[i-step], joint_offsets[i])
    return joint_positions, joint_orientations


def fk(path, joint_offsets, joint_rotations, root_num):
    path_forward = []
    path_backward = []
    if path.count(root_num) != 0:
        root_index = path.index(root_num)
    else:
        root_index = 0
    # print(root_index)
    if root_index > 0:
        path_forward = path[root_index:]
        path_backward = path[: root_index+1]
    else:
        path_forward = path
    joint_position_0, joint_orientations_0 =\
        _forward_fk(
            path_forward, joint_offsets[root_index:], joint_rotations[root_index:])
    joint_position_1, joint_orientations_1 =\
        _backward_fk(
            path_backward, joint_offsets[: root_index+1], joint_rotations[: root_index+1])
    if len(path_backward) == 0:
        joint_positions = joint_position_0
        joint_orientations = joint_orientations_0
    else:
        joint_positions = torch.concat(
            (joint_position_1[: -1], joint_position_0))
        joint_orientations = torch.concat(
            (joint_orientations_1[: -1], joint_orientations_0))
    return joint_positions, joint_orientations


class TargetSet(Dataset):
    def __init__(self, size):
        self.target = torch.tensor(np.random.normal(0.5, 0.25, size=(size, 1, 3)),
                                   dtype=torch.float32)
        self.len = size

    def __getitem__(self, index):
        return self.target[index, ...]

    def __len__(self):
        return self.len


class IK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IK, self).__init__()
        self.in_channels = in_channels
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, 100),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(100, 50),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(50, 50),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(50, 20),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(20, 20),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(20, 10),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(10, out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x, y):
        input = torch.concat((x, y), dim=1)
        input = input.view(-1, self.in_channels)
        # print(input.shape)
        output = self.fc1(input)
        return output


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    joint_positions = joint_positions.copy()
    joint_orientations = joint_orientations.copy()
    result_positions = np.zeros_like(joint_positions)
    result_orientations = np.zeros_like(joint_orientations)
    alpha = torch.tensor(1e-1)
    epsilon = 1e-1
    lamb = 0.5
    path, path_name, _, _ = meta_data.get_path_from_root_to_end()
    w = torch.tensor([[1.] for i in range(len(path))], dtype=torch.float32)
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent

    def is_root(joint_num):
        return joint_parent[joint_num] == -1

    new_joint_parent = joint_parent.copy()
    now_joint = path[0]
    new_joint_parent[now_joint] = -1
    # reparent the model
    while not is_root(now_joint):
        parent_joint = joint_parent[now_joint]
        new_joint_parent[parent_joint] = now_joint
        now_joint = parent_joint
    joint_parent = new_joint_parent.copy()
    # w = torch.tensor(np.array([[1] for i in range(len(path), 0, -1)]))

    # 所有关节的方向，旋转矩阵
    joint_orientations_matrices = R.from_quat(
        joint_orientations).as_matrix()
    result_orientations_matrices = np.zeros_like(joint_orientations_matrices)

    # torch.autograd.set_detect_anomaly(True)

    def get_offsets():
        joint_offsets = np.zeros((len(joint_name), 3))
        for i in range(len(joint_name)):
            if is_root(i):
                joint_offsets[i] = np.zeros((3))
            else:
                joint_offsets[i] = np.matmul(np.transpose(joint_orientations_matrices[joint_parent[i]]),
                                             (joint_positions[i] - joint_positions[joint_parent[i]]))
        return joint_offsets

    def get_rotations():
        joint_rotations_matrices = np.zeros_like(
            joint_orientations_matrices)
        for i in range(len(joint_name)):
            if is_root(i):
                joint_rotations_matrices[i] = joint_orientations_matrices[i]
            else:
                joint_rotations_matrices[i] = np.matmul(np.transpose(
                    joint_orientations_matrices[i-1]), joint_orientations_matrices[i])
        return R.from_matrix(joint_rotations_matrices).as_euler('XYZ')

    def get_root_num():
        for i in range(len(joint_name)):
            if is_root(i):
                return i

    # change_parent()
    root_num = get_root_num()

    # 所有关节的偏移
    joint_offsets = get_offsets()
    joint_rotations = get_rotations()

    # 运动关节的偏移
    changable_offsets = torch.tensor(joint_offsets[path], dtype=torch.float32)

    result_rotations = torch.zeros((len(path), 3), dtype=torch.float32)
    distance = torch.tensor(1e12, dtype=torch.float32)
    time = 0
    temp_positions = None
    temp_orientations = None
    final_rotations = np.zeros_like(joint_positions)

    while distance.item() > epsilon and time < 2e2:
        if time == 0:
            changable_rotations = torch.tensor(joint_rotations[path],
                                               dtype=torch.float32, requires_grad=True)
        else:
            changable_rotations = result_rotations.clone().detach().requires_grad_(True)

        matrices = [euler_angles_to_matrix(changable_rotations[i], 'XYZ') for i in range(
            changable_rotations.shape[0])]

        changable_rotations_matrices = torch.stack(matrices, 0)

        fk_positions, fk_orientations = fk(
            path, changable_offsets, changable_rotations_matrices, root_num)
        fk_positions = fk_positions + \
            torch.tensor(joint_positions[path[0]]) - fk_positions[0]
        final_position = fk_positions[-1]

        target_pose_tensor = torch.tensor(target_pose)
        rotations_loss = torch.linalg.norm(w * changable_rotations, 2).pow(2)
        loss = 0.5 * torch.dist(final_position, target_pose_tensor)\
            + lamb * 0.5 * rotations_loss

        loss.backward()
        delta_theta = changable_rotations.grad

        with torch.no_grad():
            result_rotations = changable_rotations - alpha * delta_theta
        changable_rotations.grad.zero_()
        distance = torch.dist(final_position, target_pose_tensor)
        time += 1
        temp_positions = fk_positions.detach().numpy()
        temp_orientations_matrices = fk_orientations.detach().numpy()
        final_rotations[path, :] = changable_rotations.detach().numpy()

    print("dist: ", distance)
    print("time: ", time)
    temp_orientations = R.from_matrix(temp_orientations_matrices).as_quat()
    result_positions[path, :] = temp_positions
    result_orientations[path, :] = temp_orientations
    result_orientations_matrices[path, :] = temp_orientations_matrices

    # reset other joints
    def calculate_orientation(joint_num):
        rotation_matrix = R.from_euler(
            'XYZ', final_rotations[joint_num]).as_matrix()
        if is_root(joint_num):
            return result_orientations_matrices[joint_num]
        parent_num = joint_parent[joint_num]
        return np.matmul(calculate_orientation(parent_num), rotation_matrix)
    
    def calculate_position(joint_num):
        if is_root(joint_num):
            return result_positions[joint_num]
        parent_num = joint_parent[joint_num]
        return np.matmul(result_orientations_matrices[parent_num], joint_offsets[joint_num])\
            + calculate_position(parent_num)
    
    for i in range(len(joint_name)):
        if path.count(i) == 0:
            result_orientations_matrices[i] = calculate_orientation(i)
            result_orientations[i] = R.from_matrix(
                result_orientations_matrices[i]).as_quat()
            
    for i in range(len(joint_name)):
        if path.count(i) == 0:
            result_positions[i] = calculate_position(i)
    return result_positions, result_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pose = None
    path, path_name, _, _ = meta_data.get_path_from_root_to_end()
    root = path[0]
    # print('before: ', joint_positions[path])
    target_pose = np.array([joint_positions[root, 0] + relative_x,
                            target_height, joint_positions[root, 2] + relative_z])
    # print('middle: ', joint_positions[path])
    # print(target_pose)
    temp_positions, temp_orientations = part1_inverse_kinematics(
        meta_data, joint_positions.copy(), joint_orientations.copy(), target_pose)
    joint_positions[path, :] = temp_positions[path, :]
    joint_orientations[path, :] = temp_orientations[path, :]
    # print('after: ', joint_positions[path])
    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    path, _, _, _ = meta_data.get_path_from_root_to_end()
    new_joint_positions, new_joint_orientations = part1_inverse_kinematics(meta_data, joint_positions,
                                                                           joint_orientations, left_target_pose)
    meta_data_r = MetaData(meta_data.joint_name, meta_data.joint_parent,
                           new_joint_positions, 'lowerback_torso', 'rWrist_end')
    result_joint_positions, result_joint_orientations = part1_inverse_kinematics(meta_data_r, new_joint_positions,
                                                                                 new_joint_orientations, right_target_pose)
    for id in path:
        result_joint_orientations[id] = new_joint_orientations[id]
        result_joint_positions[id] = new_joint_positions[id]
        
    return result_joint_positions, result_joint_orientations
