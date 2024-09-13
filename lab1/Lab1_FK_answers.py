import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = None
    joint_parent = None
    joint_offset = None
    stack = []
    record = 0
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        joint_name = []
        joint_parent = []
        joint_offset_list = []
        for i in range(len(lines)):
            # print(lines[i])
            if lines[i].startswith('ROOT'):
                joint_name.append(lines[i].split(' ')[-1].split('\n')[0])
                joint_parent.append(-1)
                break
        for line in lines[i+1:]:
            if line.find('{') != -1:
                stack.append(record)
                record += 1
            if line.find('OFFSET') != -1:
                [x, y, z] = line.split()[1: 4]
                # print(x, y, z)
                joint_offset_list.append([float(x), float(y), float(z)])
            if line.find('JOINT') != -1:
                joint_name.append(line.split(' ')[-1].split('\n')[0])
                joint_parent.append(stack[-1])
            if line.find('End') != -1:
                parent = stack[-1]
                joint_parent.append(parent)
                joint_name.append(joint_name[parent] + '_end')
            if line.find('}') != -1:
                stack.pop()
            if len(stack) == 0:
                break
        joint_offset = np.array(joint_offset_list, dtype=np.float32)
        # print(joint_name)
        # print(joint_parent)
        # print(joint_offset)
        # print(len(joint_name))
        # print(len(joint_parent))
        # print(joint_offset.shape)

    return joint_name, joint_parent, joint_offset


def deg2rad(degree):
    return (degree + 180) / 360 * 2 * np.pi


def calculate_position(parent_num, joint_positions, joint_rotations):
    if parent_num == -1:
        return


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None
    # print(motion_data[frame_id])
    joint_nums = []
    for i in range(len(joint_name)):
        if joint_name[i].find('_end') == -1:
            joint_nums.append(i)

    origin = None
    rotation = np.zeros((len(joint_name), 3), dtype=np.float32)

    def split_motion_data(motion_data, joint_num):
        origin = np.array(motion_data[: 3], dtype=np.float32)
        # print(rotation.shape)
        for i in range(1, len(joint_num)+1):
            # print(joint_num[i-1])
            rotation[joint_num[i-1]] = np.array(
                motion_data[3*(i): 3*(i+1)].reshape(1, -1), dtype=np.float32)
        return origin, rotation

    origin, rotation = split_motion_data(motion_data[frame_id], joint_nums)
    joint_positions = np.zeros((len(joint_name), 3), dtype=np.float32)
    joint_orientations = np.zeros((len(joint_name), 4), dtype=np.float32)
    joint_orientations_matrix = []

    def is_root(joint_num):
        return joint_parent[joint_num] == -1

    def calculate_position(joint_num):
        if is_root(joint_num):
            return joint_offset[joint_num] + origin
        parent_num = joint_parent[joint_num]
        return np.dot(joint_orientations_matrix[parent_num], joint_offset[joint_num])\
            + calculate_position(parent_num)

    def calculate_orientation(joint_num):
        rotation_matrix = R.from_euler(
            'XYZ', rotation[joint_num], degrees=True).as_matrix()
        if is_root(joint_num):
            return rotation_matrix
        parent_num = joint_parent[joint_num]
        return np.dot(calculate_orientation(parent_num), rotation_matrix)

    for i in range(len(joint_name)):
        joint_orientations_matrix.append(calculate_orientation(i))
        joint_orientations[i] = R.from_matrix(
            joint_orientations_matrix[-1]).as_quat()
        joint_positions[i] = calculate_position(i)

    # print(joint_orientations)
    # print(joint_positions)

    return joint_positions, joint_orientations


def is_leaf(joint_num, parent_list):
        return parent_list.count(joint_num) == 0


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    T_pose_joint_name, T_pose_joint_parent, _ =\
        part1_calculate_T_pose(T_pose_bvh_path)
    A_pose_joint_name, A_pose_joint_parent, _ =\
        part1_calculate_T_pose(A_pose_bvh_path)

    retarget_pairs = []
    A_pose_non_leaf_name = []
    T_pose_non_leaf_name = []
    for i in range(len(A_pose_joint_name)):
        if is_leaf(i, A_pose_joint_parent):
            continue
        A_pose_non_leaf_name.append(A_pose_joint_name[i])
    print(A_pose_non_leaf_name)
    for i in range(len(T_pose_joint_name)):
        if is_leaf(i, T_pose_joint_parent):
            continue
        T_pose_non_leaf_name.append(T_pose_joint_name[i])
    print(T_pose_non_leaf_name)
    for name in A_pose_non_leaf_name:
        numT = T_pose_non_leaf_name.index(name)
        retarget_pairs.append(numT)
    print(retarget_pairs)

    motion_data = load_motion_data(A_pose_bvh_path)
    frame_num = motion_data.shape[0]
    print(motion_data.shape)

    lShouder_numT = T_pose_non_leaf_name.index('lShoulder')
    rShouder_numT = T_pose_non_leaf_name.index('rShoulder')
    new_motion_data = np.zeros(motion_data.shape)
    new_motion_data[:, : 3] = motion_data[:, : 3]
    for i in range(1, len(retarget_pairs)+1):
        numT = retarget_pairs[i - 1] + 1
        new_motion_data[:, 3*numT: 3*(numT+1)] = motion_data[:, 3*i: 3*(i+1)]
        if numT == lShouder_numT:
            new_motion_data[:, 3*(numT+1)-1] -= 45
        elif numT == rShouder_numT:
            new_motion_data[:, 3*(numT+1)-1] += 45
    return new_motion_data
