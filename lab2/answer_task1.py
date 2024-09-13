import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
# ------------- lab1里的代码 -------------#
import time
filename = "record_" + \
    time.strftime('%a_%b_%d_%H_%M_%S_%Y', time.localtime()) + ".npy"


def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x)
                                             for x in line.split()[-3:]]).reshape(1, 3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order) + ''.join(rot_order)

            elif 'Frame Time:' in line:
                break

    joint_parents = [-1] + [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets


def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
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

# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#


'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''


class BVHMotion():
    def __init__(self, bvh_file_name=None) -> None:

        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []

        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None  # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None  # (N,M,4)的ndarray, 用四元数表示的局部旋转
        self.joint_translation = None
        self.joint_orientation = None

        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass

    # ------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)

        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros(
            (motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros(
            (motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                continue
            elif self.joint_channel[i] == 3:
                self.joint_position[:, i, :] = joint_offset[i].reshape(1, 3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:,
                                                           cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler(
                'XYZ', rotation, degrees=True).as_quat()
            cur_channel += self.joint_channel[i]

        return

    def batch_forward_kinematics(self, joint_position=None, joint_rotation=None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation

        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:, :, 3] = 1.0  # 四元数的w分量默认为1

        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向

        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:, pi, :])
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (
                parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation

    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name)
               for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name)
                   for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:, idx, :]
        self.joint_rotation = self.joint_rotation[:, idx, :]
        pass

    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)

    @property
    def motion_length(self):
        return self.joint_position.shape[0]

    def sub_sequence(self, start, end='e'):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        if end == 'e':
            res.joint_position = res.joint_position[start:, :, :]
            res.joint_rotation = res.joint_rotation[start:, :, :]
        else:
            res.joint_position = res.joint_position[start:end, :, :]
            res.joint_rotation = res.joint_rotation[start:end, :, :]
        return res

    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate(
            (self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate(
            (self.joint_rotation, other.joint_rotation), axis=0)
        pass

    # --------------------- 你的任务 -------------------- #

    def decompose_rotation_with_yaxis(self, rotation, method=0):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        # TODO: 你的代码
        rotation = R.from_quat(rotation)
        y_local = rotation.as_matrix()[..., :, 1]
        y_global = np.array([0., 1., 0.])

        _rotation_vec = np.cross(y_local, y_global)
        _rotation_sin = np.linalg.norm(
            _rotation_vec, axis=-1, keepdims=True)
        _rotation_cos = np.expand_dims(
            np.clip(np.dot(y_local, y_global), -1., 1.), axis=-1)
        _rotation_vec = _rotation_vec / (_rotation_sin + 1e-10)

        _rotation = np.arctan2(_rotation_sin, _rotation_cos)
        _rotation = R.from_rotvec(_rotation * _rotation_vec)

        Ry = _rotation * rotation
        Rxz = Ry.inv() * rotation
        return Ry, Rxz

    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz: np.ndarray):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        '''

        res = self.raw_copy()  # 拷贝一份，不要修改原始数据

        # TODO: 你的代码
        # Get target animation information
        tgt_pos_xz = target_translation_xz
        tgt_pos = np.array([tgt_pos_xz[0], 0., tgt_pos_xz[1]])
        tgt_rot_y = R.from_rotvec(np.arctan2(
            target_facing_direction_xz[0], target_facing_direction_xz[1]) * np.array([0, 1, 0]))
        # tgt_rot_y = R.from_euler("zxy", np.array([0., 0., tgt_rot_y]))

        # Get origin animation information
        org_pos = self.joint_position[frame_num, 0]  # Root Position
        org_pos_xz = org_pos[[0, 2]]

        org_rot = self.joint_rotation[frame_num, 0]  # Root Rotation
        org_rot_y, _ = self.decompose_rotation_with_yaxis(
            org_rot)  # decompose rotation

        # Root Rotation of All Frames
        org_anim_rot = R.from_quat(self.joint_rotation[:, 0])

        # Get position and rotation offset
        pos_offset_xz = tgt_pos_xz - org_pos_xz
        rot_offset_y = org_rot_y.inv() * tgt_rot_y

        # Debug
        # res.joint_position[:, 0, [0, 2]] = 0.
        # Step 1: Translate origin anim to a trace that pass the target pos on target frame
        res.joint_position[:, 0, [0, 2]] += pos_offset_xz
        pos_offset2kf = res.joint_position[:, 0] - tgt_pos

        # Step 2: Rotate origin anim's root rotation frame by frame
        res.joint_rotation[:, 0] = (
            rot_offset_y * org_anim_rot).as_quat()

        # Step 3: Rotate origin anim's root position frame by frame with key frame as pivot
        pos_offset2kf_roted = rot_offset_y.apply(pos_offset2kf)
        res.joint_position[:, 0, [0, 2]
                           ] = pos_offset2kf_roted[:, [0, 2]] + tgt_pos_xz
        return res

    def get_speed(self):
        temp = self.raw_copy()

        temp = temp.translation_and_rotation(
            0, np.array([0, 0]), np.array([0, 1]))
        return (temp.joint_position[-1, 0, 2] - temp.joint_position[0, 0, 2]) / temp.motion_length * 60

# part2


def blend_two_motions(bvh_motion1: BVHMotion, bvh_motion2: BVHMotion, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros(
        (len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros(
        (len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[..., 3] = 1.0

    # TODO: 你的代码
    length1 = bvh_motion1.motion_length - 1
    length2 = bvh_motion2.motion_length - 1
    indices1 = np.linspace(0, length1, len(alpha))
    indices2 = np.linspace(0, length2, len(alpha))
    joint_num = len(bvh_motion1.joint_name)

    for i, (j, k) in enumerate(zip(indices1, indices2)):
        j = np.floor(j).astype(np.int32)
        k = np.floor(k).astype(np.int32)
        bvh1_jpos_0 = bvh_motion1.joint_position[j, 0]
        bvh2_jpos_0 = bvh_motion2.joint_position[k, 0]
        bvh1_jrot = bvh_motion1.joint_rotation[j]
        bvh2_jrot = bvh_motion2.joint_rotation[k]

        res.joint_position[i, 0] = (
            1 - alpha[i]) * bvh1_jpos_0 + alpha[i] * bvh2_jpos_0
        res.joint_position[i, 1:] = bvh_motion1.joint_position[0, 1:]
        joint_rotations = []
        for jn in range(joint_num):
            bvh1_jn_rot = R.from_quat(bvh1_jrot[jn])
            bvh2_jn_rot = R.from_quat(bvh2_jrot[jn])
            bvh_jrots = R.concatenate([bvh1_jn_rot, bvh2_jn_rot])
            slerp = Slerp([0, 1], bvh_jrots)
            joint_rotations.append(slerp(alpha[i]).as_quat())
        res.joint_rotation[i] = np.array(joint_rotations)
    return res

# part3


def build_loop_motion_without_root_rot(bvh_motion: BVHMotion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()
    res_jpos_y = res.joint_position[:, 0, 1]
    res_jrot_euler = np.zeros((res.joint_position.shape[0],
                               res.joint_position.shape[1],
                               3), dtype=np.float32)
    for i in range(res.motion_length):
        res_jrot_euler[i] = R.from_quat(res.joint_rotation[i]).as_euler("zxy")
    dpos = res_jpos_y[-1] - res_jpos_y[0]
    drot = res_jrot_euler[-1, 1:] - res_jrot_euler[0, 1:]
    np.where(drot > np.pi, drot - 2 * np.pi, drot)
    np.where(drot < -np.pi, drot + 2 * np.pi, drot)
    drot = np.expand_dims(drot, axis=0)
    drot = np.repeat(drot, res.motion_length, axis=0)
    betas = np.linspace(0.5, -0.5, res.motion_length)
    betas = 2 * betas * np.abs(betas)
    rbetas = np.expand_dims(betas, axis=-1)
    rbetas = np.repeat(rbetas, res.joint_rotation.shape[1] - 1, axis=-1)
    rbetas = np.expand_dims(rbetas, axis=-1)
    new_jpos_y = res_jpos_y + betas * dpos
    new_jrot_euler = res_jrot_euler[:, 1:] + rbetas * drot
    res.joint_position[:, 0, 1] = new_jpos_y
    for i in range(res.motion_length):
        res.joint_rotation[i, 1:] = R.from_euler(
            "zxy", new_jrot_euler[i]).as_quat()

    # res2 = bvh_motion.raw_copy()
    # from smooth_utils import build_loop_motion
    # return build_loop_motion(res2)
    return res


def build_loop_motion(bvh_motion: BVHMotion, half_life=0.2, front_blend_weight=0.5):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res2 = bvh_motion.raw_copy()
    from smooth_utils import build_loop_motion
    return build_loop_motion(res2, half_life=half_life, blend_weight=front_blend_weight)


# part4
def blend_motions(motion1: BVHMotion, motion2: BVHMotion, mix_frame1, mix_time):
    alpha = np.linspace(0, 1, mix_time)
    blend_motion1 = motion1.sub_sequence(mix_frame1, mix_frame1 + mix_time)
    blend_motion2 = motion2.sub_sequence(0, mix_time)
    res = blend_two_motions(blend_motion1, blend_motion2, alpha)
    return res


def inertial_transition(motion1: BVHMotion, motion2: BVHMotion, mix_frame1):
    # process the rotation
    res = motion2.raw_copy()
    from smooth_utils import quat_to_avel, \
        decay_spring_implicit_damping_rot, decay_spring_implicit_damping_pos
    dt = 1. / 60.
    half_life = 0.2
    fps = 60
    motion1_avel = quat_to_avel(motion1.joint_rotation, dt)
    motion2_avel = quat_to_avel(motion2.joint_rotation, dt)

    rot_diff = (R.from_quat(motion1.joint_rotation[mix_frame1, ...]) *
                R.from_quat(motion2.joint_rotation[0, ...]).inv()).as_rotvec()
    avel_diff = motion1_avel[mix_frame1] - motion2_avel[0]
    for i in range(res.motion_length):
        offset1 = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i/fps
        )
        # offset2 = decay_spring_implicit_damping_rot(
        #     -rot_diff, -avel_diff, half_life, (res.motion_length-i-1)/fps
        # )
        offset_rot = R.from_rotvec(offset1[0])  # + offset1[0])
        res.joint_rotation[i] = (
            offset_rot * R.from_quat(motion2.joint_rotation[i])).as_quat()

    pos_diff = motion1.joint_position[mix_frame1] - motion2.joint_position[0]
    pos_diff[:, [0, 2]] = 0
    vel1 = motion1.joint_position[mix_frame1] - \
        motion1.joint_position[mix_frame1 - 1]
    vel2 = motion2.joint_position[1] - motion2.joint_position[0]
    vel_diff = (vel1 - vel2) / 60

    for i in range(res.motion_length):
        offset1 = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i/fps
        )
        # offset2 = decay_spring_implicit_damping_pos(
        #     -bbw * pos_diff, -bbw *
        #     vel_diff, half_life, (bvh_motion.motion_length-i-1)/fps
        # )
        offset_pos = offset1[0]  # + offset2[0]
        res.joint_position[i] = motion2.joint_position[i] + offset_pos

    return res


def concatenate_two_motions(bvh_motion1: BVHMotion, bvh_motion2: BVHMotion, mix_frame1, mix_time):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''

    motion1 = bvh_motion1.raw_copy()
    motion2 = bvh_motion2.raw_copy()

    m1p = motion1.joint_position[mix_frame1, 0, [0, 2]]
    m1fd = R.from_quat(motion1.joint_rotation[mix_frame1, 0]).apply(
        np.array([0, 0, 1]))[[0, 2]]
    # print("m1fd: ", m1fd)

    motion2 = motion2.translation_and_rotation(0, m1p, m1fd)

    res = blend_motions(motion1, motion2, mix_frame1, mix_time)
    res_motion = motion1.sub_sequence(0, mix_frame1)
    res_motion.append(res)
    res_motion.append(motion2.sub_sequence(mix_time))

    # res = inertial_transition(motion1, motion2, mix_frame1)
    # res_motion = motion1.sub_sequence(0, mix_frame1)
    # res_motion.append(res)

    return res_motion

    # return concatenate_two_motions_with_phase_info(bvh_motion1, bvh_motion2, mix_frame1, mix_time,
    #                                                100, 45)


def concatenate_two_motions_with_phase_info(
        bvh_motion1: BVHMotion, bvh_motion2: BVHMotion, mix_frame1, mix_time,
        loop_length1: int, loop_length2: int):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()
    motion2 = bvh_motion2.raw_copy()
    motion2 = build_loop_motion(motion2)
    pos = motion2.joint_position[-1, 0, [0, 2]]
    rot = motion2.joint_rotation[-1, 0]
    facing_axis = R.from_quat(rot).apply(
        np.array([0, 0, 1])).flatten()[[0, 2]]
    new_motion = motion2.translation_and_rotation(0, pos, facing_axis)
    motion2.append(new_motion)
    origin_motion2 = motion2

    # compute phase information from given motion length
    l1 = mix_time  # mix time should be less than one loop motion length
    start_phase = mix_frame1 / loop_length1 - \
        np.floor(mix_frame1 / loop_length1)
    phase_period = mix_time / loop_length1
    l2 = int(phase_period * loop_length2)
    motion2_start = int(start_phase * loop_length2)

    # align the motion phase
    motion1 = res.sub_sequence(0, mix_frame1)
    blend_motion1 = res.sub_sequence(mix_frame1, mix_frame1 + l1)
    blend_motion2 = motion2.sub_sequence(motion2_start, motion2_start + l2)

    compute_motion1 = blend_motion1.translation_and_rotation(
        0, np.array([0, 0]), np.array([0, 1]))
    compute_motion2 = blend_motion2.translation_and_rotation(
        0, np.array([0, 0]), np.array([0, 1]))

    # compute blend parameters
    v1 = compute_motion1.joint_position[-1, 0,
                                        2] / compute_motion1.motion_length
    v2 = compute_motion2.joint_position[-1, 0,
                                        2] / compute_motion2.motion_length

    # blend
    # blended_motion = blend_motions(
    #     compute_motion1, compute_motion2, v1, v2, l1, l2)

    # inertialization
    blended_motion = inertial_transition(
        compute_motion1, compute_motion2, v1, v2, l1, l2)

    res_loc_xz = res.joint_position[mix_frame1, 0][[0, 2]]
    res_facing_xz = R.from_quat(
        res.joint_rotation[mix_frame1, 0]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    blended_motion = blended_motion.translation_and_rotation(
        0, res_loc_xz, res_facing_xz)

    # ###concatenate motion1 with motion2###
    # clip the motion2 to fit the phase
    motion2_end = motion2.motion_length
    motion2 = motion2.sub_sequence(motion2_start + l2 - 1, motion2_end)

    # concatenate motion1 with the blended motion
    motion1.append(blended_motion)
    res = motion1.raw_copy()

    d_trans_loc_xz = motion2.joint_position[1, 0][[0, 2]] -\
        motion2.joint_position[0, 0][[0, 2]]
    trans_loc_xz = motion2.joint_position[0, 0][[0, 2]] - d_trans_loc_xz
    trans_facing_xz = R.from_quat(
        motion2.joint_rotation[0, 0]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
    res = res.translation_and_rotation(
        -1, trans_loc_xz, trans_facing_xz)
    res.joint_position = np.concatenate(
        [res.joint_position, motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate(
        [res.joint_rotation, motion2.joint_rotation], axis=0)

    # plt.plot(np.array([mix_frame1, mix_frame1]), np.array([0.6, 1]))
    # plt.plot(bvh_motion1.joint_position[:, 0, 1], c="red")
    # plt.plot(np.arange(mix_frame1, mix_frame1 + best_fn),
    #          blended_motion.joint_position[:, 0, 1] - 0.1, c="pink")
    # plt.plot(np.arange(res.motion_length - origin_motion2.motion_length, res.motion_length),
    #                    origin_motion2.joint_position[:, 0, 1], c="purple")
    # plt.plot(res.joint_position[:, 0, 1], label='y')
    # plt.plot(np.array([mix_frame1 + best_fn, mix_frame1 + best_fn]), np.array([0.6, 1]))
    # plt.legend()
    # plt.show()
    # for i in range(len(blended_motion.joint_name)):
    #     plt.figure("Plot %d" % i)
    #     plt.plot(np.array([mix_frame1, mix_frame1]), np.array([-1, 1]))
    #     plt.plot(R.from_quat(res.joint_rotation[:, i]).as_euler(
    #         'zxy')[:, 0], label='z')
    #     plt.plot(R.from_quat(res.joint_rotation[:, i]).as_euler(
    #         'zxy')[:, 1], label='x')
    #     plt.plot(R.from_quat(res.joint_rotation[:, i]).as_euler(
    #         'zxy')[:, 2], label='y')
    #     plt.plot(np.array([mix_frame1 + best_fn, mix_frame1 + best_fn]), np.array([-1, 1]))
    #     plt.show()
    # with open(filename, 'ab') as fobj:
    #     for i in range(len(blended_motion.joint_name)):
    #         # plt.figure("Plot %d" % i)
    #         np.save(fobj, R.from_quat(
    #             res.joint_rotation[:, i]).as_euler('zxy')[:, 0])
    #         np.save(fobj, R.from_quat(
    #             res.joint_rotation[:, i]).as_euler('zxy')[:, 1])
    #         np.save(fobj, R.from_quat(
    #             res.joint_rotation[:, i]).as_euler('zxy')[:, 2])

    return res
