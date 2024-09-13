import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose： (20,4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs：指定参数，可能包含kp,kd
    输出： global_torque: (20,3)的numpy数组，表示每个关节的全局坐标下的目标力矩，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    kp = kargs.get('kp', 300)  # 需要自行调整kp和kd！ 而且也可以是一个数组，指定每个关节的kp和kd
    kd = kargs.get('kd', 20)

    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    joint_orientation = physics_info.get_joint_orientation()
    joint_avel = physics_info.get_body_angular_velocity()

    global_torque = np.zeros((20, 3))

    joint_orientation_rot = R.from_quat(joint_orientation[1:])
    parent_orientation_rot = R.from_quat(joint_orientation[parent_index[1:]])
    joint_cur_rotation = parent_orientation_rot.inv() * joint_orientation_rot
    joint_aim_rotation = R.from_quat(pose[1:])
    rotation_diff = (joint_aim_rotation *
                     joint_cur_rotation.inv()).as_euler('xyz', degrees=True)
    joint_cur_avel = joint_avel[1:]
    local_torque = kp * rotation_diff
    global_torque[1:] = parent_orientation_rot.apply(
        local_torque) - kd * joint_cur_avel
    root_torque = kp * (R.from_quat(pose[0]) * R.from_quat(
        joint_orientation[0]).inv()).as_euler('xyz', degrees=True)\
        - kd * joint_avel[0]
    global_torque[0] = root_torque
    for i in range(20):
        if np.linalg.norm(global_torque[i]) > 1000:
            global_torque[i] /= 50

    return global_torque


def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力
          global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
    '''
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 4000)  # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 20)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    global_root_force = kp * \
        (target_position - root_position) - kd * root_velocity
    return global_root_force, global_torque


def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    其余同上
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均
        为了仿真稳定最好不要在Toe关节上加额外力矩
    '''
    tar_pos = bvh.joint_position[0][0]
    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    parent_index = physics_info.parent_index

    joint_positions = physics_info.get_joint_translation()
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    # 适当前移
    tar_pos = tar_pos * 0.8 + \
        joint_positions[9] * 0.1 + joint_positions[10] * 0.1

    torque = np.zeros((20, 3))
    root_force, torque = part2_cal_float_base_torque(
        tar_pos, pose, physics_info)

    """
    compute necessary joint torques using Jacobian Transpose Control to repalce the effect of this force 
    (on COM --- RootJoint), only consider lower body joints, from Ankle to Hip
    the power is similar : COM_force * COM_velocity = joint_torque * joint_angular_velocity 
    -----> COM_velocity = Jocabian * joint_angular_velocity
    -----> joint_torque = COM_force * Jacobian
    -----> torque_i = (x - p_i) x f
    """
    torque -= np.cross(root_position - joint_positions, root_force)
    # # print(joint_name)
    # for i in range(len(torque)):
    #     if 'Knee' in joint_name[i] or 'Ankle' in joint_name[i] or 'Hip' in joint_name[i]:
    #         torque[i] -= np.cross(root_position -
    #                               joint_positions[i], root_force)

    return torque
