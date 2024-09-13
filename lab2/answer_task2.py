# 以下部分均为可更改部分

from answer_task1 import *
import torch
from scipy.spatial import KDTree
from pfnn.pfnn_model import PFNN
from smooth_utils import quat_to_avel, \
    decay_spring_implicit_damping_rot, decay_spring_implicit_damping_pos
import os
dir = "record_" + time.strftime('%a_%b_%d_%H_%M_%S_%Y', time.localtime())
fps = 60.
eps = 1e-3


def init_idle_motion(ctrler):
    init_pos_xz = np.array([0, 0])
    init_facing_xz = np.array([0, 1])
    idle_loop_motion = build_loop_motion(
        BVHMotion('motion_material/idle.bvh'))
    idle_loop_motion = idle_loop_motion.translation_and_rotation(
        0, init_pos_xz, init_facing_xz)
    idle_loop_motion.joint_position[-1, 0, [0, 2]] = np.array([0, 0])
    cur_pos_xz = ctrler.current_desired_position[[0, 2]]
    cur_facing_xz = R.from_quat(ctrler.current_desired_rotation).apply(
        np.array([0, 0, 1])).flatten()[[0, 2]]
    idle_loop_motion = idle_loop_motion.translation_and_rotation(
        0, cur_pos_xz, cur_facing_xz)
    return idle_loop_motion.raw_copy()


def init_walk_motion(ctrler):
    init_pos_xz = np.array([0, 0])
    init_facing_xz = np.array([0, 1])
    walk_loop_motion = build_loop_motion(
        BVHMotion('motion_material/walk_forward.bvh'))
    walk_loop_motion = walk_loop_motion.translation_and_rotation(
        0, init_pos_xz, init_facing_xz)
    walk_loop_motion_0 = walk_loop_motion.sub_sequence(9)
    walk_loop_motion_1 = walk_loop_motion.sub_sequence(0, 9)
    last_pos_xz = walk_loop_motion_0.joint_position[-1, 0, [0, 2]]
    last_facing_xz = R.from_quat(walk_loop_motion_0.joint_rotation[-1, 0]).apply(
        np.array([0, 0, 1])).flatten()[[0, 2]]
    walk_loop_motion_1 = walk_loop_motion_1.translation_and_rotation(
        0, last_pos_xz, last_facing_xz)
    walk_loop_motion_0.append(walk_loop_motion_1)
    walk_loop_motion = walk_loop_motion_0.raw_copy()
    cur_pos_xz = ctrler.current_desired_position[[0, 2]]
    cur_facing_xz = R.from_quat(ctrler.current_desired_rotation).apply(
        np.array([0, 0, 1])).flatten()[[0, 2]]
    walk_loop_motion = walk_loop_motion.translation_and_rotation(
        0, cur_pos_xz, cur_facing_xz)
    walk_loop_motion.joint_position[:, 0, 0] = 0.
    # walk_loop_motion.joint_rotation[:, 0] = walk_loop_motion.joint_rotation[0, 0]
    return walk_loop_motion.raw_copy()


def init_run_motion(ctrler):
    init_pos_xz = np.array([0, 0])
    init_facing_xz = np.array([0, 1])
    run_loop_motion = build_loop_motion(
        BVHMotion('motion_material/run_forward.bvh'))
    run_loop_motion = run_loop_motion.translation_and_rotation(
        0, init_pos_xz, init_facing_xz)
    cur_pos_xz = ctrler.current_desired_position[[0, 2]]
    cur_facing_xz = R.from_quat(ctrler.current_desired_rotation).apply(
        np.array([0, 0, 1])).flatten()[[0, 2]]
    run_loop_motion = run_loop_motion.translation_and_rotation(
        0, cur_pos_xz, cur_facing_xz)
    run_loop_motion.joint_position[:, 0, 0] = 0
    run_loop_motion.joint_rotation[:, 0] = run_loop_motion.joint_rotation[0, 0]
    return run_loop_motion.raw_copy()


def align_motion_with_desire(desire_pos, desire_rot, motion, frame=0):
    res = motion.raw_copy()
    cur_pos_xz = desire_pos[[0, 2]]
    cur_facing_xz = R.from_quat(desire_rot).apply(np.array([0, 0, 1]))[[0, 2]]
    res = res.translation_and_rotation(
        frame, cur_pos_xz, cur_facing_xz)
    return res


def local_motion(motion: BVHMotion):
    res = motion.raw_copy()
    res.joint_position[:, 0, [0, 2]] = np.array([0, 0])
    return res


class State:
    def __init__(self, name: str, animation: BVHMotion):
        self.name = name
        self.anim = animation

    def enter(self, last_state):
        pass

    def update(self, frame):
        return self.anim[frame]

    def exit(self, exit_state):
        pass


class TransformCondition:
    def __init__(self,):
        pass


class StateMachine():
    any_state = State("AnyState", None)

    def __init__(self):
        self.variables = {}

        self.init_state = None
        self.cur_state = self.init_state

        self.states = {}
        self.graph = {}
        self.transforms = {}

    def register_state(self, state: State):
        if self.init_state is None:
            self.init_state = state
            self.cur_state = self.init_state
        self.states[state.name] = state

    def register_transform(self, state1: str, state2: str, condition: TransformCondition):
        assert self.states.get(
            state1) is not None, "Invalid state1: %s" % state1
        assert self.states.get(
            state2) is not None, "Invalid state2: %s" % state2

        self.graph[state1].append(state2)
        self.transforms[(state1, state2)] = condition

    def begin_play(self):
        pass

    def tick(self):
        pass


def lerp(t, x1, x2):
    return (1 - t) * x1 + t * x2


def slerp(t, q1, q2):
    # 暂时先写主要逻辑，还有判断长短弧以及对齐四元数的问题
    theta = np.arccos(np.einsum("...i,...i->...", q1, q2))
    return (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)


def mse(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))


class Pose():
    def __init__(self, root_vel, lf_local_pos, lf_vel, rf_local_pos, rf_vel, motion_flag, index):
        self.root_vel = root_vel
        # self.lh_local_pos = lh_local_pos
        # self.lh_vel = lh_vel
        self.lf_local_pos = lf_local_pos
        self.lf_vel = lf_vel
        # self.rh_local_pos = rh_local_pos
        # self.rh_vel = rh_vel
        self.rf_local_pos = rf_local_pos
        self.rf_vel = rf_vel
        self.motion_flag = motion_flag
        self.index = index
        self.data = np.concatenate([
            self.root_vel, self.lf_local_pos, self.rf_local_pos, self.lf_vel, self.rf_vel]).reshape(-1)

    def get_cost(self, other):
        root_vel_cost = mse(self.root_vel, other.root_vel)
        foot_pos_cost = mse(self.lf_local_pos, other.lf_local_pos) + \
            mse(self.rf_local_pos, other.rf_local_pos)
        foot_vel_cost = mse(self.lf_vel, other.lf_vel) + \
            mse(self.rf_vel, other.rf_vel)
        # hand_pos_cost = mse(self.lh_local_pos, other.lh_local_pos) + \
        #     mse(self.rh_local_pos, other.rh_local_pos)
        # hand_vel_cost = mse(self.lh_vel, other.lh_vel) + \
        #     mse(self.rh_vel, other.rh_vel)
        # + hand_pos_cost + hand_vel_cost
        return root_vel_cost + foot_pos_cost + foot_vel_cost

    @staticmethod
    def build(flag: str, motion: BVHMotion, index):
        root_vel = (motion.joint_position[index + 1, 0] -
                    motion.joint_position[index, 0]) * fps
        lf_pos = motion.joint_translation[index, 3] - \
            motion.joint_translation[index, 0]
        rf_pos = motion.joint_translation[index, 8] - \
            motion.joint_translation[index, 0]
        lf_vel = (motion.joint_translation[index + 1, 3] -
                  motion.joint_translation[index, 3]) * fps
        rf_vel = (motion.joint_translation[index + 1, 8] -
                  motion.joint_translation[index, 8]) * fps
        # lh_pos = motion.joint_translation[index, 18] - \
        #     motion.joint_translation[index, 0]
        # rh_pos = motion.joint_translation[index, 23] - \
        #     motion.joint_translation[index, 0]
        # lh_vel = (motion.joint_translation[index + 1, 18] -
        #           motion.joint_translation[index, 18]) * fps
        # rh_vel = (motion.joint_translation[index + 1, 23] -
        #           motion.joint_translation[index, 23]) * fps
        return Pose(root_vel, lf_pos, lf_vel, rf_pos, rf_vel, flag, index)


class TrajectoryPoint():
    def __init__(self, desired_pos, desired_vel, desired_rot):
        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
        self.desired_rot = desired_rot


class Trajectory():
    def __init__(self, desired_pos, desired_vel, desired_rot):
        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
        self.desired_rot = desired_rot

    def get_cost(self, other):
        pos_cost = np.sum(mse(self.desired_pos, other.desired_pos))
        vel_cost = np.sum(mse(self.desired_vel, other.desired_vel))
        return pos_cost + vel_cost


class Goal():
    def __init__(self, cur_vel, desired_pos, desired_vel, desired_rot):
        self.velocity = cur_vel
        self.trajectory = Trajectory(desired_pos, desired_vel, desired_rot)

    def get_cost(self, pose: Pose, trajectory: Trajectory):
        velocity_cost = mse(self.velocity, pose.root_vel)
        trajectory_cost = self.trajectory.get_cost(trajectory)
        return velocity_cost + trajectory_cost


class MotionMatching():
    def __init__(self, joint_name):
        self.joint_name = joint_name
        walk_motion = BVHMotion(
            'motion_material/kinematic_motion/long_walk.bvh')
        walk_motion = walk_motion.sub_sequence(160, 15600)
        # walk_motion = walk_motion.sub_sequence(160, 1000)
        walk_motion.adjust_joint_name(self.joint_name)
        walk_motion.joint_translation, walk_motion.joint_orientation = \
            walk_motion.batch_forward_kinematics()
        walk_mirror_motion = BVHMotion(
            'motion_material/kinematic_motion/long_walk_mirror.bvh')
        walk_mirror_motion = walk_mirror_motion.sub_sequence(160, 15600)
        walk_mirror_motion.adjust_joint_name(self.joint_name)
        walk_mirror_motion.joint_translation, walk_mirror_motion.joint_orientation = \
            walk_mirror_motion.batch_forward_kinematics()
        run_motion = BVHMotion(
            'motion_material/kinematic_motion/long_run.bvh')
        run_motion = run_motion.sub_sequence(170, 14180)
        # run_motion = run_motion.sub_sequence(170, 1000)
        run_motion.adjust_joint_name(self.joint_name)
        run_motion.joint_translation, run_motion.joint_orientation = \
            run_motion.batch_forward_kinematics()
        run_mirror_motion = BVHMotion(
            'motion_material/kinematic_motion/long_run_mirror.bvh')
        # run_mirror_motion = run_mirror_motion.sub_sequence(170, 14180)
        run_mirror_motion = run_mirror_motion.sub_sequence(170, 1000)
        run_mirror_motion.adjust_joint_name(self.joint_name)
        run_mirror_motion.joint_translation, run_mirror_motion.joint_orientation = \
            run_mirror_motion.batch_forward_kinematics()

        self.motions = {
            "walk": walk_motion,
            # "walk_mirror": walk_mirror_motion,
            "run": run_motion,
            # "run_mirror": run_mirror_motion
        }

        self.gaits = {
            "walk": 0,
            "run": 1
        }

        database = []
        for flag, motion in self.motions.items():
            for index in range(motion.motion_length - 61):
                pose = Pose.build(flag, motion, index)
                database.append(pose.data)
        self.database = np.array(database)
        self.kd_tree = KDTree(self.database, leafsize=100)

    def which_motion(self, index):
        last_end = 0
        count = 0
        for flag, motion in self.motions.items():
            count += motion.motion_length - 61
            if index < count:
                return flag, motion, index - last_end
            last_end = count

    def query(self, cur_pose: Pose, goal: Goal, base_frame, gait):
        query = cur_pose.data
        res_dist, res_index = self.kd_tree.query(query, 1000)
        best_cost = 10000000.
        best_pose = cur_pose

        for i in res_index:
            flag, motion, frame = self.which_motion(i)
            candidate = Pose.build(flag, motion, frame)
            cost = self.compute_cost(cur_pose, candidate, goal, gait, 100.)
            if cost < best_cost:
                best_cost = cost
                best_pose = candidate

        is_candidate_and_cur_same = (best_pose.motion_flag == cur_pose.motion_flag and
                                     abs(best_pose.index - cur_pose.index - base_frame) < 60)
        print("best motion: {}, best frame: {}, score: {}, is same: {}, cur frame: {}, base frame: {}, diff: {}".format(
            best_pose.motion_flag, best_pose.index, best_cost, is_candidate_and_cur_same,
            cur_pose.index, base_frame, abs(best_pose.index - cur_pose.index - base_frame)))
        if not is_candidate_and_cur_same:
            return True, best_pose
        else:
            return False, cur_pose

    def get_future(self, pose):
        motion = self.motions[pose.motion_flag]
        tp = np.array([20, 40, 60])
        root_pos = motion.joint_position[pose.index + tp, 0] - \
            motion.joint_position[pose.index, 0]
        root_vel = (motion.joint_position[pose.index + tp + 1, 0] -
                    motion.joint_position[pose.index + tp, 0]) * fps
        root_rot = (R.from_quat(motion.joint_rotation[pose.index, 0]).inv() *
                    R.from_quat(motion.joint_rotation[pose.index + tp, 0])).as_quat()
        trajectory = Trajectory(root_pos, root_vel, root_rot)
        return trajectory

    def compute_cost(self, current_pose: Pose, candidate_pose: Pose, goal: Goal, gait, responsiveness=1.):
        current_cost = current_pose.get_cost(candidate_pose)

        trajectory = self.get_future(candidate_pose)
        future_cost = goal.get_cost(candidate_pose, trajectory)

        gait_cost = 100000 * abs(gait - self.gaits[candidate_pose.motion_flag])

        return current_cost + responsiveness * future_cost + gait_cost


def lerp(t, x1, x2):
    return (1 - t) * x1 + t * x2


def inertial_blend_two_motions(motion1: BVHMotion, motion2: BVHMotion, mix_frame1):
    # process the rotation
    res = motion2.raw_copy()
    dt = 1. / fps
    half_life = 0.2

    motion1_avel = quat_to_avel(motion1.joint_rotation, dt)
    motion2_avel = quat_to_avel(motion2.joint_rotation, dt)
    rot_diff = (R.from_quat(motion1.joint_rotation[mix_frame1, ...]) *
                R.from_quat(motion2.joint_rotation[0, ...]).inv()).as_rotvec()
    avel_diff = motion1_avel[mix_frame1] - motion2_avel[0]

    pos_diff = motion1.joint_position[mix_frame1] - motion2.joint_position[0]
    pos_diff[:, [0, 2]] = 0
    vel1 = motion1.joint_position[mix_frame1] - \
        motion1.joint_position[mix_frame1 - 1]
    vel2 = motion2.joint_position[1] - motion2.joint_position[0]
    vel_diff = (vel1 - vel2) / fps

    for i in range(res.motion_length):
        rot_offset = decay_spring_implicit_damping_rot(
            rot_diff, avel_diff, half_life, i/fps
        )
        pos_offset = decay_spring_implicit_damping_pos(
            pos_diff, vel_diff, half_life, i/fps
        )
        offset_rot = R.from_rotvec(rot_offset[0])
        offset_pos = pos_offset[0]
        res.joint_rotation[i] = (
            offset_rot * R.from_quat(motion2.joint_rotation[i])).as_quat()
        res.joint_position[i] = motion2.joint_position[i] + offset_pos

    return res


class CharacterController():
    def __init__(self, controller) -> None:
        self.controller = controller
        self.idle_motion = build_loop_motion(
            BVHMotion('motion_material/idle.bvh'))
        self.idle_motion.joint_position[:, 0, [0, 2]] = 0.
        self.idle_motion.joint_translation, self.idle_motion.joint_orientation = \
            self.idle_motion.batch_forward_kinematics()
        self.walk_motion = build_loop_motion(
            BVHMotion('motion_material/walk_forward.bvh'))
        self.walk_motion.adjust_joint_name(self.idle_motion.joint_name)
        walk_speed = 0.8
        self.walk_motion.joint_position[:, 0, [0, 2]] = 0.
        self.run_motion = build_loop_motion(
            BVHMotion('motion_material/run_forward.bvh'))
        self.run_motion.adjust_joint_name(self.idle_motion.joint_name)
        run_speed = 4.0
        self.run_motion.joint_position[:, 0, [0, 2]] = 0.
        self.motions = {
            "idle": self.idle_motion,
            "walk": self.walk_motion,
            "run": self.run_motion
        }
        self.motion_vel = {
            "idle": np.array([0., 0., 0.]),
            "walk": np.array([walk_speed, walk_speed, walk_speed]),
            "walk_mirror": np.array([walk_speed, walk_speed, walk_speed]),
            "run": np.array([run_speed, run_speed, run_speed]),
            "run_mirror": np.array([run_speed, run_speed, run_speed])
        }
        self.cur_motion = self.idle_motion
        self.next_motion = self.idle_motion
        self.controller = controller
        self.cur_root_pos = self.controller.current_desired_position
        self.cur_root_rot = self.controller.current_desired_rotation
        self.cur_frame = 0
        self.last_state = "idle"
        self.cur_state = "idle"
        self.next_state = "idle"
        self.changing_state = False
        self.mix_time = 30

        self.controller_speed = np.array([0., 0., 0.])

        self.motion_matching = MotionMatching(self.idle_motion.joint_name)
        self.query_count = 0
        self.query_interval = 10
        self.base_frame = 0
        self.clip_length = 200

    def update_state(self,
                     desired_pos_list,
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他

        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''

        # 获取controller信息
        self.controller_speed = self.motion_vel[self.cur_state]
        self.controller.move_speed = self.controller_speed
        gait = self.controller.gait
        raw_input_vel = self.controller.input_vel
        input_vel = raw_input_vel * self.controller_speed[0]
        controller_rot = desired_rot_list[0]
        controller_pos = desired_pos_list[0]
        roted_input_vel = R.from_quat(controller_rot).apply(input_vel)
        joint_name = self.cur_motion.joint_name

        # State Machine
        # cur_motion_length = self.cur_motion.motion_length
        # if self.cur_frame == cur_motion_length or self.changing_state:
        #     if not self.changing_state:
        #         self.last_state = self.cur_state
        #     self.changing_state = False
        #     self.cur_frame = 0
        #     self.cur_motion = self.next_motion
        #     self.next_motion = self.motions[self.cur_state]

        # # update controller speed
        # if self.last_state != self.cur_state and self.cur_frame <= self.mix_time:
        #     self.controller_speed = lerp(self.cur_frame / self.mix_time,
        #                                  self.motion_vel[self.last_state], self.motion_vel[self.cur_state])
        # self.controller.move_speed = self.controller_speed

        # self.cur_root_pos = controller_pos
        # self.cur_root_rot = controller_rot
        # cur_root_pos_xz = self.cur_root_pos[[0, 2]]
        # cur_facing_xz = R.from_quat(self.cur_root_rot).apply(
        #     np.array([0, 0, 1]))[[0, 2]]
        # self.cur_motion = self.cur_motion.translation_and_rotation(
        #     self.cur_frame, cur_root_pos_xz, cur_facing_xz)

        # joint_translation, joint_orientation = self.cur_motion.batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]

        # if np.linalg.norm(roted_input_vel) > 1e-3:
        #     if gait == 0 and self.cur_state != "walk":
        #         target_motion = self.motions["walk"]
        #         target_start_frame = 0
        #         target_motion_length = self.motions["walk"].motion_length
        #         if self.cur_state == "run":
        #             mix_phase = self.cur_frame / cur_motion_length
        #             target_start_frame = np.floor(
        #                 mix_phase * target_motion_length).astype(np.int64)
        #             target_motion = self.motions["walk"].sub_sequence(
        #                 target_start_frame)
        #             target_motion.append(self.motions["walk"])
        #         self.cur_motion.append(self.cur_motion)
        #         self.next_motion = blend_motions(
        #             self.cur_motion, target_motion, self.cur_frame, self.mix_time)
        #         self.next_motion.append(
        #             self.motions["walk"].sub_sequence(
        #                 (target_start_frame + self.mix_time) % target_motion_length))

        #         # self.next_motion = inertial_transition(
        #         #     self.cur_motion, self.motions["walk"], self.cur_frame)

        #         self.last_state = self.cur_state
        #         self.cur_state = "walk"
        #         self.changing_state = True
        #     elif gait == 1 and self.cur_state != "run":
        #         target_motion = self.motions["run"]
        #         target_start_frame = 0
        #         target_motion_length = self.motions["run"].motion_length
        #         if self.cur_state == "walk":
        #             mix_phase = self.cur_frame / cur_motion_length
        #             target_start_frame = np.floor(
        #                 mix_phase * target_motion_length).astype(np.int64)
        #             target_motion = self.motions["run"].sub_sequence(
        #                 target_start_frame)
        #             target_motion.append(self.motions["run"])
        #         self.cur_motion.append(self.cur_motion)
        #         self.next_motion = blend_motions(
        #             self.cur_motion, target_motion, self.cur_frame, self.mix_time)
        #         self.next_motion.append(
        #             self.motions["run"].sub_sequence(
        #                 (target_start_frame + self.mix_time) % target_motion_length))

        #         # self.next_motion = inertial_transition(
        #         #     self.cur_motion, self.motions["run"], self.cur_frame)

        #         self.last_state = self.cur_state
        #         self.cur_state = "run"
        #         self.changing_state = True
        # elif np.linalg.norm(roted_input_vel) <= 1e-3 and self.cur_state != "idle":
        #     self.cur_motion.append(self.cur_motion)
        #     self.next_motion = blend_motions(
        #         self.cur_motion, self.motions["idle"], self.cur_frame, self.mix_time)
        #     self.next_motion.append(
        #         self.motions["idle"].sub_sequence(self.mix_time))

        #     # self.next_motion = inertial_transition(
        #     #     self.cur_motion, self.motions["idle"], self.cur_frame)

        #     self.last_state = self.cur_state
        #     self.cur_state = "idle"
        #     self.changing_state = True

        # self.cur_frame += 1
        # return joint_name, joint_translation, joint_orientation

        # Motion Matching
        cur_motion_length = self.cur_motion.motion_length
        if self.cur_frame == cur_motion_length or self.changing_state:
            if not self.changing_state:
                self.last_state = self.cur_state
            self.changing_state = False
            self.cur_frame = 0
            self.cur_motion = self.next_motion
            if self.cur_state == "idle":
                self.next_motion = self.idle_motion

        self.cur_root_pos = self.cur_motion.joint_position[self.cur_frame, 0]
        self.cur_root_rot = controller_rot
        cur_root_pos_xz = self.cur_root_pos[[0, 2]]
        cur_facing_xz = R.from_quat(self.cur_root_rot).apply(
            np.array([0, 0, 1]))[[0, 2]]
        self.cur_motion = self.cur_motion.translation_and_rotation(
            self.cur_frame, cur_root_pos_xz, cur_facing_xz)

        joint_translation, joint_orientation = self.cur_motion.batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]

        # print("cur_gait: {}, cur frame: {}, flag: {}, query count: {}".format(
        #     current_gait, self.cur_frame, self.cur_state, self.query_count))

        if self.query_count == self.query_interval:
            self.query_count = 0
            if np.linalg.norm(raw_input_vel) < eps:
                if self.cur_state != "idle":
                    self.changing_state = True
                    self.last_state = self.cur_state
                    next_motion = self.idle_motion
                    next_motion = next_motion.translation_and_rotation(
                        0, cur_root_pos_xz, cur_facing_xz)
                    self.next_motion = inertial_blend_two_motions(
                        self.cur_motion, next_motion, self.cur_frame)
                self.base_frame = 0
                self.cur_state = "idle"
            else:
                pose = Pose.build(
                    self.cur_state, self.cur_motion, self.cur_frame)
                goal = Goal(
                    roted_input_vel, desired_pos_list[1:4] -
                    desired_pos_list[0], desired_vel_list[1:4],
                    (R.from_quat(desired_rot_list[0]).inv() *
                        R.from_quat(desired_rot_list[1:4])).as_quat())
                not_same, best_pose = self.motion_matching.query(
                    pose, goal, self.base_frame, current_gait)
                if not_same:
                    self.changing_state = True
                    self.last_state = self.cur_state
                    self.cur_state = best_pose.motion_flag
                    self.base_frame = best_pose.index
                    next_motion = self.motion_matching.motions[
                        self.cur_state].sub_sequence(self.base_frame, self.base_frame + self.clip_length)
                    next_motion = next_motion.translation_and_rotation(
                        0, cur_root_pos_xz, cur_facing_xz)
                    self.next_motion = inertial_blend_two_motions(
                        self.cur_motion, next_motion, self.cur_frame)

        self.cur_frame += 1
        self.query_count += 1
        return joint_name, joint_translation, joint_orientation

        # PFNN
        # 1st step: get the desired trajectory and concatenate to the old trajectory
        future_position = torch.zeros([6, 3])
        future_rotation = torch.zeros([6, 4])
        future_gait = torch.zeros([6, 1])
        rot = torch.zeros((30, 4))
        rot[:, 0] = torch.linspace(desired_rot_list[0, 0],
                                   desired_rot_list[2, 0], 40)[:30]
        rot[:, 1] = torch.linspace(desired_rot_list[0, 1],
                                   desired_rot_list[2, 1], 40)[:30]
        rot[:, 2] = torch.linspace(desired_rot_list[0, 2],
                                   desired_rot_list[2, 2], 40)[:30]
        rot[:, 3] = torch.linspace(desired_rot_list[0, 3],
                                   desired_rot_list[2, 3], 40)[:30]
        rot_norm = rot.norm(dim=1).unsqueeze(dim=1)
        # rot_norm = torch.tensor(
        #     [torch.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) for q in rot])
        # print(rot.shape)
        # print(rot)
        # print(rot_norm.shape)
        # print(rot_norm)
        assert (rot.shape == (30, 4))
        assert (rot_norm.shape == (30, 1))
        rot = rot / rot_norm
        future_rotation = rot[::5]
        assert (future_rotation.shape == (6, 4))

        pos = torch.zeros(3)
        for i, q in enumerate(rot):
            pos += R.from_quat(q).apply(desired_vel_list[0])
            if i % 5 == 0:
                future_position[int(i // 5)] = pos
        gait = -1
        input_vel = self.controller.input_vel
        if (np.abs(input_vel) < 1e-6).all():
            gait = 0
        else:
            gait = self.controller.gait + 1
        gait_vec = torch.zeros(3)
        gait_vec[gait] = 1
        future_gait = gait_vec.repeat(30, 1)[:30:5]
        assert (future_gait.shape == (6, 3))
        trajectory_pos = torch.concat([self.last_trajectory[::5, :3], future_position],
                                      dim=0)
        trajectory_rot = torch.concat([self.last_trajectory[::5, 3:7], future_rotation],
                                      dim=0)
        trajectory_gait = torch.concat([self.last_trajectory[::5, 7:], future_gait],
                                       dim=0)
        assert (trajectory_pos.shape == (12, 3))
        assert (trajectory_rot.shape == (12, 4))
        assert (trajectory_gait.shape == (12, 3))
        trajectory = Trajectory(
            trajectory_pos, trajectory_rot, trajectory_gait)

        # 2nd step: input the trajectory into the pfnn
        local_motions, self.phase = self.pfnn.impl(
            self.pfnn_motion, trajectory, self.phase)
        print("phase: ", self.phase)

        # 3rd step: get the output and explain it into the translation and orientation
        self.pfnn_motion = local_motions
        joint_name = self.pfnn_motion.joint_name
        joint_translation = self.pfnn_motion.joint_translation[1]
        joint_orientation = self.pfnn_motion.joint_orientation[1]

        self.last_trajectory[:-1] = self.last_trajectory[1:].clone()
        self.last_trajectory[-1,
                             :3] = torch.tensor(self.pfnn_motion.joint_position[1, 0])
        self.last_trajectory[-1,
                             3:7] = torch.tensor(self.pfnn_motion.joint_rotation[1, 0])
        self.last_trajectory[-1, 7:] = gait_vec

        return joint_name, joint_translation, joint_orientation

        # self.last_tragectory = None

    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''

        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        # controller.set_rot(self.cur_root_rot)

        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.


# class Trajectory:
#     def __init__(self, pos, rot, gait):
#         assert (pos.shape[0] == rot.shape[0] == gait.shape[0])
#         self.len = pos.shape[0]
#         self.position = pos
#         self.rotation = rot
#         self.gait = gait

#     def __len__(self):
#         return self.len


# class PFNN_Impl:
#     def __init__(self, ):
#         self.model = PFNN(234, 277, torch.device("cuda"))
#         state_dict = torch.load("./pfnn_model_100.pth").state_dict()
#         self.model.load_state_dict(state_dict)

#     def impl(self, motion, trajectory, phase):
#         x = self.make_data(motion, trajectory)

#         # print("phase type: ", phase.dtype)
#         y, _ = self.model(x, phase)
#         # print(y.shape)
#         tp = y[0, :24].reshape(12, 2).detach().numpy()
#         td = y[0, 24:48].reshape(12, 2).detach().numpy()
#         lr = y[0, 198:273].reshape(-1, 3).detach().numpy()
#         # np.where(lr > np.pi, lr - 2 * np.pi, lr)
#         # np.where(lr < -np.pi, lr + 2 * np.pi, lr)
#         dp = y[0, 276].detach().numpy()
#         # print("dp type: ", dp.dtype)
#         res = motion.raw_copy()
#         res.joint_position[0, :] = res.joint_position[1, :]

#         res.joint_rotation[0, :] = res.joint_rotation[1, :]
#         res.joint_rotation[1, :] = R.from_rotvec(lr).as_quat()
#         res.joint_translation, res.joint_orientation = res.batch_forward_kinematics()
#         # root_pos = 0.9 * trajectory.root_position[]
#         res.translation_and_rotation(1, tp[6], td[6])
#         print("tp[6]: ", tp[6])
#         print("td[6]: ", td[6])

#         phase = phase + dp
#         phase = phase % 4.0

#         return res, phase

#     def make_data(self, motion: BVHMotion, trajectory: Trajectory):
#         assert (len(trajectory) == 12)
#         assert (motion.motion_length == 2)
#         tp = trajectory.position[:, [0, 2]].reshape(1, -1)[0]
#         td = R.from_quat(trajectory.rotation[:,]).apply(
#             torch.tensor([0, 0, 1]))[:, [0, 2]]
#         td = td.reshape(1, -1)[0]
#         tg = trajectory.gait.reshape(1, -1)[0]
#         assert (tp.shape == (24,))
#         assert (td.shape == (24,))
#         assert (tg.shape == (36,))
#         local_motion = motion.raw_copy()
#         local_motion.joint_position[:, 0] = np.array([0, 0, 0])
#         local_motion.joint_rotation[:, 0] = R.from_euler(
#             'zxy', np.array([0, 0, 0])).as_quat()
#         local_trans, _ = local_motion.batch_forward_kinematics()
#         local_vel = local_trans[1, ...] - local_trans[0, ...]
#         local_trans = torch.tensor(local_trans[0])
#         local_vel = torch.tensor(local_vel)
#         lp = local_trans.reshape(1, -1)[0]
#         lv = local_vel.reshape(1, -1)[0]
#         x = []
#         assert (lp.shape == (len(motion.joint_name) * 3,))
#         assert (lv.shape == (len(motion.joint_name) * 3,))
#         x.append(torch.tensor(tp, dtype=torch.float32))
#         x.append(torch.tensor(td, dtype=torch.float32))
#         x.append(torch.tensor(tg, dtype=torch.float32))
#         x.append(torch.tensor(lp, dtype=torch.float32))
#         x.append(torch.tensor(lv, dtype=torch.float32))
#         x = torch.concatenate(x, dim=0)
#         x.unsqueeze_(dim=0)
#         # print("x dtype: ", x.dtype)
#         assert (x.shape == (1, 234))
#         assert (x.dtype == torch.float32)
#         return x


# def record(array, filename):
#     with open(filename, 'a+') as f:
#         np.save(f, array)
