from answer_task1 import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py


def show_phase(motion: BVHMotion, rToe_c, lToe_c, phase_info):
    joint_trans, joint_orient = motion.batch_forward_kinematics()
    motion_len = motion.motion_length
    rToe = motion.joint_name.index("rToeJoint")
    lToe = motion.joint_name.index("lToeJoint")
    print("rToe_num: %d, lToe_num: %d" % (rToe, lToe))
    rToe_pos = joint_trans[:, rToe, 1]
    lToe_pos = joint_trans[:, lToe, 1]
    lToe_min = np.min(lToe_pos)
    rToe_vel = joint_trans[1:, rToe, 1] - joint_trans[:-1, rToe, 1]
    lToe_vel = joint_trans[1:, lToe, 1] - joint_trans[:-1, lToe, 1]
    # phase = map(lambda x, y: 1 if np.abs(x - lToe_min)
    #             < 1e-9 else 0, lToe_pos, lToe_vel)
    # phase = np.array(list(phase))
    phase = np.zeros_like(rToe_pos)
    phase_nodes = []
    for start in range(0, motion_len, phase_info[1]):
        for p in phase_info[0]:
            phase_nodes.append(start + p)
    print(phase_nodes)
    for idx, node in enumerate(phase_nodes[1:], 1):
        phase[phase_nodes[idx - 1]:node] = \
            np.linspace(0, 2 * np.pi, node - phase_nodes[idx - 1])
    curve = np.linspace(0, 2 * np.pi,
                        phase_info[0][0] + phase_info[1] - phase_info[0][-1])
    phase[phase_nodes[-1]:] = curve[:phase.shape[0] - phase_nodes[-1]]
    phase[:phase_nodes[0]] = curve[phase.shape[0] - phase_nodes[-1]:]
    # plt.plot(motion.joint_position[:, 0, 0], motion.joint_position[:, 0, 2])
    # plt.show()
    # fig, axs = plt.subplots(4, 1)
    # for p in phase_info[0]:
    #     for i in range(p, motion_len, phase_info[1]):
    #         lmin = np.min(lToe_pos)
    #         lmax = np.max(lToe_pos)
    #         axs[0].plot(np.array([i, i]), np.array([lmin, lmax]))
    # axs[0].plot(rToe_pos, label="rToe_pos")
    # axs[0].plot(lToe_pos, label="lToe_pos")
    # axs[0].plot(np.zeros_like(rToe_pos), label="base", linestyle='--')
    # axs[0].set_title("position")
    # axs[0].legend(loc="upper right")
    # axs[1].plot(rToe_vel, label="rToe_vel")
    # axs[1].plot(lToe_vel, label="lToe_vel")
    # axs[1].plot(np.zeros_like(rToe_vel), label="base", linestyle='--')
    # axs[1].set_title("velocity")
    # axs[1].legend(loc="upper right")
    # axs[2].plot(phase)
    # axs[3].plot(rToe_c, label="rToe contact")
    # axs[3].plot(lToe_c, label="lToe contact")
    # axs[3].legend(loc="upper right")
    # axs[0].set_title("contact")
    # plt.show()
    return phase


def compute_contact(motion: BVHMotion):
    joint_trans, joint_orient = motion.batch_forward_kinematics()
    rToe = motion.joint_name.index("rToeJoint")
    lToe = motion.joint_name.index("lToeJoint")
    rToe_pos = joint_trans[:, rToe, 1]
    lToe_pos = joint_trans[:, lToe, 1]
    rToe_c = np.zeros(motion.motion_length)
    lToe_c = np.zeros(motion.motion_length)
    for i in range(motion.motion_length):
        if rToe_pos[i] < 0:
            rToe_c[i] = 1
        if lToe_pos[i] < 0:
            lToe_c[i] = 1
    return rToe_c, lToe_c


def make_dataset(motion: BVHMotion, phase, label: np.ndarray):
    motion_len = motion.motion_length
    trajectory_pos = motion.joint_position[:, 0, [0, 2]]
    trajectory_dir = R.from_quat(motion.joint_rotation[:, 0]).apply(
        np.array([0, 0, 1]))[:, [0, 2]]
    label = np.expand_dims(label, axis=0)
    label = np.repeat(label, motion_len, axis=0)
    assert (label.shape == (motion_len, 3))
    local_motion = motion.raw_copy()
    local_motion.joint_position[:, 0] = np.array([0, 0, 0])
    local_motion.joint_rotation[:, 0] = R.from_euler(
        'zxy', np.array([0, 0, 0])).as_quat()
    local_trans, _ = local_motion.batch_forward_kinematics()
    # local_orient = local_motion.joint_rotation
    local_vel = np.zeros_like(local_trans)
    local_rot = local_motion.joint_rotation
    local_orient_rotvec = np.zeros_like(local_trans)
    local_vel[1:] = local_trans[1:, ...] - local_trans[:-1, ...]
    root_avel = np.zeros(motion_len)
    root_vel = np.zeros_like((motion_len, 2))
    for i in range(0, motion_len):
        local_orient_rotvec[i] = R.from_quat(
            local_rot[i, ...]).as_rotvec()
    local_orient = local_orient_rotvec
    root_avel = (R.from_quat(motion.joint_rotation[1:, 0])
                 * R.from_quat(motion.joint_rotation[:-1, 0]).inv()).as_euler('zxy')[:, 2]
    root_vel = motion.joint_position[1:, 0, [
        0, 2]] - motion.joint_position[:-1, 0, [0, 2]]
    x = []
    y = []
    pp = []
    for i in range(1, motion_len - 1):
        j = i + 1
        k = i - 1
        if i < 30:
            tp0 = np.concatenate([np.zeros([(30-i)//5, 2]),
                                  trajectory_pos[:i:5]], axis=0)
            td0 = np.concatenate([np.zeros([(30-i)//5, 2]),
                                  trajectory_dir[:i:5]], axis=0)
            tg0 = np.concatenate([np.zeros([(30-i)//5, 3]),
                                  label[:i:5]], axis=0)
            tg0[:(30-i)//5, 0] = 1
        else:
            tp0 = trajectory_pos[(i-30):i:5]
            td0 = trajectory_dir[(i-30):i:5]
            tg0 = label[(i-30):i:5]
        if motion_len - i < 30:
            tp1 = np.concatenate([trajectory_pos[i::5],
                                  np.zeros([(30-motion_len+i)//5, 2])], axis=0)
            td1 = np.concatenate([trajectory_dir[i::5],
                                  np.zeros([(30-motion_len+i)//5, 2])], axis=0)
            tg1 = np.concatenate([label[i::5],
                                  np.zeros([(30-motion_len+i)//5, 3])], axis=0)
            tg1[(30-motion_len+i)//5:, 0] = 1
        else:
            tp1 = trajectory_pos[i:(i+30):5]
            td1 = trajectory_dir[i:(i+30):5]
            tg1 = label[i:(i+30):5]
        tp = np.concatenate([tp0, tp1], axis=0).reshape((1, -1))[0]
        td = np.concatenate([td0, td1], axis=0).reshape((1, -1))[0]
        tg = np.concatenate([tg0, tg1], axis=0).reshape((1, -1))[0]
        lp = local_trans[k].reshape((1, -1))[0]
        lv = local_vel[k].reshape((1, -1))[0]
        p = phase[k].reshape((1, -1))[0]
        assert (tp.shape == (24,))
        assert (td.shape == (24,))
        assert (tg.shape == (36,))
        assert (lp.shape == (len(motion.joint_name) * 3,))
        assert (lv.shape == (len(motion.joint_name) * 3,))
        x0 = []
        x0.append(tp)  # all the positions (x,y) of the total 12 frames, [0:24]
        # all the orientations (x,y) of the total 12 frames, [24:48]
        x0.append(td)
        # all the gait information (bool, bool, bool) of the total 12 frames, [48:84]
        x0.append(tg)
        # all the local positions of the joints in current frame, [84:159]
        x0.append(lp)
        # all the local velocities of the joints in current frame, [159:234]
        x0.append(lv)
        x.append(x0)

        if j < 30:
            tp0 = np.concatenate([np.zeros([(30-j)//5, 2]),
                                  trajectory_pos[:j:5]], axis=0)
            td0 = np.concatenate([np.zeros([(30-j)//5, 2]),
                                  trajectory_dir[:j:5]], axis=0)
        else:
            tp0 = trajectory_pos[(j-30):j:5]
            td0 = trajectory_dir[(j-30):j:5]
        if motion_len - j < 30:
            tp1 = np.concatenate([trajectory_pos[j::5],
                                  np.zeros([(30-motion_len+j)//5, 2])], axis=0)
            td1 = np.concatenate([trajectory_dir[j::5],
                                  np.zeros([(30-motion_len+j)//5, 2])], axis=0)
        else:
            tp1 = trajectory_pos[j:(j+30):5]
            td1 = trajectory_dir[j:(j+30):5]
        tpj = np.concatenate([tp0, tp1], axis=0).reshape((1, -1))[0]
        tdj = np.concatenate([td0, td1], axis=0).reshape((1, -1))[0]
        lpj = local_trans[i].reshape((1, -1))[0]
        lvj = local_vel[i].reshape((1, -1))[0]
        laj = local_orient[i].reshape((1, -1))[0]
        pj = phase[i].reshape((1, -1))[0]
        ra = root_avel[i]
        facing_dir = R.from_quat(motion.joint_rotation[i, 0]).apply(
            np.array([0, 0, 1])).flatten()[[0, 2]]
        side_dir = R.from_quat(motion.joint_rotation[i, 0]).apply(
            np.array([1, 0, 0])).flatten()[[0, 2]]
        rz = np.dot(root_vel[i], facing_dir)
        rx = np.dot(root_vel[i], side_dir)
        y0 = []
        # all the positions (x,y) of the total 12 frames, [0:24]
        y0.append(tpj)
        # all the orientations (x,y) of the total 12 frames, [24:48]
        y0.append(tdj)
        # all the local positions of the joints in current frame, [48:123]
        y0.append(lpj)
        # all the local velocities of the joints in current frame, [123:198]
        y0.append(lvj)
        # all the local angle of the joints in current frame, [198:273]
        y0.append(laj)
        # the local x velocity of root joint in current frame, [273]
        y0.append(np.array([rx]))
        # the local z velocity of root joint in current frame, [274]
        y0.append(np.array([rz]))
        # the angular velocity of root joint in current frame, [275]
        y0.append(np.array([ra]))
        y0.append(pj - p)  # the phase change in current frame, [276]
        y.append(y0)

        pp.append(p)
    return [x, y, pp]


if __name__ == '__main__':
    idle_motion = BVHMotion("motion_material/idle.bvh")
    walk_motion = BVHMotion("motion_material/walk_forward.bvh")
    run_motion = BVHMotion("motion_material/run_forward.bvh")
    turn_left_motion = BVHMotion("motion_material/walk_and_turn_left.bvh")
    turn_right_motion = BVHMotion("motion_material/walk_and_turn_right.bvh")
    motions = [
        idle_motion, walk_motion, run_motion,
        turn_left_motion, turn_right_motion
    ]
    phases = [
        [[30], 200],
        [[2], 100],
        [[41], 45],
        [[20, 143, 225, 313], 420],
        [[55, 100, 165], 197]
    ]
    joint_name = idle_motion.joint_name
    for idx, motion in enumerate(motions):
        motion.adjust_joint_name(joint_name)
        motion = build_loop_motion(motion)
        new_motion = motion.raw_copy()
        t = 3
        for i in range(t):
            next_pos = motion.joint_position[-1, 0, [0, 2]]
            next_facing = R.from_quat(motion.joint_rotation[-1, 0]).apply(
                np.array([0, 0, 1])).flatten()[[0, 2]]
            next_motion = new_motion.translation_and_rotation(
                0, next_pos, next_facing)
            motion.append(next_motion)
        motions[idx] = motion
    phase_curves = []
    for idx, motion in enumerate(motions):
        print(motion.motion_length)
        rToe_c, lToe_c = compute_contact(motion)
        phase_curves.append(
            show_phase(motion, rToe_c, lToe_c, phases[idx]))
    datasets = []
    datasets.append(make_dataset(
        motions[0], phase_curves[0], np.array([1, 0, 0])))
    datasets.append(make_dataset(
        motions[1], phase_curves[1], np.array([0, 1, 0])))
    datasets.append(make_dataset(
        motions[2], phase_curves[2], np.array([0, 0, 1])))
    datasets.append(make_dataset(
        motions[3], phase_curves[3], np.array([0, 1, 0])))
    datasets.append(make_dataset(
        motions[4], phase_curves[4], np.array([0, 1, 0])))
    f = h5py.File("pfnn_dataset.h5", "w")
    x = []
    y = []
    p = []
    for i, dataset in enumerate(datasets):
        for j, record in enumerate(dataset[0]):
            record = np.concatenate(record, axis=0)
            dataset[0][j] = record
        for j, record in enumerate(dataset[1]):
            record = np.concatenate(record, axis=0)
            dataset[1][j] = record
        x.append(dataset[0])
        y.append(dataset[1])
        p.append(dataset[2])
    x_dset = np.concatenate(x, axis=0).astype(np.float32)
    y_dset = np.concatenate(y, axis=0).astype(np.float32)
    p_dset = np.concatenate(p, axis=0).astype(np.float32)

    x_mean = x_dset.mean(axis=0)
    print("x_mean shape: ", x_mean.shape)
    y_mean = y_dset.mean(axis=0)

    x_std = x_dset.std(axis=0)
    print("x_std shape: ", x_std.shape)
    y_std = y_dset.std(axis=0)

    x_w = np.ones_like(x_dset)
    x_w[84:] = 0.1
    y_w = np.ones_like(y_dset)

    # j_w = np.ones(25) * 0.1

    # w = 12 # frame num
    # j = 25 # joint num
    # x_mean[w*0:w*2:2] = np.mean(x_mean[w*0:w*2:2]) # mean of all 12 frame x positions
    # x_mean[w*0+1:w*2:2] = np.mean(x_mean[w*0+1:w*2:2]) # mean of all 12 frame z positions
    # x_mean[w*2:w*4:2] = np.mean(x_mean[w*2:w*4:2]) # mean of all 12 frame x orientations
    # x_mean[w*2+1:w*4:2] = np.mean(x_mean[w*2+1:w*4:2]) # mean of all 12 frame z orientations
    # x_mean[w*4:w*7] = np.mean(x_mean[w*4:w*7]) # mean of all 12 frame gait info
    # x_mean[w*7]

    x_dset = (x_dset - x_mean) / (x_std + 1e-8)
    x_dset = x_dset * x_w
    y_dset = (y_dset - y_mean) / (y_std + 1e-8)
    y_dset = y_dset * y_w

    x_dset.dtype = np.float32
    y_dset.dtype = np.float32

    print(x_dset.shape)
    print(y_dset.shape)
    print(p_dset.shape)
    grp = f.create_group("training")
    grp["input"] = x_dset
    grp["output"] = y_dset
    grp["phase"] = p_dset
