from bvh_utils import *
# ---------------你的代码------------------#
# translation 和 orientation 都是全局的


def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()
    joint_T_translation = T_pose_joint_translation.copy()
    # ---------------你的代码------------------#
    n = vertex_translation.shape[0]  # number of vertices
    m = joint_translation.shape[0]  # number of joints

    # 1. 求出顶点到各相关关节的距离
    vertex_joint_dist = np.repeat(
        np.expand_dims(vertex_translation, axis=1), 4, axis=1)\
        - joint_T_translation[skinning_idx]  # (n, 4, 3)

    # 2. 求出顶点的在目标姿态下通过各相关关节计算出的位置
    joint_orientation_matrix = R.from_quat(
        joint_orientation).as_matrix()  # (m, 3, 3)
    vertex_joint_dist_expand = np.expand_dims(
        vertex_joint_dist, axis=-1)  # (n, 4, 3, 1)
    new_vertex_translation = np.matmul(joint_orientation_matrix[skinning_idx],
                                       vertex_joint_dist_expand) + np.expand_dims(
                                           joint_translation[skinning_idx], axis=-1)  # (n, 4, 3, 1)
    new_vertex_translation = np.squeeze(
        new_vertex_translation, axis=-1)  # (n, 4, 3)

    # 3. 求出顶点位置加权平均后的位置
    skinning_weight_expand = np.expand_dims(
        skinning_weight, axis=1)  # (n, 1, 4)
    vertex_translation_expand = np.matmul(
        skinning_weight_expand, new_vertex_translation)  # (n, 1, 3)
    vertex_translation = np.squeeze(
        vertex_translation_expand, axis=1)  # (n, 3)

    return vertex_translation
