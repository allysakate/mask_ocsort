import math
import numpy as np


def determine_focal_length(vps, image):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    forcal = []
    if len(vps) != 2:
        for i in range(len(vps)):
            if i == 2:
                j = 0
            else:
                j = i + 1
            if vps[i][0] - vps[j][0] == 0:
                return math.fabs(pp[0] - vps[j][0])
            if vps[i][1] - vps[j][1] == 0:
                return math.fabs(pp[1] - vps[j][1])
            k_uv = (vps[i][1] - vps[j][1]) / (vps[i][0] - vps[j][0])
            b_uv = vps[j][1] - k_uv * vps[j][0]
            pp_uv = math.fabs(k_uv * pp[0] - pp[1] + b_uv) / math.pow(
                k_uv * k_uv + 1, 0.5
            )
            length_uv = math.sqrt(
                (vps[i][1] - vps[j][1]) ** 2 + (vps[i][0] - vps[j][0]) ** 2
            )
            length_pu = math.sqrt((vps[i][1] - pp[1]) ** 2 + (vps[i][0] - pp[0]) ** 2)
            up_uv = math.sqrt(length_pu**2 - pp_uv**2)
            vp_uv = abs(length_uv - up_uv)
            forcal.append(math.sqrt(abs(up_uv * vp_uv - (pp_uv) ** 2)))
    else:
        if vps[0][0] - vps[1][0] == 0:
            return math.fabs(pp[0] - vps[j][0])
        if vps[0][1] - vps[1][1] == 0:
            return math.fabs(pp[1] - vps[j][1])
        k_uv = (vps[0][1] - vps[1][1]) / (vps[0][0] - vps[1][0])
        b_uv = vps[1][1] - k_uv * vps[1][0]
        pp_uv = math.fabs(k_uv * pp[0] - pp[1] + b_uv) / math.pow(k_uv * k_uv + 1, 0.5)
        length_uv = math.sqrt(
            (vps[0][1] - vps[1][1]) ** 2 + (vps[0][0] - vps[1][0]) ** 2
        )
        length_pu = math.sqrt((vps[0][1] - pp[1]) ** 2 + (vps[0][0] - pp[0]) ** 2)
        up_uv = math.sqrt(length_pu**2 - pp_uv**2)
        vp_uv = abs(length_uv - up_uv)
        forcal.append(math.sqrt((up_uv * vp_uv) - ((pp_uv) ** 2)))
    return forcal


def calculate_rotation_matrix(vps, image, f):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    M_r_o2c = []
    u = np.array([vps[0][0] - pp[0], vps[0][1] - pp[1], f])
    u_norm = u / np.sqrt((u * u).sum())
    v = np.array([vps[1][0] - pp[0], vps[1][1] - pp[1], f])
    v_norm = v / np.sqrt((v * v).sum())
    w_norm = np.cross(u_norm, v_norm)
    M_r_o2c.append(np.c_[u_norm, v_norm, w_norm])
    return M_r_o2c[0]


def calculate_translation_vector_h(image, f, M_r_o2c, px_x, px_y, h):
    V_t_o2c = []
    dpi = 3779.5  # px/m  96dpi
    pp = np.array([image.shape[1] / 2, image.shape[0] / 2, 0]) / dpi
    xy_img = np.array([1209, 2382, f]) / dpi
    xy_c = xy_img - pp
    length_xy_c = np.sqrt((xy_c * xy_c).sum())
    xy_c_norm = xy_c / np.sqrt((xy_c * xy_c).sum())
    # h_c = np.dot(M_r_o2c, np.array([0, 0, -h]))
    h_c = np.dot([M_r_o2c], np.array([0, 0, h]))
    alpha = math.acos(np.dot(h_c, xy_c) / (length_xy_c * h))
    oxy_c = (h / math.cos(alpha)) * xy_c_norm
    print(math.cos(alpha))
    V_t_o2c.append(np.dot(np.transpose(M_r_o2c), oxy_c))
    return V_t_o2c[0]
