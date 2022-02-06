import numpy as np


def get_dolly_zoom_z(initial_aov, camera_z, initial_z, new_aov):
    # See: https://en.wikipedia.org/wiki/Dolly_zoom#Calculating_distances.
    width = (camera_z - initial_z) * 2 * np.tan(np.radians(initial_aov / 2))
    camera_distance = width / (2 * np.tan(np.radians(new_aov / 2)))
    # The object's new z after doing a dolly zoom.
    return camera_z - camera_distance


def gen_single_angle_rotation_matrix(which_angle, angle):
    if which_angle == "yaw":
        (first_idx, second_idx) = (0, 2)
        negs = np.array([1.0, 1.0, -1.0, 1.0])
    elif which_angle == "pitch":
        (first_idx, second_idx) = (1, 2)
        negs = np.array([1.0, -1.0, 1.0, 1.0])
    elif which_angle == "roll":
        (first_idx, second_idx) = (0, 1)
        negs = np.array([1.0, -1.0, 1.0, 1.0])

    R = np.eye(3)
    R[first_idx, first_idx] = negs[0] * np.cos(angle)
    R[first_idx, second_idx] = negs[1] * np.sin(angle)
    R[second_idx, first_idx] = negs[2] * np.sin(angle)
    R[second_idx, second_idx] = negs[3] * np.cos(angle)
    return R


def gen_rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    R_yaw = gen_single_angle_rotation_matrix("yaw", yaw)
    R_pitch = gen_single_angle_rotation_matrix("pitch", pitch)
    R_roll = gen_single_angle_rotation_matrix("roll", roll)
    return R_yaw @ R_pitch @ R_roll


def gen_rotation_matrix_from_azim_elev_in_plane(
    azimuth=0.0, elevation=0.0, in_plane=0.0
):
    # See: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function.
    y = np.sin(elevation)
    radius = np.cos(elevation)
    x = radius * np.sin(azimuth)
    z = radius * np.cos(azimuth)

    cam_from = np.array([x, y, z])
    cam_to = np.zeros(3)
    tmp = np.array([0.0, 1.0, 0.0])

    diff = cam_from - cam_to
    forward = diff / np.linalg.norm(diff)
    crossed = np.cross(tmp, forward)
    right = crossed / np.linalg.norm(crossed)
    up = np.cross(forward, right)

    R = np.stack([right, up, forward])
    R_in_plane = gen_single_angle_rotation_matrix("roll", in_plane)
    return R_in_plane @ R


def get_yaw_from_rotation_matrix(R):
    return np.arctan2(R[0, 2], R[2, 2])


def get_pitch_from_rotation_matrix(R):
    return np.arctan2(-R[1, 2], np.sqrt(R[1, 0] ** 2 + R[1, 1] ** 2))


def get_roll_from_rotation_matrix(R):
    return np.arctan2(R[1, 0], R[1, 1])


def get_yaw_pitch_roll_from_matrix(R):
    yaw = get_yaw_from_rotation_matrix(R)
    pitch = get_pitch_from_rotation_matrix(R)
    roll = get_roll_from_rotation_matrix(R)
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def get_azim_from_matrix(R):
    return np.arctan2(R[2, 0], R[2, 2])


def get_elev_from_matrix(R):
    return np.arcsin(R[2, 1])


def get_azim_elev_from_matrix(R):
    # WARNING: only works when rotation matrix was generated using azimuth and elevation
    # angles.
    azimuth = get_azim_from_matrix(R)
    elevation = get_elev_from_matrix(R)
    return (azimuth, elevation)


def get_rotation_matrix_from_axis_angle(axis, angle):
    # See: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    # and: http://www.songho.ca/opengl/gl_anglestoaxes.html. "RyRxRz" --> (yaw, pitch, roll)
    c = np.cos(angle)
    s = np.sin(angle)
    (ux, uy, uz) = (axis[0], axis[1], axis[2])
    x_col = np.array(
        [
            [c + (ux ** 2) * (1 - c)],
            [uy * ux * (1 - c) + uz * s],
            [uz * ux * (1 - c) - uy * s],
        ]
    )
    y_col = np.array(
        [
            [ux * uy * (1 - c) - uz * s],
            [c + (uy ** 2) * (1 - c)],
            [uz * uy * (1 - c) + ux * s],
        ]
    )
    z_col = np.array(
        [
            [ux * uz * (1 - c) + uy * s],
            [uy * uz * (1 - c) - ux * s],
            [c + (uz ** 2) * (1 - c)],
        ]
    )
    return np.hstack((x_col, y_col, z_col))
