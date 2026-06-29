import torch
import torch.nn.functional as F


def _sqrt_positive_part(x):
    return torch.sqrt(torch.clamp(x, min=0.0))


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / torch.clamp((quaternions * quaternions).sum(-1), min=1e-12)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = _sqrt_positive_part(
        torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    )
    cand = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], -1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
        ],
        -2,
    )
    return cand[F.one_hot(q_abs.argmax(-1), 4).bool(), :].reshape(matrix.shape[:-2] + (4,)) / (2 * q_abs.max(-1).values[..., None].clamp(min=0.1))


def axis_angle_to_quaternion(axis_angle):
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half = 0.5 * angles
    small = angles.abs() < 1e-6
    sin_over_angle = torch.where(small, 0.5 - angles * angles / 48.0, torch.sin(half) / angles)
    return torch.cat([torch.cos(half), axis_angle * sin_over_angle], -1)


def quaternion_to_axis_angle(quaternions):
    q = F.normalize(quaternions, dim=-1)
    norms = torch.linalg.norm(q[..., 1:], dim=-1, keepdim=True)
    half = torch.atan2(norms, q[..., :1])
    angles = 2 * half
    small = angles.abs() < 1e-6
    scale = torch.where(small, 2.0 + angles * angles / 12.0, angles / norms.clamp(min=1e-12))
    return q[..., 1:] * scale


def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


def _axis_angle_rotation(axis, angle):
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    c = torch.cos(angle)
    s = torch.sin(angle)
    if axis == 'X':
        flat = (one, zero, zero, zero, c, -s, zero, s, c)
    elif axis == 'Y':
        flat = (c, zero, s, zero, one, zero, -s, zero, c)
    elif axis == 'Z':
        flat = (c, -s, zero, s, c, zero, zero, zero, one)
    else:
        raise ValueError(f'Invalid axis {axis}')
    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention):
    if len(convention) != 3:
        raise ValueError('Convention must have 3 letters')
    matrices = [_axis_angle_rotation(c, euler_angles[..., i]) for i, c in enumerate(convention)]
    return matrices[0] @ matrices[1] @ matrices[2]


def matrix_to_euler_angles(matrix, convention):
    if convention != 'XYZ':
        raise NotImplementedError('Only XYZ euler inverse is implemented in this local shim')
    sy = torch.clamp(matrix[..., 0, 2], -1.0, 1.0)
    y = torch.asin(sy)
    x = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    z = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    return torch.stack((x, y, z), -1)
