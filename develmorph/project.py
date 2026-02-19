import numpy as np


class CollisionConstraint:
    """
    XPBD inequality constraint: |p_i - p_j| >= rest_length.

    Only applies correction when particles overlap (C < 0).
    Modifies positions in-place.
    """

    def __init__(self, i: int, j: int, rest_length: float, compliance: float = 0.0):
        self.i = i
        self.j = j
        self.rest_length = rest_length
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        dx = x[self.i] - x[self.j]
        dist = np.linalg.norm(dx)

        if dist < 1e-12:
            return lambda_acc

        C = dist - self.rest_length

        # Inequality: only correct overlap (C < 0)
        if C >= 0.0:
            return lambda_acc

        grad = dx / dist
        w_sum = inv_masses[self.i] + inv_masses[self.j]

        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0

        denom = w_sum + alpha_tilde
        if denom < 1e-12:
            return lambda_acc

        delta_lambda = (-C - alpha_tilde * lambda_acc) / denom
        lambda_acc += delta_lambda

        x[self.i] += delta_lambda * inv_masses[self.i] * grad
        x[self.j] -= delta_lambda * inv_masses[self.j] * grad

        return lambda_acc


class ShellConstraint:
    """
    XPBD inequality constraint: particle stays inside ellipsoidal boundary.

    Effective boundary is shrunk by cell radius so the cell surface
    (not just center) remains inside. Constraint value:
        C(x) = (px/(a-r))^2 + (py/(b-r))^2 + (pz/(c-r))^2 - 1
    Only corrects when C > 0 (outside).
    Modifies positions in-place.
    """

    def __init__(
        self,
        i: int,
        a: float,
        b: float,
        c: float,
        radius: float = 0.0,
        compliance: float = 0.0,
    ):
        self.i = i
        self.ea = max(a - radius, 0.1)
        self.eb = max(b - radius, 0.1)
        self.ec = max(c - radius, 0.1)
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        p = x[self.i]
        sx = p[0] / self.ea
        sy = p[1] / self.eb
        sz = p[2] / self.ec

        C = sx * sx + sy * sy + sz * sz - 1.0

        # Inequality: only correct when outside (C > 0)
        if C <= 0.0:
            return lambda_acc

        # Gradient of C w.r.t. position
        grad = 2.0 * np.array(
            [
                p[0] / (self.ea * self.ea),
                p[1] / (self.eb * self.eb),
                p[2] / (self.ec * self.ec),
            ]
        )
        grad_norm_sq = np.dot(grad, grad)

        if grad_norm_sq < 1e-12:
            return lambda_acc

        w = inv_masses[self.i]
        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0

        denom = w * grad_norm_sq + alpha_tilde
        if denom < 1e-12:
            return lambda_acc

        delta_lambda = (-C - alpha_tilde * lambda_acc) / denom
        lambda_acc += delta_lambda

        x[self.i] += delta_lambda * w * grad

        return lambda_acc
