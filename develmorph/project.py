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


class AdhesionConstraint:
    """
    XPBD inequality constraint for cell-cell adhesion: |p_i - p_j| <= rest_length.

    Only corrects when cells are too far apart (C > 0, attraction).
    Overlap (C < 0) is handled by CollisionConstraint separately.
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

        # Inequality: only correct when too far apart (C > 0)
        if C <= 0.0:
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


class VolumeConstraint:
    """
    XPBD equality constraint: mesh volume = target_volume.

    Uses signed volume of triangulated surface mesh:
        V = (1/6) * sum( dot(p0, cross(p1, p2)) )

    Compliance ~= 0 for incompressible (liquid cell interior).
    """

    def __init__(
        self, center_idx, surface_indices, triangles, target_volume, compliance=0.0
    ):
        self.center_idx = center_idx
        self.surface_indices = list(surface_indices)
        self.triangles = triangles  # list of (i, j, k) global indices
        self.target_volume = target_volume
        self.compliance = compliance
        self.all_indices = [center_idx] + list(surface_indices)

    def _compute_volume(self, x):
        """Compute signed volume of the triangulated mesh."""
        vol = 0.0
        for i0, i1, i2 in self.triangles:
            vol += np.dot(x[i0], np.cross(x[i1], x[i2]))
        return vol / 6.0

    def _compute_gradients(self, x):
        """Compute gradient of volume w.r.t. each particle position."""
        grads = {}
        for idx in self.all_indices:
            grads[idx] = np.zeros(3)

        for i0, i1, i2 in self.triangles:
            # dV/dp0 = (1/6) * cross(p1, p2)
            # dV/dp1 = (1/6) * cross(p2, p0)
            # dV/dp2 = (1/6) * cross(p0, p1)
            grads[i0] += np.cross(x[i1], x[i2]) / 6.0
            grads[i1] += np.cross(x[i2], x[i0]) / 6.0
            grads[i2] += np.cross(x[i0], x[i1]) / 6.0

        return grads

    def project(self, x, inv_masses, dt, lambda_acc):
        V = self._compute_volume(x)

        # Guard: if mesh has inverted (negative volume), skip to avoid
        # pushing it further inside-out. Distance constraints will
        # restore shape first.
        if V <= 0.0 and self.target_volume > 0.0:
            return lambda_acc

        C = V - self.target_volume

        if abs(C) < 1e-12:
            return lambda_acc

        grads = self._compute_gradients(x)

        w_grad_sq = 0.0
        for idx in self.all_indices:
            w_grad_sq += inv_masses[idx] * np.dot(grads[idx], grads[idx])

        if w_grad_sq < 1e-12:
            return lambda_acc

        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0

        denom = w_grad_sq + alpha_tilde
        if denom < 1e-12:
            return lambda_acc

        delta_lambda = (-C - alpha_tilde * lambda_acc) / denom

        # Clamp: limit max displacement per particle to a fraction of
        # the characteristic length (cube root of target volume)
        char_len = abs(self.target_volume) ** (1.0 / 3.0)
        max_disp = 0.01 * char_len
        scale = 1.0
        for idx in self.all_indices:
            disp = abs(delta_lambda) * inv_masses[idx] * np.linalg.norm(grads[idx])
            if disp > 1e-12:
                s = max_disp / disp
                if s < scale:
                    scale = s
        delta_lambda *= scale

        lambda_acc += delta_lambda

        for idx in self.all_indices:
            x[idx] += delta_lambda * inv_masses[idx] * grads[idx]

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


class SurfacePenetrationConstraint:
    """
    XPBD inequality constraint: surface particle stays outside another cell's sphere.

    Pushes a single surface particle (surf_idx) out of another cell's nominal sphere
    (centered at other_center_idx with radius min_dist). ONE-SIDED: only surf_idx
    moves, other_center_idx is governed by CollisionConstraint separately.

    Constraint value:
        C = ||x[surf_idx] - x[other_center_idx]|| - min_dist
    Only corrects when C < 0 (penetrating).
    """

    def __init__(
        self,
        surf_idx: int,
        other_center_idx: int,
        min_dist: float,
        compliance: float = 0.0,
    ):
        self.surf_idx = surf_idx
        self.other_center_idx = other_center_idx
        self.min_dist = min_dist
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_prev):
        diff = x[self.surf_idx] - x[self.other_center_idx]
        dist = np.linalg.norm(diff)
        C = dist - self.min_dist
        if C >= 0 or dist < 1e-9:
            return lambda_prev
        n = diff / dist
        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0
        w = inv_masses[self.surf_idx]
        dlambda = (-C - alpha_tilde * lambda_prev) / (w + alpha_tilde)
        x[self.surf_idx] += w * dlambda * n
        return lambda_prev + dlambda


class HalfSpaceConstraint:
    """
    XPBD inequality constraint: surface particle stays on cell i's side of the
    contact plane between cells i and j.

    Contact plane: unit normal n = (cj-ci)/||cj-ci|| pointing from ci toward cj,
    passing through midpoint M = (ci+cj)/2.

    C = -dot(x[surf_idx] - M, n)
    C >= 0  =>  particle is on cell i's side  (correct, no correction)
    C <  0  =>  particle has crossed into cell j's half-space (fire constraint)

    Only surf_idx moves; the correction direction is always -n (contact normal),
    so there is no directional bias regardless of where on the surface the
    particle sits.  This produces a flat contact interface naturally.
    """

    def __init__(self, surf_idx: int, M, n, compliance: float = 0.0):
        self.surf_idx = surf_idx
        self.M = np.asarray(M, dtype=float)
        self.n = np.asarray(n, dtype=float)  # unit normal ci -> cj
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_prev):
        C = -np.dot(x[self.surf_idx] - self.M, self.n)
        if C >= 0:
            return lambda_prev
        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0
        w = inv_masses[self.surf_idx]
        dlambda = (-C - alpha_tilde * lambda_prev) / (w + alpha_tilde)
        x[self.surf_idx] += w * dlambda * (-self.n)
        return lambda_prev + dlambda


class DistanceConstraint:
    """
    XPBD equality constraint: |p_i - p_j| = rest_length.

    Unlike CollisionConstraint (inequality), this corrects in both directions:
    pushes apart when too close AND pulls together when too far.
    Used for membrane distance maintenance.
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


class TriangleAreaConstraint:
    """
    XPBD inequality constraint: triangle area <= target_area.

    Fires only when A > A_target (stretch prevention).
    Compression is permitted — membranes can fold but not stretch.

    Indices i0, i1, i2 are surface particle indices (center excluded).
    """

    def __init__(
        self, i0: int, i1: int, i2: int, target_area: float, compliance: float = 0.0
    ):
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2
        self.target_area = target_area
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        p0 = x[self.i0]
        p1 = x[self.i1]
        p2 = x[self.i2]
        n_unnorm = np.cross(p1 - p0, p2 - p0)
        area2 = np.linalg.norm(n_unnorm)
        if area2 < 1e-12:
            return lambda_acc

        A = 0.5 * area2
        C = A - self.target_area

        # Inequality: only correct when area exceeds target
        if C <= 0.0:
            return lambda_acc

        n_hat = n_unnorm / area2
        grad0 = 0.5 * np.cross(p1 - p2, n_hat)
        grad1 = 0.5 * np.cross(p2 - p0, n_hat)
        grad2 = 0.5 * np.cross(p0 - p1, n_hat)

        w_grad_sq = (
            inv_masses[self.i0] * np.dot(grad0, grad0)
            + inv_masses[self.i1] * np.dot(grad1, grad1)
            + inv_masses[self.i2] * np.dot(grad2, grad2)
        )
        if w_grad_sq < 1e-12:
            return lambda_acc

        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0
        delta_lambda = (-C - alpha_tilde * lambda_acc) / (w_grad_sq + alpha_tilde)
        lambda_acc += delta_lambda

        x[self.i0] += delta_lambda * inv_masses[self.i0] * grad0
        x[self.i1] += delta_lambda * inv_masses[self.i1] * grad1
        x[self.i2] += delta_lambda * inv_masses[self.i2] * grad2

        return lambda_acc


class TetrahedralVolumeConstraint:
    """
    XPBD equality constraint: volume of tetrahedron(center, p0, p1, p2) = target.

    Each surface triangle of a cell, together with the cell center, forms a
    tetrahedron. Preserving each tetrahedron's volume independently produces
    a local pressure model where cytoplasm pushes outward on each membrane patch.

    Constraint: C = V - target_volume = 0

    Volume formula:
        a = p0 - pc, b = p1 - pc, c = p2 - pc
        V = dot(a, cross(b, c)) / 6

    Gradients:
        dV/dp0 = cross(b, c) / 6
        dV/dp1 = cross(c, a) / 6
        dV/dp2 = cross(a, b) / 6
        dV/dpc = -(dV/dp0 + dV/dp1 + dV/dp2)
    """

    def __init__(
        self, center_idx: int, i0: int, i1: int, i2: int,
        target_volume: float, compliance: float = 0.0
    ):
        self.center_idx = center_idx
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2
        self.target_volume = target_volume
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        pc = x[self.center_idx]
        p0, p1, p2 = x[self.i0], x[self.i1], x[self.i2]
        a, b, c = p0 - pc, p1 - pc, p2 - pc

        V = np.dot(a, np.cross(b, c)) / 6.0
        C = V - self.target_volume

        if abs(C) < 1e-12:
            return lambda_acc

        g0 = np.cross(b, c) / 6.0
        g1 = np.cross(c, a) / 6.0
        g2 = np.cross(a, b) / 6.0
        gc = -(g0 + g1 + g2)

        w_grad_sq = (
            inv_masses[self.center_idx] * np.dot(gc, gc)
            + inv_masses[self.i0] * np.dot(g0, g0)
            + inv_masses[self.i1] * np.dot(g1, g1)
            + inv_masses[self.i2] * np.dot(g2, g2)
        )

        if w_grad_sq < 1e-12:
            return lambda_acc

        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0
        delta_lambda = (-C - alpha_tilde * lambda_acc) / (w_grad_sq + alpha_tilde)
        lambda_acc += delta_lambda

        x[self.center_idx] += delta_lambda * inv_masses[self.center_idx] * gc
        x[self.i0] += delta_lambda * inv_masses[self.i0] * g0
        x[self.i1] += delta_lambda * inv_masses[self.i1] * g1
        x[self.i2] += delta_lambda * inv_masses[self.i2] * g2

        return lambda_acc
