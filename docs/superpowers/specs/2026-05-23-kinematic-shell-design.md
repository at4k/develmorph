# Kinematic Shell Model Design

**Date:** 2026-05-23  
**Branch:** feat-xpbd-solver  
**Status:** Approved

## Background and Motivation

### Model History

The project has iterated through three model generations:

- **Model B (PhD thesis):** Dense triangle mesh membrane with edge springs. Physically accurate but O(N_mesh) per cell — too expensive for 1,000 cells.
- **Model A (project start):** Center-only dynamics. 1 particle per cell. Failed to reproduce diamond formation because the ellipsoidal shell's geometric effect on cells was invisible to a single center point.
- **Current model:** Center + 6–12 surface points (oct/ico topology). Improved contact geometry, but surface points participate in XPBD, creating a zero-energy rotational mode that causes visible drift during post-division settling.

### Problem Statement

The surface point drift has a structural root cause: rigid rotation of the surface shell around the center costs zero energy under distance constraints alone (edge lengths are preserved; shell orientation is unconstrained). With only 6–12 points, small perturbations from cell-cell contact during settling excite this mode.

Adding dihedral constraints (Model B direction) would suppress the mode but increase per-cell complexity, contradicting the 1,000-cell scalability goal.

### Design Goal

Replace surface point dynamics with an analytical contact geometry layer, while retaining the geometric information that Model A lacked. Target: 1,000 cells in 1–10 minutes, tracking center positions, contact topology, and approximate cell shape.

---

## Architecture: Three-Layer Model

```
┌─────────────────────────────────────────────┐
│  Layer 3: Visual Shell (kinematic)          │
│  Reconstructed post-step from center +      │
│  contact records. Not part of XPBD.         │
│  Drift = zero by construction.              │
├─────────────────────────────────────────────┤
│  Layer 2: Contact Geometry (analytical)     │
│  Overlap δ_ij → contact area A_ij →        │
│  contact normal n_ij. Computed each step.   │
├─────────────────────────────────────────────┤
│  Layer 1: XPBD Physics (centers only)       │
│  1 particle per cell. Constraints:          │
│  Collision + Adhesion + OffsetShell         │
└─────────────────────────────────────────────┘
```

---

## Layer 1: XPBD Physics

### PhysicsCell

Replaces `CellBody`. Contains only what participates in XPBD:

```python
class PhysicsCell:
    center_idx: int        # index into flat positions array
    radius: float          # constant between divisions
    name: str
    orientation: np.ndarray  # 3x3, for division axis
```

Volume conservation is implicit: `r` is constant between divisions, so volume `(4/3)pi r^3` is conserved without a dynamic constraint. This eliminates `VolumeConstraint`, `DistanceConstraint` (membrane), and `TriangleAreaConstraint` entirely.

### OffsetShellConstraint

The key fix for Model A's failure. Instead of constraining the center to lie inside the raw ellipsoid `(a, b, c)`, constrain it to lie inside the effective ellipsoid `(a-r, b-r, c-r)`:

```
C_eff = sqrt( (px/(a-r))^2 + (py/(b-r))^2 + (pz/(c-r))^2 ) - 1
```

Correct only when `C_eff > 0` (center too close to wall).

**Why this fixes diamond formation:** For the C. elegans shell `(1.84, 1.26, 1.04)` with 4-cell-stage radius `r ≈ 0.82`, the effective semi-axes become `(1.02, 0.44, 0.22)`. The L-R direction (0.22) is extremely confined. Any cell displaced from the A-P axis feels a strong restoring force toward it. This geometric asymmetry — invisible to a bare center-point model — is the physical driver that Model A was missing.

The approximation is exact at the principal axis tips and slightly off elsewhere; this is acceptable for this level of biological modeling.

### CollisionConstraint

Unchanged. Center-to-center repulsion:

```
C = |c_i - c_j| - (r_i + r_j),  corrects when C < 0
```

### AdhesionConstraint (contact-area-weighted)

Replaces the hysteresis-only on/off adhesion. Force is proportional to contact area:

```
overlap   = max(0, r_i + r_j - |c_i - c_j|)
r_harm    = 2 r_i r_j / (r_i + r_j)
A_contact = pi * overlap * r_harm        # Hertz-like contact area
F_adhesion = k_adh * A_contact * n_ij   # continuous, not on/off
```

Cells with larger contact area attract more strongly. EMS-P2 contact (both large cells, large overlap) will naturally produce stronger adhesion than ABa-ABp, without requiring a Wnt bias parameter.

Implementation note: In XPBD, this is best expressed as compliance modulation rather than a direct force addition. Compliance `alpha = f(1/A_contact)` — larger contact area yields stiffer (smaller compliance) constraint, which produces stronger correction. The exact mapping is to be decided in Phase 2.

---

## Layer 2: Contact Geometry

Computed after each XPBD step, before visualization:

```python
@dataclass
class ContactRecord:
    i: int              # PhysicsCell index
    j: int              # PhysicsCell index
    overlap: float      # delta_ij >= 0
    area: float         # A_contact (Hertz)
    normal: np.ndarray  # unit vector from j to i
```

Contact topology = set of `(i, j)` pairs where `overlap > 0`. This is the observable used for signaling (Wnt/Notch contact detection).

---

## Layer 3: Visual Shell (kinematic)

Reconstructed each step from center position and contact records. Not stored between steps; not part of any constraint.

Default (no contacts): sphere mesh (oct or ico, any density, chosen for visualization quality — not constrained by physics cost).

With contacts: for each `ContactRecord`, flatten the sphere toward the contact plane by an amount proportional to `overlap`. Implemented as a post-processing mesh deformation in `viz.py`.

Because the shell is purely derived from physics outputs, it cannot drift.

---

## Division

Simplified significantly:

1. Compute daughter positions: `c_daughter = c_parent ± (r_parent/2) * axis`
2. Daughter radius: `r_d = r_parent / 2^(1/3)` (volume conservation)
3. Add two `PhysicsCell` entries; remove parent
4. Clear adhesion pairs involving parent (indices are stale)

No surface point rearrangement. Daughters start overlapping and are separated by `CollisionConstraint` during settling.

---

## Data Structure Changes

| Component | Current | Approach C |
|-----------|---------|-----------|
| `CellBody` | center + 6–12 surface points + mesh | `PhysicsCell`: center + radius only |
| `VolumeConstraint` | dynamic, per cell | removed (r = constant) |
| `DistanceConstraint` (membrane) | per edge, per cell | removed |
| `TriangleAreaConstraint` | per face, per cell | removed |
| `ShellConstraint` | applied to surface points | `OffsetShellConstraint` on center |
| `AdhesionConstraint` | on/off hysteresis | contact-area-weighted |
| Surface visualization | surface_indices in XPBD | `SurfaceShell` reconstructed post-step |

**Particle count (1,000 cells):**

| Model | Particles | Constraints (approx) |
|-------|-----------|----------------------|
| Current (oct) | 7,000 | ~50,000 |
| Approach C | 1,000 | ~5,000 |

---

## Implementation Phases

### Phase 1: OffsetShellConstraint + PhysicsCell (validation gate)

- Implement `PhysicsCell`, `OffsetShellConstraint`
- Remove surface point dynamics from solver
- Port diamond formation demo to new model
- Success criterion: `|EMS_y| > 0.5` (comparable to current 0.84) without Wnt bias
- Write new tests for `PhysicsCell`, `OffsetShellConstraint`, division

### Phase 2: Contact-area-weighted adhesion

- Replace hysteresis adhesion with `A_contact`-weighted force
- Re-tune `k_adh` parameter
- Verify contact topology tracking

### Phase 3: SurfaceShell visualization

- Implement `SurfaceShell` kinematic reconstruction in `viz.py`
- Flatten at contact planes (proportional to overlap)
- No physics impact

### Phase 4: Spatial hashing for 1,000 cells

- Implement grid-based neighbor search
- O(N) collision detection instead of O(N²)
- Required before scaling past ~100 cells

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Diamond not reproduced by OffsetShell alone | Medium | Phase 2 contact-area adhesion as fallback |
| OffsetShell approximation inaccurate near non-principal directions | Low | Can replace with iterative nearest-point-on-ellipsoid if needed |
| 137 existing tests break | Certain | Rewrite tests for `PhysicsCell`; confirm before deleting old tests |
| Phase 1 settling slower without surface dynamics | Low | Center-only settling is simpler; likely faster |

---

## Open Questions

1. **k_adh tuning:** Contact-area adhesion introduces a new force scale. Need to re-tune against diamond formation metric.
2. **Shell deformation fidelity:** For Phase 3, how much contact-plane flattening is needed to make the visualization biologically meaningful?
3. **Spatial hashing threshold:** At what cell count does O(N²) collision become the bottleneck? Benchmark at 100, 200, 500 cells.
