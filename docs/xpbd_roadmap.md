# XPBD-based Cell Development Simulation: Roadmap

## 1. Core Concept: Why XPBD?

Unlike traditional spring models (Force-based), XPBD directly corrects positions
and introduces Compliance (flexibility, alpha_tilde) in the denominator,
providing the following advantages:

- **Absolute stability**: No divergence even when cells collide violently or divide.
- **Stiffness independence**: Cell "stiffness" remains constant regardless of
  time step (dt) or iteration count.
- **Flexible expression**: Membrane tension, volume conservation, and adhesion
  can all be handled as unified "constraints".

## 2. Cell Physical Model Design (1000-cell scale)

Configuration for creating membrane structures while keeping computational cost low:

- **Basic structure**: 1 cell = 1 particle (position, velocity, mass, polarity vector).
- **Adhesion constraint**: Dynamic distance constraints generated between neighboring cells.
- **Polarity**: Cells have "orientation"; adhesion strengthens only when orientations
  align (key to membrane formation).

## 3. Mathematical Algorithm (Python/NumPy Implementation)

### Adhesion/Collision Formula

- `C(x) = |p1 - p2| - L` (current distance - target distance)
- `alpha` (Compliance): 0 = rigid body, larger = softer spring.

### Main Loop

1. **Predict**: Apply gravity and autonomous movement forces to compute
   tentative position x_hat.
2. **Neighbor Search**: Use Spatial Hashing for fast neighbor pair extraction (O(N)).
3. **Solve Constraints (Loop)**:
   - Adhesion/Collision: Push back overlaps, pull back separations.
   - Polarity synchronization: Align orientation with neighboring cells.
4. **Finalize**: Determine velocity as `v = (x - x_old) / dt`.

## 4. Developmental Process Reproduction Techniques

| Phenomenon | XPBD Realization |
|---|---|
| Membrane formation | Make adhesion alpha small (hard) only when neighboring cells have aligned polarity. |
| Cell division | Duplicate particle and inherit polarity. Strengthen adhesion over several steps. |
| Invagination | Shorten target distance (L) for cells in specific area to contract the surface. |
| Cell sorting | Simply vary adhesion alpha by cell type (A, B) for automatic separation. |

## 5. Implementation Details

### 5.1 Shape Matching: Representing Membranes as a Single Constraint

When using dozens of points per membrane is too expensive, compose each cell
from **a few particles + Shape Matching**.

- **Model**: 1 cell = 1 center point + 4-6 surrounding points.
- **Mechanism**: No matter how particles move, a strong constraint pulls them
  back toward the original shape (sphere or ellipsoid).
- **Advantage**: Far fewer calculations than computing each mesh edge individually,
  while cell elasticity is cleanly controlled via XPBD Compliance (alpha_tilde).

### 5.2 Simplified Volume Conservation via Center Point

Standard volume preservation computes from surface polygons, which is expensive
at 1000-cell scale. Replace it with **center-point distance constraints**.

- **Pressure constraint**: Constrain the distance between center point P_center
  and surrounding points P_i to be >= R (radius).
- **Pseudo-fluid**: Add distance constraints between surrounding points to
  maintain membrane continuity.
- **XPBD strength**: PBD/XPBD does not tolerate penetration and directly
  restores positions, so cell overlap / "crushing" is avoided at very low
  computational cost.

### 5.3 Grid-based Neighbor Search (Spatial Hashing)

At 1000 cells (5000 points at 5 per cell), naive O(N^2) collision detection
is prohibitive.

- **Solution**: Introduce Spatial Hashing to partition space into a grid.
  Before the XPBD loop, list which grid cell contains which point.
  This reduces computational cost to O(N).
- 5000 points is comfortably real-time on modern CPUs (or GPUs).

### 5.4 Cell Division in Developmental Processes

XPBD handles the moment of cell division gracefully:

1. **Copy parent cell**: At a threshold, duplicate particles with a slight
   positional offset.
2. **Dynamic constraint addition**: Add distance constraints between new points.
3. **XPBD solver**: Since positions overlap, the solver pushes them apart in
   the next step.
4. **Smooth division**: Gradually increase constraint target values over several
   steps rather than applying sudden changes.

### 5.5 Recommended Data Structure and Update Loop

For 1000-cell scale, use **1 cell = 1 rigid body (center point) + surface points**.

XPBD update loop:

1. Apply external forces (gravity, adhesion, etc.)
2. Update neighbor search (Spatial Hash)
3. **XPBD iteration loop** (core):
   - Internal constraints (maintain cell shape)
   - External constraints (cell-cell collision / adhesion)
4. Velocity update

## 6. Cell Adhesion Model

### 6.1 Distance Constraint as "Spring" Simulation

The foundation of adhesion in XPBD is dynamically generating **Distance Constraints**
between particles of neighboring cells.

- **Mechanism**: When a surface point of cell A and a surface point of cell B
  come within a certain distance, an "adhesion constraint" is created between them.
- **XPBD advantage**: Traditional springs diverge when pulling force is too strong,
  but XPBD adjusts Compliance (alpha_tilde) to achieve adhesion that is
  "hard to tear apart, yet soft" — computed stably.

### 6.2 Surface Tension vs. Adhesion Balance (Energy Minimization)

Cells aggregate into tissues because they minimize surface energy.
In XPBD, this is expressed by balancing two opposing forces:

- **Repulsion (Collision)**: Pushes cells apart to prevent penetration.
- **Adhesion constraint**: Keeps surface points at target distance L
  (near zero or very small).
- **Result**: When adhesion wins, cells pack tightly and naturally form
  honeycomb-like (hexagonal) tissue structures.

### 6.3 Lightweight Adhesion Flag Management

Checking all surface point pairs across 1000 cells is expensive.
Use **heuristic adhesion management**:

- **Spatial Hashing filtering**: Only consider cells in the same grid cell
  as adhesion candidates.
- **Adhesion lifetime (State Machine)**: Once an adhesion constraint is created,
  maintain it until the distance exceeds a tear-off threshold.
- **Physical definition of adhesion strength**:
  - Compliance alpha_tilde: Larger = "sticky weak adhesion",
    smaller = "rigid bond".
  - Dynamically vary this value according to developmental stage
    (differentiation, etc.) to express cell migration and cell sorting.

### 6.4 Adhesion Handling During Cell Division

How adhesion is inherited at division is critical:

- **Parent cell inheritance**: Adhesion constraints the parent held with
  neighbors are split "half each" to the two daughter cells.
- **New adhesion suppression**: For several steps after division, set adhesion
  strength low. Wait for cells to settle into a physically stable configuration
  before strengthening — this prevents "explosion" at division time.

## 7. Python Implementation Skeleton

```python
# 1. Use Spatial Hashing to list neighbor pairs
# 2. Position correction using XPBD terms (alpha_prime)
# 3. Orientation maintenance through polarity vector averaging
```
