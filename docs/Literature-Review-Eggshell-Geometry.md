# Literature Review: Eggshell Geometry and Cell Arrangement

## Overview

This document summarizes key findings from the literature on C. elegans
eggshell shape, extra-embryonic space, and their effects on early embryonic
cell arrangement. These findings inform the geometric constraint-based
approach for DevelMorph.

## Key Papers

### 1. Yamamoto & Kimura (2017) - Asymmetric Attraction Model

**Paper**: "An asymmetric attraction model for the diversity and robustness
of cell arrangement in nematodes"
(Development 144(23):4437, [PMC5769634](https://pmc.ncbi.nlm.nih.gov/articles/PMC5769634/))

**Eggshell model**: Rigid ellipsoid with variable aspect ratio (AR).

**Cell model**: Soft spheres with repulsive and attractive forces.

**Key findings**:
- AR 1.3-2.0 produces diamond-type arrangement (wild-type C. elegans)
- AR > 2.8 produces T-shaped and linear arrangements
- Wild-type C. elegans AR = 1.6 +/- 0.1
- Asymmetric attraction (different adhesion strengths between cell pairs)
  improves robustness: EMS-P2 adhesion (alpha=0.90) is stronger than
  other pairs (alpha=0.75)

**Limitations**: Uses simplified ellipsoid for eggshell. Does not account
for local contour differences or extra-embryonic space distribution.

### 2. Yamamoto et al. (2022) - Extra-Embryonic Space

**Paper**: "The extra-embryonic space and the local contour are crucial
geometric constraints regulating cell arrangement"
(Development 149(10), [PMC9148568](https://pmc.ncbi.nlm.nih.gov/articles/PMC9148568/))

**Eggshell model**: Actual eggshell shapes traced from microscopy images,
represented as phase-field functions. Real eggshells described as
**capsule-shaped** (cylinder + two hemispheres), not ellipsoidal.

**Key findings**:
- Real eggshells are closer to capsules than ellipsoids
- Capsule eggshells leave more extra-embryonic space than ellipsoids
  with the same aspect ratio
- Extra-embryonic space (ES) quantity and location are independent
  geometric factors affecting cell arrangement
- Same AR can produce different arrangements depending on ES
- Increasing ES experimentally transforms diamond to linear arrangement
- ES effect interacts with cell-cell adhesion parameters

**Implication for DevelMorph**: Use capsule-shaped eggshell instead of
ellipsoid to better match real C. elegans geometry and provide
appropriate extra-embryonic space for cell rearrangement.

### 3. Tian et al. (2022) - Phase Field Model

**Paper**: "Computable early Caenorhabditis elegans embryo with a phase
field model"
(PLoS Comput Biol 18(1), [PMC8794267](https://pmc.ncbi.nlm.nih.gov/articles/PMC8794267/))

**Eggshell model**: Phase-field representation reconstructed from
experimental measurements.

**Cell model**: Phase fields phi_i(r,t) for each cell. Four forces:
- Surface tension: F_ten
- Repulsion (cell-cell and cell-eggshell): F_rep
- Attraction (E-cadherin mediated): F_atr
- Volume constraint: F_vol

**Key findings**:
- Phase field approach naturally handles cell deformation and contact areas
- Cell division implemented as instantaneous bisection
- Can reproduce early morphogenesis including cell shape changes
- Overdamped dynamics: d(phi_i)/dt = -(1/tau)(F_ten + F_rep + F_atr + F_vol)

### 4. WormBook - Eggshell Structure

**Source**: [The C. elegans eggshell](https://www.ncbi.nlm.nih.gov/books/NBK535513/)

- Chitin layer provides mechanical support for oval shape
- Without chitin, embryos are round and misshapen
- Eggshell is porous but rigid meshwork structure

## Key Parameters from Literature

| Parameter | Value | Source |
|-----------|-------|--------|
| Wild-type AR | 1.6 +/- 0.1 | Yamamoto & Kimura 2017 |
| Diamond AR range | 1.3 - 2.0 | Yamamoto & Kimura 2017 |
| EMS-P2 adhesion (alpha) | 0.90 | Yamamoto & Kimura 2017 |
| Other cell adhesion (alpha) | 0.75 | Yamamoto & Kimura 2017 |
| dpy-11 RNAi AR | 1.5 +/- 0.2 | Yamamoto et al. 2022 |

## Design Implications for DevelMorph

### Eggshell Shape

Replace the current ellipsoid `(x/a)^2 + (y/b)^2 + (z/c)^2 = 1` with
a **capsule-shaped boundary**: a cylinder of length L capped by two
hemispheres of radius R. This better approximates the real eggshell and
provides more realistic extra-embryonic space distribution.

Alternatively, a **superellipsoid** `(x/a)^n + (y/b)^n + (z/c)^n = 1`
with n > 2 can approximate the capsule shape continuously. The capsule
is the limit as n -> infinity.

### Cell Contact Model

The literature uses either:
- Soft sphere with point-based force calculation (Yamamoto models)
- Phase field with natural area contact (Tian model)

For DevelMorph, a geometric contact area approximation may capture the
essential effect of face-to-face pushing without the full complexity
of phase field computation. See the section below.

### Contact Area vs Point Contact

Current DevelMorph uses point contact (closest point between capsule
segments). This is insufficient because:

1. During division, two cells push on a neighbor as a connected pair
   (capsule shape), creating a **contact surface**, not a contact point.
2. The shape of the contact surface determines the effective push
   direction (centroid of contact area).
3. If the spindle (capsule axis) shifts slightly, the contact surface
   shape changes asymmetrically, causing the push direction to shift.
4. This asymmetry makes cell rearrangement easier than point-contact
   models predict, contributing to the diamond arrangement.

A possible geometric approximation: instead of using the closest point
between capsule and sphere, compute the **contact region** (the set of
surface points within a threshold distance) and use its centroid as
the effective contact point. This would capture the face-pushing effect
without full deformation computation.

## Mechanistic Model: Spindle, Volume Conservation, and Space-Driven Movement

### The Chain of Causation in Cell Rearrangement

Through analysis of the physical mechanism of cell division and
rearrangement, we identified the following causal chain:

1. **Spindle is the primary force-generating structure**
   - During cell division, the strongest mechanical structure is the
     mitotic spindle, not the cell membrane.
   - The spindle is thin and rod-like, pushing locally rather than
     uniformly across the cell surface.

2. **Cell interior is liquid (incompressible)**
   - The cytoplasm is essentially a liquid. When the spindle pushes
     from within, the cell cannot compress.
   - Volume is conserved: the cell deforms rather than shrinks.

3. **Deformation follows available space**
   - Like pushing a water balloon with a stick: the balloon bulges
     in directions where space is available.
   - The cell membrane extends into nearby gaps between other cells
     or between cells and the eggshell.

4. **Deformation shifts the center of mass**
   - As the cell bulges into available space, its center of mass
     moves toward that space.
   - This is not a "force" in the traditional sense but a geometric
     consequence of volume conservation + space availability.

5. **Movement is self-reinforcing**
   - Once the center of mass shifts, the cell occupies more of the
     available space, which in turn creates new gaps elsewhere.
   - The cell "flows" into the space, and the flow direction is
     determined by the geometry of surrounding constraints.

### Relationship to Extra-Embryonic Space

This mechanistic picture directly explains why extra-embryonic space
(ES) is the key determinant of cell arrangement (Yamamoto et al. 2022):

- ES defines **where cells can bulge into**
- The distribution of ES determines the **direction** of deformation
- Capsule-shaped eggshells provide more ES at the sides than
  ellipsoids, allowing more lateral movement

### Example: P1 Division Pushing AB

When P1 divides along the x-axis while AB is dividing along the y-axis:

```
Before:
    Eggshell (capsule)
    +---------------------------+
    |                           |
    |     ABa                   |
    |      |   P1 spindle       |
    |     AB----[===]----P1     |
    |      |                    |
    |     ABp                   |
    |                           |
    +---------------------------+

P1's spindle pushes locally (thin rod, not whole cell surface).
P1's volume is conserved -> bulges where space exists.
The bulge pushes against AB capsule's midsection.
Eggshell constrains outward movement.
AB capsule rotates because the push + eggshell constraint
deflects the force into the y-direction.
```

The push is not a uniform force from P1's center. It is a
**local pressure from the spindle region**, redistributed by
volume conservation into the available space. This makes the
interaction inherently asymmetric and direction-dependent,
which is what drives the diamond arrangement.

### Implications for DevelMorph's Geometric Model

Rather than computing forces (repulsion, attraction, damping),
the model should directly compute **where space is available**
for each cell and move cells toward that space:

1. **For each cell, find surrounding obstacles**
   (other cells, eggshell, dividing pair capsules)
2. **Compute the Voronoi-like centroid of available space**
   around the cell
3. **Move the cell center toward that centroid**
   This naturally captures:
   - Volume conservation (cell fills available space)
   - Spindle-driven asymmetry (dividing cells constrain
     space differently than resting cells)
   - Eggshell geometry effects (capsule vs ellipsoid
     changes space distribution)
4. **Hard constraints remain**: no overlap, stay inside eggshell

This approach eliminates force-balance parameters (damping,
adhesion_factor, etc.) and replaces them with pure geometry:
the cell arrangement is determined by the shape of the available
space.

## Comparison with Prior Work and Originality

### Existing Approaches in the Literature

| Model | Cell shape | Eggshell | Mechanism | Parameters |
|-------|-----------|----------|-----------|------------|
| Yamamoto 2017 | Soft spheres | Ellipsoid | Force balance (repulsion + attraction) | alpha, friction, noise |
| Yamamoto 2022 | Soft spheres | Real contour (phase field) | Force balance + ES analysis | Same + ES measurement |
| Tian 2022 | Phase fields | Phase field | PDE evolution (4 force terms) | gamma, g, sigma, M, tau |
| **DevelMorph** | **Capsules** | **Capsule** | **Space-centroid seeking** | **Eggshell geometry only** |

### What the Literature Does

All existing models use **force-based dynamics**:
- Define forces (repulsion, attraction, surface tension, volume constraint)
- Solve equations of motion (overdamped Langevin or PDE)
- Tune force parameters to reproduce observed arrangements
- Yamamoto 2022 discovered the importance of ES empirically but still
  uses force-based simulation to compute cell positions

### What DevelMorph Does Differently

DevelMorph's approach is **geometry-first, not force-first**:

1. **No force parameters to tune**: Cell positions are determined by
   the geometry of available space, not by balancing competing forces.
   This eliminates the parameter sensitivity problem (small changes
   in damping, adhesion_factor, etc. causing large changes in outcome).

2. **Space-centroid as the driving mechanism**: Instead of computing
   forces and solving dynamics, each cell moves toward the centroid
   of its available space. This directly implements the physical
   insight that volume-conserving incompressible cells flow into
   available gaps.

3. **Mechanistic justification**: The space-centroid approach is not
   ad hoc but follows from the physical chain: spindle pushes locally
   -> incompressible cytoplasm redistributes -> cell bulges into
   available space -> center of mass shifts. Force-based models
   approximate this indirectly through parameter tuning.

4. **Dividing pairs as capsules**: Existing models treat dividing
   cells as separate spheres. DevelMorph models them as capsules
   (connected by the spindle), which correctly represents the
   mechanical coupling during division and the asymmetric space
   occupation.

5. **Capsule-shaped eggshell**: Following Yamamoto 2022's finding
   that real eggshells are capsule-shaped, DevelMorph uses this
   directly as a geometric boundary rather than tracing real images
   with phase fields.

### Summary of Originality

The core originality is the **inversion of the modeling paradigm**:

- **Literature**: Forces -> dynamics -> equilibrium positions
  (then explain why ES matters post-hoc)
- **DevelMorph**: Available space -> centroid -> positions
  (ES is the primary input, not a derived observation)

This is analogous to the difference between Newtonian mechanics
(compute forces, integrate trajectories) and variational principles
(find the configuration that satisfies geometric constraints). Both
can produce the same result, but the geometric approach makes the
role of space explicit and eliminates force-parameter sensitivity.
