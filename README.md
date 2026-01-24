# DevelMorph

## Overview

DevelMorph is a developmental morphogenesis simulator focused on modeling early embryonic cell mechanics in 3D.
The initial scope is to simulate robust, physically plausible 1 → 2 → 4 cell progression with projection-based constraint correction.

## World Configuration
```
Space: 3D
Cells: Points with radius
Dynamics: Overdamped motion
Constraints: Projection-based correction
Shell: Ellipsoidal boundary
Visualization: matplotlib (initial stage)
```

## Goals

- Stable 3D cell dynamics
- Projection-based constraint handling
- Modular and extendable design
- GPU support via JAX/PyTorch/MPS

## Current Status

- Repository initialized
- Basic skeleton in place

## Next Steps

1. Build a minimal world with 1 cell
2. Add basic force interactions
3. Implement cell division
4. Add projection correction