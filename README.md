# godot_ext_particles

A Godot 4 GDExtension that implements the fluid/particle simulation from `particles.cpp` using GPU compute shaders via Godot's **RenderingDevice** API.

## Overview

Implements a grid-based particle fluid simulation where each particle:
- Occupies a cell in a 3D spatial chunk grid
- Exchanges momentum with its 6 (or 15) neighbors each step
- Experiences gravity, surface tension, and attraction/repulsion forces
- Supports two particle types: **solid** (alpha > 200) and **liquid** (alpha ≤ 200)
- Renders as depth-sorted point sprites using a custom GLSL shader

## Architecture

```
FluidParticleSystem (Node3D)
  ├── RenderingDevice compute pipeline
  │     ├── velocity_spread.glsl   — main physics step (one invocation per particle)
  │     ├── depth_sort_key.glsl    — compute camera-distance sort keys
  │     └── clear_grid.glsl        — zero chunk grid each frame
  ├── GPU buffers
  │     ├── particles[]            — Particle structs (position, color, temp/opacity, neighbors_filled)
  │     ├── chunk_grid[]           — ChunkCell structs (occupant index + accumulated velocity)
  │     ├── runnable_indices[]     — indirection for LOD dispatch
  │     └── sort_keys[]            — float distances for depth sorting
  └── MultiMeshInstance3D         — point sprite rendering (depth-sorted)
```

## Building

### Prerequisites
- Godot 4.3+ (for RenderingDevice compute shader support)
- SCons (`pip install scons`)
- C++17 compiler (g++ / clang++ / MSVC)

### Setup
```bash
git clone --recursive https://github.com/YOUR_USER/godot_ext_particles
cd godot_ext_particles
# godot-cpp submodule
git submodule update --init --recursive
cd godot-cpp && git checkout godot-4.3-stable && cd ..
# Build
scons platform=linux target=template_debug
```

Or add godot-cpp manually:
```bash
git submodule add https://github.com/godotengine/godot-cpp.git godot-cpp
git -C godot-cpp checkout godot-4.3-stable
```

### Output
Shared library lands in `demo/bin/`:
- Linux: `libgodot_ext_particles.linux.template_debug.x86_64.so`
- Windows: `libgodot_ext_particles.windows.template_debug.x86_64.dll`
- macOS: `libgodot_ext_particles.macos.template_debug.framework/`

## Usage

1. Open `demo/` as a Godot 4 project.
2. Add a `FluidParticleSystem` node to your scene.
3. Configure properties in the Inspector:

| Property | Type | Description |
|---|---|---|
| `grid_width` | int | Chunk grid X size (default 128) |
| `grid_height` | int | Chunk grid Y size (default 64) |
| `grid_depth` | int | Chunk grid Z size (default 128) |
| `num_particles` | int | Total particle count |
| `gravity` | float | Downward acceleration per step |
| `surface_tension` | float | Repulsion from empty space (positive = attract surface) |
| `water_viscosity` | float | Attraction force magnitude for liquid particles |
| `neighbor_mode` | int | 6 (face) or 15 (face+diagonal) neighbors |

## Simulation Details

### Particle struct (GPU / C++ mirror)
```glsl
struct Particle {
    vec3  position;          // world-space grid coordinates
    float _pad;
    uint  color_packed;      // r,g,b,a as uint8 packed into uint32
    float attraction_force;  // temperatureopacity[0]
    float opacity_fade;      // temperatureopacity[1]
    float neighbors_filled;  // smoothed neighbor occupancy (0–1)
    float _pad2;
};  // 32 bytes, std430
```

### Chunk cell struct (GPU)
```glsl
struct ChunkCell {
    int   occupant;   // particle index (-1 = empty)
    float vel_x;
    float vel_y;
    float vel_z;
    float vel_w;      // weight accumulator
    float _pad[3];
};  // 32 bytes, std430
```

### Physics step per particle
1. Round `position` to nearest integer grid cell → `cell`
2. If `chunk_grid[cell].occupant == -1`, register this particle
3. Accumulate `momentum` from `swap0()` on each neighbor cell (atomically read+zero)
4. Count filled neighbors (`allfilled`)
5. If not fully surrounded OR liquid type:
   - Apply accumulated momentum
   - Attenuate attraction force by surface coverage
   - Update `neighbors_filled` (smoothed)
6. Apply gravity (`momentum.y -= gravity`)
7. Clamp to grid boundaries (elastic reflection)
8. Try to move to new cell via `atomicCompSwap` (CAS)
   - On collision: revert position, transfer momentum
9. Spread `momentum * friction` to all neighbor cells atomically

## License
MIT
