# Fluid Particles

A Godot 4.8 GDExtension that implements a grid-based particle fluid simulation on the GPU using Godot's **RenderingDevice** compute API.

Each particle occupies a cell in a 3D spatial chunk grid and exchanges momentum with its neighbors every step, producing fluid-like behavior with gravity, surface tension, and attraction/repulsion forces. Two particle types are supported: **solid** (alpha > 200) and **liquid** (alpha ≤ 200). Rendering is done as depth-sorted point sprites via a custom GLSL shader.

> **Requires Godot 4.8 or newer.** The extension uses APIs and features not available in 4.7 and earlier.

## Nodes

| Class | Extends | Description |
|---|---|---|
| `FluidParticleSystem` | `Node3D` | Main simulation node. Owns the RenderingDevice compute pipeline, GPU buffers, and a `MultiMeshInstance3D` for rendering. |
| `FluidSource` | `Node3D` | Spawns particles near its world position into a parent `FluidParticleSystem`. |
| `FluidSink` | `Node3D` | Deactivates particles near its world position in a parent `FluidParticleSystem`. |

## Architecture

```
FluidParticleSystem (Node3D)
  ├── RenderingDevice compute pipeline
  │     ├── velocity_spread.glsl   — main physics step (one invocation per particle)
  │     ├── depth_sort_key.glsl    — compute camera-distance sort keys
  │     ├── clear_grid.glsl        — zero chunk grid each frame
  │     └── fluid_source_sink.glsl — source/sink dispatch (shared by FluidSource/FluidSink)
  ├── GPU buffers
  │     ├── particles[]            — Particle structs (position, color, temp/opacity, neighbors_filled)
  │     ├── chunk_grid[]           — ChunkCell structs (occupant index + accumulated velocity)
  │     ├── runnable_indices[]     — indirection for LOD dispatch
  │     └── sort_keys[]            — float distances for depth sorting
  └── MultiMeshInstance3D         — point sprite rendering (depth-sorted)
```

### Particle struct (GPU / C++ mirror, std430, 32 bytes)
```glsl
struct Particle {
    vec3  position;          // world-space grid coordinates
    float _pad;
    uint  color_packed;      // r,g,b,a as uint8 packed into uint32
    float attraction_force;  // temperatureopacity[0]
    float opacity_fade;      // temperatureopacity[1]
    float neighbors_filled;  // smoothed neighbor occupancy (0–1)
    float _pad2;
};
```

### Chunk cell struct (GPU, std430, 32 bytes)
```glsl
struct ChunkCell {
    int   occupant;   // particle index (-1 = empty)
    float vel_x;
    float vel_y;
    float vel_z;
    float vel_w;      // weight accumulator
    float _pad[3];
};
```

### Physics step (per particle)
1. Round `position` to nearest integer grid cell → `cell`
2. If `chunk_grid[cell].occupant == -1`, register this particle
3. Accumulate `momentum` from `swap0()` on each neighbor cell (atomic read+zero)
4. Count filled neighbors (`allfilled`)
5. If not fully surrounded **or** liquid type:
   - Apply accumulated momentum
   - Attenuate attraction force by surface coverage
   - Update `neighbors_filled` (smoothed)
6. Apply gravity (`momentum.y -= gravity`)
7. Clamp to grid boundaries (elastic reflection)
8. Try to move to new cell via `atomicCompSwap` (CAS)
   - On collision: revert position, transfer momentum
9. Spread `momentum * friction` to all neighbor cells atomically

## Building

### Prerequisites
- Godot 4.8+ (uses APIs not in 4.7)
- SCons (`pip install scons`)
- C++17 compiler (g++ / clang++ / MSVC)

### Setup
```bash
git clone --recursive https://github.com/YOUR_USER/godot_ext_particles
cd godot_ext_particles
git submodule update --init --recursive
```

The `godot-cpp` submodule is pinned to `master` because Godot 4.8 is still in prerelease and no `godot-4.8-stable` tag exists yet. Bindings are regenerated at build time from the committed [`extension_api.json`](extension_api.json) (dumped from Godot 4.8.dev3) via godot-cpp's `custom_api_file` option — see [`SConstruct`](SConstruct).

To refresh the API dump after upgrading Godot:
```bash
godot --headless --dump-extension-api --output extension_api.json
```

### Build
```bash
scons platform=linux target=template_debug
```

Supported `platform` values: `linux`, `windows`, `macos`. Supported `target` values: `template_debug`, `template_release`, `editor`.

### Output
Shared libraries land in `addons/fluid_particles/bin/`:
- Linux: `libgodot_ext_particles.linux.template_debug.x86_64.so`
- Windows: `libgodot_ext_particles.windows.template_debug.x86_64.dll`
- macOS: `libgodot_ext_particles.macos.template_debug.framework/`

## Usage

1. Copy the `addons/fluid_particles/` folder into your Godot 4.8 project (or clone this repo alongside it).
2. Enable **Fluid Particles** in Project Settings → Plugins.
3. Add a `FluidParticleSystem` node to your scene.
4. Configure properties in the Inspector:

### `FluidParticleSystem` properties

| Property | Type | Description |
|---|---|---|
| `grid_width` | int | Chunk grid X size (default 128) |
| `grid_height` | int | Chunk grid Y size (default 64) |
| `grid_depth` | int | Chunk grid Z size (default 128) |
| `num_particles` | int | Total particle count |
| `gravity` | Vector3 | Gravity vector |
| `gravity_local` | bool | Interpret gravity in local space |
| `surface_tension` | float | Repulsion from empty space (positive = attract surface) |
| `water_viscosity` | float | Attraction force magnitude for liquid particles |
| `attraction_force` | float | Global attraction force (range -2..2) |
| `neighbor_mode` | int | 6 (face) or 15 (face+diagonal) neighbors |
| `simulation_active` | bool | Run/pause the simulation |

**Initial Chunk** group:

| Property | Type | Description |
|---|---|---|
| `initial_chunk/use_initial_chunk` | bool | Spawn a block of particles on ready |
| `initial_chunk/origin` | Vector3 | Block origin |
| `initial_chunk/size` | Vector3i | Block dimensions |
| `initial_chunk/color` | Color | Block particle color |
| `initial_chunk/attraction` | float | Block particle attraction (range -2..2) |

**Shaders** group:

| Property | Type | Description |
|---|---|---|
| `clear_shader_path` | String | Path to `clear_grid.glsl` |
| `physics_shader_path` | String | Path to `velocity_spread.glsl` |
| `sortkey_shader_path` | String | Path to `depth_sort_key.glsl` |
| `render_material` | ShaderMaterial | Material for the MultiMeshInstance3D |

**Methods:**

| Method | Description |
|---|---|
| `spawn_block(origin, w, h, d, color, attraction)` | Spawn a block of particles |
| `add_velocity_impulse(impulse)` | Apply a velocity impulse to all particles |
| `reset_grid()` | Clear the chunk grid |
| `get_grid_aabb()` | Returns the simulation AABB (used by the editor gizmo) |

### `FluidSource` / `FluidSink` properties

| Property | Type | Description |
|---|---|---|
| `radius` | float | Effect radius (0.1..100) |
| `rate` | int | Particles per frame (1..1000) |
| `active` | bool | Enable/disable |
| `shader_path` | String | Path to `fluid_source_sink.glsl` |

`FluidSource` additionally:

| Property | Type | Description |
|---|---|---|
| `color` | Color | Spawned particle color |
| `attraction` | float | Spawned particle attraction (-2..2) |
| `opacity` | float | Spawned particle opacity (0..1) |

## CI

Pre-built binaries for Linux (x86_64, arm64), Windows (x86_64, arm64), and macOS (universal) are produced by the [GitHub Actions workflow](.github/workflows/build.yml) on every push and tag. Download the latest `fluid-particles-addon` artifact from the [Releases](../../releases) page and unzip it into your project's `addons/` folder.

## License

MIT
