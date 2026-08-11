#!/usr/bin/env python
import os, sys

# godot-cpp has no released tag for 4.8 (still in prerelease), so we regenerate
# its bindings from the extension_api.json committed in this repo (dumped from
# Godot 4.8.dev3). The custom_api_file option takes precedence over the bundled
# api_version JSON inside godot-cpp/gdextension/.
api_file = "extension_api.json"
if not os.path.isfile(api_file):
    print("ERROR: %s not found. Dump it with: godot --headless --dump-extension-api --output extension_api.json" % api_file)
    sys.exit(1)

# Build a local env and set custom_api_file on it so godot-cpp's SConstruct
# picks it up via env.get("custom_api_file") when defining its options.
local_env = Environment(tools=["default"], PLATFORM="")
local_env["custom_api_file"] = os.path.abspath(api_file)

env = SConscript("godot-cpp/SConstruct", {"env": local_env})

# Extension sources
env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp")

# Platform-specific library naming
if env["platform"] == "macos":
    library = env.SharedLibrary(
        "addons/fluid_particles/bin/libgodot_ext_particles.{}.{}.framework/libgodot_ext_particles.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "addons/fluid_particles/bin/libgodot_ext_particles{}{}".format(
            env["suffix"], env["SHLIBSUFFIX"]
        ),
        source=sources,
    )

Default(library)
