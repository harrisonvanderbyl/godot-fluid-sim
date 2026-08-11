#!/usr/bin/env python
import os, sys

env = SConscript("godot-cpp/SConstruct")

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
