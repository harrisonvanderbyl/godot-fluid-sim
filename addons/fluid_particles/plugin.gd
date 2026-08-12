@tool
extends EditorPlugin

# --- FluidParticleSystem editor plugin ---------------------------------------
# Drop this addon into res://addons/fluid_particles/ of any Godot 4.8+ project.
# Provides:
#   - Bounding-box gizmo drawn in the 3-D viewport (always visible when selected)
#   - Semi-transparent fill showing the simulation volume
#   - Center marker showing the rotation center
# ------------------------------------------------------------------------------

var _gizmo_plugin: FluidGizmoPlugin = null

func _enter_tree() -> void:
	_gizmo_plugin = FluidGizmoPlugin.new()
	add_node_3d_gizmo_plugin(_gizmo_plugin)

func _exit_tree() -> void:
	remove_node_3d_gizmo_plugin(_gizmo_plugin)

func _handles(object: Object) -> bool:
	return object != null and object.get_class() == "FluidParticleSystem"

func _edit(object: Object) -> void:
	if _gizmo_plugin and object:
		update_overlays()

# ------------------------------------------------------------------------------
class FluidGizmoPlugin extends EditorNode3DGizmoPlugin:

	func _get_gizmo_name() -> String:
		return "FluidParticleSystem"

	func _has_gizmo(node: Node3D) -> bool:
		return node.get_class() == "FluidParticleSystem"

	func _init() -> void:
		create_material("bb_wire", Color(0.2, 1.0, 0.4, 0.9), false, true)
		create_material("bb_fill", Color(0.2, 1.0, 0.4, 0.08), false, true, true)
		create_material("rot_center", Color(1.0, 0.4, 0.2, 1.0), false, true)
		create_handle_material("handles")

	func _redraw(gizmo: EditorNode3DGizmo) -> void:
		gizmo.clear()
		var node: Node3D = gizmo.get_node_3d()
		if not node.has_method("get_grid_aabb"):
			return

		var aabb: AABB = node.call("get_grid_aabb")
		var p := aabb.position
		var s := aabb.size
		var c := [
			p,
			p + Vector3(s.x, 0,   0  ),
			p + Vector3(s.x, s.y, 0  ),
			p + Vector3(0,   s.y, 0  ),
			p + Vector3(0,   0,   s.z),
			p + Vector3(s.x, 0,   s.z),
			p + Vector3(s.x, s.y, s.z),
			p + Vector3(0,   s.y, s.z),
		]
		var edges := [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
		var lines := PackedVector3Array()
		for e in edges:
			lines.append(c[e[0]])
			lines.append(c[e[1]])

		gizmo.add_lines(lines, get_material("bb_wire", gizmo))

		# Semi-transparent fill triangles (two per face, 6 faces)
		var fill := PackedVector3Array()
		var faces := [
			[0,1,2,3], # -Z
			[4,5,6,7], # +Z
			[0,1,5,4], # -Y
			[3,2,6,7], # +Y
			[0,3,7,4], # -X
			[1,2,6,5], # +X
		]
		for f in faces:
			fill.append(c[f[0]]); fill.append(c[f[1]]); fill.append(c[f[2]])
			fill.append(c[f[0]]); fill.append(c[f[2]]); fill.append(c[f[3]])
		gizmo.add_triangles(fill, get_material("bb_fill", gizmo))

		# Rotation center marker (center of the grid)
		var center := p + s * 0.5
		var hsize := 0.15 * min(s.x, min(s.y, s.z))
		var hlines := PackedVector3Array()
		hlines.append(center - Vector3(hsize, 0, 0)); hlines.append(center + Vector3(hsize, 0, 0))
		hlines.append(center - Vector3(0, hsize, 0)); hlines.append(center + Vector3(0, hsize, 0))
		hlines.append(center - Vector3(0, 0, hsize)); hlines.append(center + Vector3(0, 0, hsize))
		gizmo.add_lines(hlines, get_material("rot_center", gizmo))
