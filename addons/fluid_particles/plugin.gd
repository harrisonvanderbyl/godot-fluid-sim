@tool
extends EditorPlugin

# --- FluidParticleSystem editor plugin ---------------------------------------
# Drop this addon into res://addons/fluid_particles/ of any Godot 4.3+ project.
# Provides:
#   - Bounding-box gizmo drawn in the 3-D viewport
#   - Inspector-editable shader paths with defaults pointing to addon shaders
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
