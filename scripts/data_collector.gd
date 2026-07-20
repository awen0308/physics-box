class_name DataCollector
extends Node

var data: Array = []
var is_recording := false


func start_recording():
	data.clear()
	is_recording = true


func stop_recording():
	is_recording = false


func record_step(state: Dictionary, action: Vector2, next_state: Dictionary):
	if not is_recording:
		return
	data.append({
		"x": state.pos.x,
		"y": state.pos.y,
		"vx": state.vel.x,
		"vy": state.vel.y,
		"ax": action.x,
		"ay": action.y,
		"next_x": next_state.pos.x,
		"next_y": next_state.pos.y,
		"next_vx": next_state.vel.x,
		"next_vy": next_state.vel.y
	})


func save_to_csv(path: String):
	if data.is_empty():
		print("No data to save")
		return

	var dir := path.get_base_dir()
	var da := DirAccess.open("res://")
	if da != null:
		da.make_dir_recursive(dir)

	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("Failed to open %s for writing" % path)
		return

	file.store_line("x,y,vx,vy,ax,ay,next_x,next_y,next_vx,next_vy")
	for row in data:
		file.store_line(
			"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % [
				row.x, row.y, row.vx, row.vy, row.ax, row.ay,
				row.next_x, row.next_y, row.next_vx, row.next_vy
			]
		)
	file.close()
	print("Saved %d samples to %s" % [data.size(), path])
