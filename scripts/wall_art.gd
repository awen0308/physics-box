extends Node2D

const SCREEN_SIZE := Vector2(1152, 648)

func _draw():
	var col := Color(0.2, 0.85, 1.0)
	draw_line(Vector2(0.0, 0.0), Vector2(SCREEN_SIZE.x, 0.0), col, 4.0)
	draw_line(Vector2(0.0, SCREEN_SIZE.y), Vector2(SCREEN_SIZE.x, SCREEN_SIZE.y), col, 4.0)
	draw_line(Vector2(0.0, 0.0), Vector2(0.0, SCREEN_SIZE.y), col, 4.0)
	draw_line(Vector2(SCREEN_SIZE.x, 0.0), Vector2(SCREEN_SIZE.x, SCREEN_SIZE.y), col, 4.0)
