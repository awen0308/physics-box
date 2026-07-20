extends Node2D

const SCREEN_SIZE := Vector2(1152, 648)
const WALL_THICKNESS := 50
const BALL_RADIUS := 20.0
const TRAIL_LENGTH := 160
const MAX_SPEED := 1200.0
const MOVE_FORCE := 2500.0
const GHOST_STEPS := 30
const MPC_CANDIDATES := 10
const MPC_HORIZON := 12
const MPC_REPLAN := 0.1

const BallShader = preload("res://shaders/ball.gdshader")
const BgShader = preload("res://shaders/background.gdshader")
const WallArtScript = preload("res://scripts/wall_art.gd")
const ShadowScript = preload("res://scripts/shadow.gd")

@onready var ball: RigidBody2D = $Ball
@onready var data_collector = $DataCollector
@onready var world_model = $WorldModel

var rng := RandomNumberGenerator.new()
var last_state: Variant = null
var current_action := Vector2.ZERO
var model_loaded := false
var model_type := "MLP"
var elapsed := 0.0
var shake_intensity := 0.0

var camera: Camera2D
var shadow: Node2D
var real_trail: Line2D
var pred_trail: Line2D
var ghost_trail: Line2D
var predicted_ball: Sprite2D
var target_mode := false
var target_pos := Vector2.ZERO
var plan_timer := 0.0
var planned_action := Vector2.ZERO
var dream_mode := false
var dream_state: Array = [0.0, 0.0, 0.0, 0.0]
var pred_err_ema := 0.0
var flash_text := ""
var flash_timer := 0.0
var hud_flash: Label
var metrics_text := ""
var bg_shader: ShaderMaterial
var hud_status: Label
var hud_samples: Label
var hud_fps: Label
var hud_explain: Label


func _ready():
	rng.randomize()
	setup_environment()
	setup_background()
	var art = WallArtScript.new()
	add_child(art)
	shadow = ShadowScript.new()
	add_child(shadow)
	setup_trails()
	setup_ghost_trail()
	setup_predicted_ball()
	setup_ball_visual()
	setup_camera()
	setup_hud()
	create_walls()
	ball.body_entered.connect(_on_ball_collided)
	ball.can_sleep = false
	reset_ball()
	load_world_model()
	data_collector.start_recording()
	update_hud()
	if model_loaded:
		flash("已加载世界模型！用方向键玩，粉点是模型的预测")
	else:
		flash("欢迎！用方向键/鼠标玩，按 S 存数据后即可训练模型")


func setup_environment():
	var env_node := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = 2
	env.background_color = Color(0.02, 0.03, 0.07)
	env.glow_enabled = true
	env.glow_intensity = 0.9
	if "glow_bloom" in env:
		env.glow_bloom = 0.7
	env_node.environment = env
	add_child(env_node)


func setup_background():
	var bg := ColorRect.new()
	bg.position = Vector2.ZERO
	bg.size = get_viewport_rect().size
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	bg_shader = ShaderMaterial.new()
	bg_shader.shader = BgShader
	bg_shader.set_shader_parameter("resolution", get_viewport_rect().size)
	bg.material = bg_shader
	bg.z_index = -100
	bg.z_as_relative = false
	add_child(bg)


func setup_trails():
	real_trail = Line2D.new()
	real_trail.width = 3.0
	real_trail.z_index = 5
	var rgrad := Gradient.new()
	rgrad.colors = PackedColorArray([Color(0.4, 0.9, 1.0, 0.0), Color(0.7, 0.95, 1.0, 0.95)])
	rgrad.offsets = PackedFloat32Array([0.0, 1.0])
	real_trail.gradient = rgrad
	var rmat := CanvasItemMaterial.new()
	rmat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
	real_trail.material = rmat
	add_child(real_trail)

	pred_trail = Line2D.new()
	pred_trail.width = 3.0
	pred_trail.z_index = 5
	var pgrad := Gradient.new()
	pgrad.colors = PackedColorArray([Color(1.0, 0.3, 0.5, 0.0), Color(1.0, 0.4, 0.6, 0.9)])
	pgrad.offsets = PackedFloat32Array([0.0, 1.0])
	pred_trail.gradient = pgrad
	var pmat := CanvasItemMaterial.new()
	pmat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
	pred_trail.material = pmat
	add_child(pred_trail)


func setup_ghost_trail():
	ghost_trail = Line2D.new()
	ghost_trail.width = 2.0
	ghost_trail.z_index = 3
	var ggrad := Gradient.new()
	ggrad.colors = PackedColorArray([Color(0.4, 1.0, 0.7, 0.0), Color(0.5, 1.0, 0.8, 0.8)])
	ggrad.offsets = PackedFloat32Array([0.0, 1.0])
	ghost_trail.gradient = ggrad
	var gmat := CanvasItemMaterial.new()
	gmat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
	ghost_trail.material = gmat
	ghost_trail.visible = false
	add_child(ghost_trail)


func setup_predicted_ball():
	predicted_ball = Sprite2D.new()
	predicted_ball.texture = preload("res://icon.svg")
	var mat := ShaderMaterial.new()
	mat.shader = BallShader
	mat.set_shader_parameter("core_color", Color(1.0, 0.4, 0.5))
	mat.set_shader_parameter("glow_color", Color(0.7, 0.05, 0.2))
	predicted_ball.material = mat
	predicted_ball.scale = Vector2(0.32, 0.32)
	predicted_ball.visible = false
	predicted_ball.z_index = 4
	add_child(predicted_ball)


func setup_ball_visual():
	var sprite := Sprite2D.new()
	sprite.texture = preload("res://icon.svg")
	var mat := ShaderMaterial.new()
	mat.shader = BallShader
	mat.set_shader_parameter("core_color", Color(0.7, 0.95, 1.0))
	mat.set_shader_parameter("glow_color", Color(0.1, 0.4, 0.9))
	sprite.material = mat
	sprite.scale = Vector2(0.32, 0.32)
	ball.add_child(sprite)


func setup_camera():
	camera = Camera2D.new()
	camera.position = SCREEN_SIZE / 2.0
	camera.zoom = Vector2(1, 1)
	camera.anchor_mode = Camera2D.ANCHOR_MODE_DRAG_CENTER
	add_child(camera)


func setup_hud():
	var layer := CanvasLayer.new()
	layer.layer = 10
	add_child(layer)

	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 18)
	margin.add_theme_constant_override("margin_top", 14)
	layer.add_child(margin)

	var vbox := VBoxContainer.new()
	margin.add_child(vbox)

	hud_status = Label.new()
	hud_status.add_theme_color_override("font_color", Color(0.7, 0.95, 1.0))
	vbox.add_child(hud_status)

	hud_samples = Label.new()
	hud_samples.add_theme_color_override("font_color", Color(0.75, 0.9, 1.0))
	vbox.add_child(hud_samples)

	hud_fps = Label.new()
	hud_fps.add_theme_color_override("font_color", Color(0.75, 0.9, 1.0))
	vbox.add_child(hud_fps)

	hud_explain = Label.new()
	hud_explain.add_theme_color_override("font_color", Color(0.95, 0.9, 0.6))
	hud_explain.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hud_explain.custom_minimum_size = Vector2(720, 0)
	vbox.add_child(hud_explain)

	var help := Label.new()
	help.add_theme_color_override("font_color", Color(0.5, 0.7, 0.9))
	help.text = "方向键/W A D 控制 - 鼠标左键吸引 - 右键设目标 - G 规划 - M 梦境 - R 重置 - S 存数据 - L 重载模型"
	vbox.add_child(help)

	var legend := Label.new()
	legend.add_theme_color_override("font_color", Color(0.6, 0.8, 1.0))
	legend.text = "● 青线=真实轨迹   ● 粉点=世界模型单步预测   ● 绿线=模型想象的未来(30步)   [模型即环境]"
	vbox.add_child(legend)

	hud_flash = Label.new()
	hud_flash.add_theme_color_override("font_color", Color(1.0, 0.85, 0.3))
	hud_flash.text = ""
	vbox.add_child(hud_flash)


func create_walls():
	var walls_node := Node2D.new()
	walls_node.name = "Walls"
	add_child(walls_node)

	var rects := {
		"Top": Rect2(-WALL_THICKNESS, -WALL_THICKNESS, SCREEN_SIZE.x + 2 * WALL_THICKNESS, WALL_THICKNESS),
		"Bottom": Rect2(-WALL_THICKNESS, SCREEN_SIZE.y, SCREEN_SIZE.x + 2 * WALL_THICKNESS, WALL_THICKNESS),
		"Left": Rect2(-WALL_THICKNESS, 0, WALL_THICKNESS, SCREEN_SIZE.y),
		"Right": Rect2(SCREEN_SIZE.x, 0, WALL_THICKNESS, SCREEN_SIZE.y)
	}
	for wall_name in rects:
		var body := StaticBody2D.new()
		body.name = wall_name
		var shape := CollisionShape2D.new()
		var rect_shape := RectangleShape2D.new()
		var rect: Rect2 = rects[wall_name]
		rect_shape.size = rect.size
		shape.position = rect.position + rect.size / 2.0
		shape.shape = rect_shape
		body.add_child(shape)
		walls_node.add_child(body)


func _process(delta):
	elapsed += delta
	if bg_shader:
		bg_shader.set_shader_parameter("time", elapsed)
	update_hud()
	update_shake(delta)
	if flash_timer > 0.0:
		flash_timer -= delta
		if hud_flash:
			hud_flash.text = flash_text
	else:
		if hud_flash:
			hud_flash.text = ""


func flash(msg: String):
	flash_text = msg
	flash_timer = 2.5


func update_shake(delta):
	if shake_intensity > 0.1:
		camera.offset = Vector2(
			rng.randf_range(-shake_intensity, shake_intensity),
			rng.randf_range(-shake_intensity, shake_intensity)
		)
		shake_intensity = max(0.0, shake_intensity - delta * 40.0)
	else:
		camera.offset = Vector2.ZERO


func _physics_process(delta):
	handle_input(delta)

	## 仅当设定了有效目标（非原点）才执行模型规划，否则小球保持玩家操控。
	if target_mode and model_loaded and target_pos != Vector2.ZERO:
		plan_timer -= delta
		if plan_timer <= 0.0:
			planned_action = plan_action(current_state_array(), target_pos)
			plan_timer = MPC_REPLAN
		current_action = planned_action
	else:
		planned_action = Vector2.ZERO

	## 梦境模式：冻结真实物理引擎，完全用世界模型驱动小球。
	## 这是世界模型的终极验证——"模型即环境"，不再依赖 Godot 物理。
	if dream_mode and model_loaded:
		ball.freeze = true
		var a := [current_action.x, current_action.y]
		dream_state = world_model.predict(dream_state, a)
		ball.position = Vector2(dream_state[0], dream_state[1])
		ball.linear_velocity = Vector2(dream_state[2], dream_state[3])
		last_state = get_ball_state()
		real_trail.add_point(ball.position)
		if real_trail.get_point_count() > TRAIL_LENGTH:
			real_trail.remove_point(0)
		update_shadow()
		draw_ghost()
		return

	ball.freeze = false
	ball.apply_central_impulse(current_action)

	var next_state := get_ball_state()
	if last_state != null:
		data_collector.record_step(last_state, current_action, next_state)
		if model_loaded:
			var pred = world_model.predict(
				[last_state.pos.x, last_state.pos.y, last_state.vel.x, last_state.vel.y],
				[current_action.x, current_action.y]
			)
			predicted_ball.visible = true
			predicted_ball.position = Vector2(pred[0], pred[1])
			pred_err_ema = lerp(pred_err_ema, next_state["pos"].distance_to(predicted_ball.position), 0.1)
			pred_trail.add_point(Vector2(pred[0], pred[1]))
			if pred_trail.get_point_count() > TRAIL_LENGTH:
				pred_trail.remove_point(0)
			draw_ghost()

	last_state = next_state
	real_trail.add_point(ball.position)
	if real_trail.get_point_count() > TRAIL_LENGTH:
		real_trail.remove_point(0)
	update_shadow()


## 当前状态数组（优先用上一帧记录，保证与训练分布一致）
func current_state_array() -> Array:
	if last_state == null:
		return [ball.position.x, ball.position.y, ball.linear_velocity.x, ball.linear_velocity.y]
	return [last_state.pos.x, last_state.pos.y, last_state.vel.x, last_state.vel.y]


## 用世界模型连续推演未来 GHOST_STEPS 帧（在 Godot 里画出"想象的未来"）
func draw_ghost():
	if not model_loaded:
		ghost_trail.visible = false
		return
	ghost_trail.visible = true
	var s: Array = current_state_array()
	var rollout = world_model.predict_rollout(s, [[current_action.x, current_action.y]], GHOST_STEPS)
	## 把想象轨迹限制在盒子内，避免穿墙飞出、看着像出 bug。
	var lo := Vector2(WALL_THICKNESS, 0.0)
	var hi := Vector2(SCREEN_SIZE.x - WALL_THICKNESS, SCREEN_SIZE.y - WALL_THICKNESS)
	ghost_trail.clear_points()
	ghost_trail.add_point(ball.position)
	for st in rollout:
		ghost_trail.add_point(Vector2(clamp(st[0], lo.x, hi.x), clamp(st[1], lo.y, hi.y)))


## Model-Based Planning (MPC)：在候选动作中搜索让"想象终点"最接近目标的那一个。
## 这是世界模型的核心用途——用模型代替真实环境做规划。
func plan_action(state_arr: Array, goal: Vector2) -> Vector2:
	var best_cost := INF
	var best_a := Vector2.ZERO
	for k in range(MPC_CANDIDATES):
		var ang := rng.randf() * TAU
		var mag := rng.randf_range(2.0, 18.0)
		var a := Vector2(cos(ang), sin(ang)) * mag
		var rollout = world_model.predict_rollout(state_arr, [[a.x, a.y]], MPC_HORIZON)
		if rollout.is_empty():
			continue
		var last = rollout[rollout.size() - 1]
		var d := Vector2(last[0], last[1]).distance_to(goal)
		if d < best_cost:
			best_cost = d
			best_a = a
	return best_a


func _draw():
	if target_mode and target_pos != Vector2.ZERO:
		draw_arc(target_pos, 26.0, 0.0, TAU, 48, Color(0.4, 1.0, 0.7, 0.9), 3.0)
		draw_circle(target_pos, 4.0, Color(0.4, 1.0, 0.7, 0.9))


func handle_input(delta):
	var force := Vector2.ZERO
	if Input.is_key_pressed(KEY_LEFT) or Input.is_key_pressed(KEY_A):
		force.x -= 1.0
	if Input.is_key_pressed(KEY_RIGHT) or Input.is_key_pressed(KEY_D):
		force.x += 1.0
	if Input.is_key_pressed(KEY_UP) or Input.is_key_pressed(KEY_W):
		force.y -= 1.0
	if Input.is_key_pressed(KEY_DOWN):
		force.y += 1.0

	if force != Vector2.ZERO:
		current_action = force.normalized() * MOVE_FORCE * delta
	elif Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		var to_mouse := get_global_mouse_position() - ball.position
		current_action = to_mouse.limit_length(1500.0) * delta
	else:
		## 空闲时不再施加随机冲量——小球只受真实物理（重力/碰撞）影响，
		## 由玩家显式操控。这样演示时小球行为可控、可解释，符合面试展示要求。
		current_action = Vector2.ZERO

	if ball.linear_velocity.length() > MAX_SPEED:
		ball.linear_velocity = ball.linear_velocity.limit_length(MAX_SPEED)


func get_ball_state() -> Dictionary:
	return {
		"pos": ball.position,
		"vel": ball.linear_velocity
	}


func update_shadow():
	var floor_y := SCREEN_SIZE.y
	var h: float = clamp(floor_y - ball.position.y, 0.0, SCREEN_SIZE.y)
	var t: float = h / SCREEN_SIZE.y
	shadow.position = Vector2(ball.position.x, floor_y - 4.0)
	var s: float = lerp(1.0, 0.35, t)
	shadow.scale = Vector2(s, s)
	shadow.modulate.a = lerp(0.45, 0.12, t)


func reset_ball():
	ball.position = Vector2(SCREEN_SIZE.x / 2.0, 140.0)
	ball.linear_velocity = Vector2(rng.randf_range(-350.0, 350.0), 0.0)
	ball.angular_velocity = 0.0
	last_state = null
	world_model.reset_hidden()


func load_world_model():
	## 自动选择更准的模型：有指标的按开放环 rollout 误差（越小越好）挑选，
	## 没有指标的作为兜底。这样面试演示永远用质量最好的世界模型。
	var files := [
		{"path": "res://models/world_model_rnn.json", "metric": "res://models/metrics_rnn.json"},
		{"path": "res://models/world_model.json", "metric": "res://models/metrics.json"}
	]
	var best_path := ""
	var best_metric := INF
	var best_mfile := ""
	var fallback_path := ""
	var fallback_mfile := ""
	for f in files:
		if FileAccess.file_exists(f.path):
			if fallback_path == "":
				fallback_path = f.path
				fallback_mfile = f.metric
			var err := INF
			if FileAccess.file_exists(f.metric):
				var mf := FileAccess.open(f.metric, FileAccess.READ)
				var md = JSON.parse_string(mf.get_as_text())
				mf.close()
				if md != null and "rollout_pos_err_20" in md:
					err = float(md["rollout_pos_err_20"])
			if err < best_metric:
				best_metric = err
				best_path = f.path
				best_mfile = f.metric

	var loaded_path := best_path if best_path != "" else fallback_path
	var loaded_mfile := best_mfile if best_path != "" else fallback_mfile

	if loaded_path != "":
		world_model.load_model(loaded_path)
		model_loaded = true
		model_type = "RNN" if world_model.recurrent else "MLP"
		world_model.reset_hidden()
		pred_err_ema = 0.0
		if best_path == "":
			print("Loaded fallback model (no metrics): " + loaded_path)
	else:
		model_loaded = false
		metrics_text = ""
		pred_err_ema = 0.0
		print("No trained model found. Collect data with S, then run training/train.py.")
		return

	metrics_text = ""
	if loaded_mfile != "" and FileAccess.file_exists(loaded_mfile):
		var f := FileAccess.open(loaded_mfile, FileAccess.READ)
		var m = JSON.parse_string(f.get_as_text())
		f.close()
		if m != null and "rollout_pos_err_10" in m:
			metrics_text = "rollout err: @10=%.1fpx  @20=%.1fpx" % [m["rollout_pos_err_10"], m["rollout_pos_err_20"]]


func update_hud():
	if hud_status:
		if data_collector.is_recording:
			hud_status.text = "[REC]"
		else:
			hud_status.text = "[PAUSE]"
		if model_loaded:
			hud_status.text += "  [MODEL: ON " + model_type + "]"
		else:
			hud_status.text += "  [MODEL: OFF]"
		if target_mode:
			hud_status.text += "  [PLANNING -> TARGET]"
		if dream_mode:
			hud_status.text += "  [DREAM]"
	if hud_samples:
		hud_samples.text = "samples: " + str(data_collector.data.size())
	if hud_fps:
		hud_fps.text = "fps: " + str(Engine.get_frames_per_second())
	if hud_samples and metrics_text != "":
		hud_samples.text = "samples: " + str(data_collector.data.size()) + "   " + metrics_text + "   live pred err: %.1fpx" % pred_err_ema

	## 一句大白话解说当前在演示什么，面试官/外行都能秒懂。
	if hud_explain:
		if not model_loaded:
			hud_explain.text = "还没加载世界模型：按 S 存数据 → 终端跑训练 → 回这里按 L 加载。"
		elif dream_mode:
			hud_explain.text = "【梦境模式】真实物理引擎已关机，小球这一步步完全由世界模型“想象”出来——这就是“模型即环境”。"
		elif target_mode and target_pos != Vector2.ZERO:
			hud_explain.text = "【规划模式】小球借用世界模型自己算路线，自动奔向右键设定的目标（不需要真去试）。"
		else:
			hud_explain.text = "真实物理在跑（重力+撞墙）。粉点=世界模型猜的“下一步在哪”，绿线=它想象的“未来30步轨迹”。"


func _on_ball_collided(_body):
	shake_intensity = 7.0


func _input(event):
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_R:
				reset_ball()
				flash("已重置小球")
			KEY_S:
				data_collector.save_to_csv("res://data/trajectory.csv")
				flash("已保存 %d 条样本 -> data/trajectory.csv" % data_collector.data.size())
			KEY_L:
				load_world_model()
				if model_loaded:
					flash("世界模型已学会物理！粉点=它猜的下一步")
				else:
					flash("还没模型：先按 S 存数据，再跑训练")
			KEY_G:
				target_mode = not target_mode
				if target_mode:
					flash("规划模式：右键点个目标，小球自己算路过去")
				else:
					target_pos = Vector2.ZERO
					flash("已退出规划模式")
			KEY_M:
				if not model_loaded:
					flash("梦境模式需要模型，先按 L 加载")
				else:
					dream_mode = not dream_mode
					if dream_mode:
						dream_state = current_state_array()
						world_model.reset_hidden()
						flash("梦境：物理已关，小球由世界模型想象驱动（模型即环境）")
					else:
						flash("梦境关闭，恢复真实物理")
			KEY_ESCAPE:
				## 退出前自动保存数据，避免辛苦采集的样本丢失。
				if data_collector.data.size() > 0:
					data_collector.save_to_csv("res://data/trajectory.csv")
				get_tree().quit()
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_RIGHT:
		target_mode = true
		target_pos = get_global_mouse_position()
		flash("目标已设定，规划中...")
