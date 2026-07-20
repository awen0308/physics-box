class_name WorldModel
extends Node

## 从 Python 训练脚本导出的 JSON 加载神经网络，在 Godot 里做前向推理。

var input_mean: Array
var input_std: Array
var output_mean: Array
var output_std: Array
var layers: Array = []
var loaded := false

## Latent recurrent (GRU) 支持：模型维护一个跨时间步的隐状态 h，
## 这是 Dreamer 类世界模型的核心——动力学建模在隐空间、带时序记忆。
var recurrent := false
var hidden_size := 0
var hidden: Array = []
var Wz: Array = []
var Wr: Array = []
var Wn: Array = []
var bz: Array = []
var br: Array = []
var bn: Array = []
var Wout_r: Array = []
var bout_r: Array = []


func load_model(path: String):
	if not FileAccess.file_exists(path):
		print("Model file not found: " + path)
		return

	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Failed to open model: " + path)
		return

	var json_text := file.get_as_text()
	file.close()

	var json := JSON.new()
	var err := json.parse(json_text)
	if err != OK:
		push_error("JSON parse error: " + json.get_error_message())
		return

	var data = json.get_data()
	input_mean = data["input_mean"]
	input_std = data["input_std"]
	output_mean = data["output_mean"]
	output_std = data["output_std"]
	recurrent = data.get("recurrent", false)
	if recurrent:
		hidden_size = int(data["hidden_size"])
		Wz = data["Wz"]
		Wr = data["Wr"]
		Wn = data["Wn"]
		bz = data["bz"]
		br = data["br"]
		bn = data["bn"]
		Wout_r = data["Wout"]
		bout_r = data["bout"]
		reset_hidden()
		loaded = true
		print("Recurrent (GRU) world model loaded: hidden=" + str(hidden_size))
	else:
		layers = data["layers"]
		loaded = true
		print("World model loaded: " + str(layers.size()) + " layers")


func predict(state: Array, action: Array) -> Array:
	if not loaded:
		push_error("Model not loaded")
		return [0.0, 0.0, 0.0, 0.0]

	if recurrent:
		return predict_recurrent(state, action)

	var x := normalize_input(state.duplicate())
	x.append_array(action.duplicate())

	for i in range(layers.size()):
		var layer = layers[i]
		x = matvec_mul(layer["W"], x)
		x = add_bias(x, layer["b"])
		if i < layers.size() - 1:
			x = relu(x)

	return denormalize_output(x)


## 重置隐状态（换回合 / 进梦境时调用），保证时序从零开始。
func reset_hidden():
	hidden = []
	for i in range(hidden_size):
		hidden.append(0.0)


## GRU 单步：z/r 门控 + 候选 n，输出新隐状态 h'。
func gru_step(h: Array, x: Array) -> Array:
	var c: Array = []
	for v in h:
		c.append(v)
	for v in x:
		c.append(v)
	var z := sig_arr(add_bias(matvec_mul(Wz, c), bz))
	var r := sig_arr(add_bias(matvec_mul(Wr, c), br))
	var hr := mul_arr(r, h)
	var cr: Array = []
	for v in hr:
		cr.append(v)
	for v in x:
		cr.append(v)
	var n := tanh_arr(add_bias(matvec_mul(Wn, cr), bn))
	var hnew: Array = []
	for i in range(hidden_size):
		hnew.append((1.0 - z[i]) * h[i] + z[i] * n[i])
	return hnew


func predict_recurrent(state: Array, action: Array) -> Array:
	var x := normalize_input(state.duplicate())
	x.append_array(action.duplicate())
	hidden = gru_step(hidden, x)
	var y := add_bias(matvec_mul(Wout_r, hidden), bout_r)
	return denormalize_output(y)


func sig_arr(a: Array) -> Array:
	var out: Array = []
	for v in a:
		out.append(1.0 / (1.0 + exp(-v)))
	return out


func tanh_arr(a: Array) -> Array:
	var out: Array = []
	for v in a:
		var e := exp(2.0 * v)
		out.append((e - 1.0) / (e + 1.0))
	return out


func mul_arr(a: Array, b: Array) -> Array:
	var out: Array = []
	for i in range(a.size()):
		out.append(a[i] * b[i])
	return out


## 多步「想象 / Rollout」：把模型自己的输出喂回输入，连续推演未来若干步。
## init_state: [x, y, vx, vy]
## action_seq: 每步动作 [ax, ay]；比 steps 短时，自动复用最后一个动作。
## 返回长度为 steps 的状态序列，每个为 [x, y, vx, vy]。
func predict_rollout(init_state: Array, action_seq: Array, steps: int) -> Array:
	if not loaded:
		return []

	if recurrent:
		var h: Array = hidden.duplicate()
		var states: Array = []
		var s: Array = init_state.duplicate()
		for i in range(steps):
			var a: Array
			if i < action_seq.size():
				a = action_seq[i]
			elif action_seq.size() > 0:
				a = action_seq[action_seq.size() - 1]
			else:
				a = [0.0, 0.0]
			var x := normalize_input(s.duplicate())
			x.append_array(a.duplicate())
			h = gru_step(h, x)
			var y := add_bias(matvec_mul(Wout_r, h), bout_r)
			s = denormalize_output(y)
			states.append(s.duplicate())
		return states

	var states: Array = []
	var s: Array = init_state.duplicate()
	for i in range(steps):
		var a: Array
		if i < action_seq.size():
			a = action_seq[i]
		elif action_seq.size() > 0:
			a = action_seq[action_seq.size() - 1]
		else:
			a = [0.0, 0.0]
		s = predict(s, a)
		states.append(s.duplicate())
	return states


func normalize_input(x: Array) -> Array:
	var out: Array = []
	for i in range(x.size()):
		out.append((x[i] - input_mean[i]) / input_std[i])
	return out


func denormalize_output(x: Array) -> Array:
	var out: Array = []
	for i in range(x.size()):
		out.append(x[i] * output_std[i] + output_mean[i])
	return out


func relu(x: Array) -> Array:
	var out: Array = []
	for v in x:
		out.append(max(0.0, v))
	return out


func matvec_mul(W: Array, x: Array) -> Array:
	var out: Array = []
	for i in range(W.size()):
		var row: Array = W[i]
		var sum := 0.0
		for j in range(row.size()):
			sum += row[j] * x[j]
		out.append(sum)
	return out


func add_bias(x: Array, b: Array) -> Array:
	var out: Array = []
	for i in range(x.size()):
		out.append(x[i] + b[i])
	return out
