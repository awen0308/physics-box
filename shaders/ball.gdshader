shader_type canvas_item;

uniform vec3 core_color : source_color = vec3(0.7, 0.95, 1.0);
uniform vec3 glow_color : source_color = vec3(0.1, 0.4, 0.9);

void fragment() {
	vec2 uv = UV - 0.5;
	float d = length(uv) * 2.0;
	float core = smoothstep(0.45, 0.0, d);
	float glow = smoothstep(1.0, 0.2, d);
	vec3 col = mix(glow_color, core_color, core);
	float alpha = 1.0 - smoothstep(0.97, 1.0, d);
	col *= 1.5;
	COLOR = vec4(col, alpha);
}
