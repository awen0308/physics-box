shader_type canvas_item;

uniform vec3 top_color : source_color = vec3(0.02, 0.04, 0.09);
uniform vec3 bottom_color : source_color = vec3(0.05, 0.02, 0.13);
uniform vec2 resolution = vec2(1152.0, 648.0);
uniform float time = 0.0;

void fragment() {
	vec2 uv = UV;
	vec3 col = mix(bottom_color, top_color, uv.y);
	vec2 g = abs(fract(uv * resolution / 46.0) - 0.5);
	float line = smoothstep(0.46, 0.5, max(g.x, g.y));
	col += vec3(0.05, 0.14, 0.22) * line * 0.35;
	float shimmer = 0.02 * sin(uv.x * 18.0 + time * 0.6);
	col += vec3(0.02, 0.05, 0.08) * shimmer;
	float vig = smoothstep(1.3, 0.25, length(uv - 0.5));
	col *= mix(0.45, 1.0, vig);
	COLOR = vec4(col, 1.0);
}
