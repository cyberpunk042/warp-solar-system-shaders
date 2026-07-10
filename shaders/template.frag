// template — minimal starter. Copy this to start a new shader.
//
// Available uniforms (Shadertoy-compatible, injected by the player):
//   uniform vec3  iResolution;  // viewport resolution in pixels (z = 1.0)
//   uniform float iTime;        // seconds since start
//   uniform float iTimeDelta;   // seconds since last frame
//   uniform int   iFrame;       // frame counter
//   uniform vec4  iMouse;       // xy = current px while dragging, zw = click origin
//   uniform vec4  iDate;        // (year, month, day, seconds-in-day)
//
// Entry point is mainImage(); the player wires it to gl_FragCoord for you.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalized coords, centered, aspect-corrected on the shorter axis.
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Simple animated gradient so a fresh shader shows something immediately.
    float d = length(uv);
    vec3 col = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0.0, 2.0, 4.0));
    col *= smoothstep(0.9, 0.2, d);

    fragColor = vec4(col, 1.0);
}
