// player.js — a small Shadertoy-compatible WebGL2 fragment-shader player.
//
// It wraps a user shader that defines `mainImage(out vec4, in vec2)` and feeds
// it the standard Shadertoy uniforms (iResolution, iTime, iTimeDelta, iFrame,
// iMouse, iDate). No build step, no dependencies — pure WebGL2.

const VERTEX_SRC = `#version 300 es
// Fullscreen triangle — no attributes needed, positions come from gl_VertexID.
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

// Prepended to every user shader. `#line 1` resets the line counter so compile
// errors report line numbers relative to the user's own file.
const FRAGMENT_PRELUDE = `#version 300 es
precision highp float;
precision highp int;

uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse;
uniform vec4  iDate;

out vec4 __outColor;

#line 1
`;

const FRAGMENT_EPILOGUE = `
void main() {
    mainImage(__outColor, gl_FragCoord.xy);
}
`;

class ShaderPlayer {
    constructor(canvas, statusEl) {
        this.canvas = canvas;
        this.statusEl = statusEl;
        this.gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
        if (!this.gl) {
            throw new Error("WebGL2 is not available in this browser.");
        }

        this.program = null;
        this.uniforms = {};
        this.startTime = performance.now();
        this.pausedAt = null;          // ms into the timeline when paused
        this.elapsedBeforePause = 0;
        this.frame = 0;
        this.lastFrameTime = performance.now();
        this.mouse = { x: 0, y: 0, clickX: 0, clickY: 0, down: false };
        this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        this.fps = 0;
        this._fpsAccum = 0;
        this._fpsFrames = 0;
        this._fpsLast = performance.now();

        this._vao = this.gl.createVertexArray();
        this._bindMouse();
        this._resize();
        window.addEventListener("resize", () => this._resize());
    }

    _bindMouse() {
        const toPixels = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
            // WebGL y is bottom-up; Shadertoy iMouse is bottom-up too.
            const y = (rect.height - (e.clientY - rect.top)) * (this.canvas.height / rect.height);
            return { x, y };
        };
        this.canvas.addEventListener("mousedown", (e) => {
            const p = toPixels(e);
            this.mouse.down = true;
            this.mouse.x = p.x; this.mouse.y = p.y;
            this.mouse.clickX = p.x; this.mouse.clickY = p.y;
        });
        window.addEventListener("mouseup", () => { this.mouse.down = false; });
        window.addEventListener("mousemove", (e) => {
            if (!this.mouse.down) return;
            const p = toPixels(e);
            this.mouse.x = p.x; this.mouse.y = p.y;
        });
    }

    _resize() {
        const w = Math.floor(this.canvas.clientWidth * this.pixelRatio);
        const h = Math.floor(this.canvas.clientHeight * this.pixelRatio);
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
        }
    }

    _compile(type, src) {
        const gl = this.gl;
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(sh) || "unknown error";
            gl.deleteShader(sh);
            throw new ShaderCompileError(log);
        }
        return sh;
    }

    // Compile a user fragment shader. Returns {ok:true} or {ok:false, error}.
    load(userSource) {
        const gl = this.gl;
        const fragSrc = FRAGMENT_PRELUDE + userSource + FRAGMENT_EPILOGUE;
        let vs, fs, prog;
        try {
            vs = this._compile(gl.VERTEX_SHADER, VERTEX_SRC);
            fs = this._compile(gl.FRAGMENT_SHADER, fragSrc);
        } catch (err) {
            if (vs) gl.deleteShader(vs);
            return { ok: false, error: this._formatError(err) };
        }

        prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            const log = gl.getProgramInfoLog(prog) || "link failed";
            gl.deleteProgram(prog);
            return { ok: false, error: log };
        }

        if (this.program) gl.deleteProgram(this.program);
        this.program = prog;
        for (const name of ["iResolution", "iTime", "iTimeDelta", "iFrame", "iMouse", "iDate"]) {
            this.uniforms[name] = gl.getUniformLocation(prog, name);
        }
        this.resetTime();
        return { ok: true };
    }

    // The prelude ends with `#line 1`, so a conformant driver already reports
    // user-code errors relative to the user's own file — no remap needed.
    _formatError(err) {
        if (!(err instanceof ShaderCompileError)) return String(err.message || err);
        return err.log.trim();
    }

    resetTime() {
        this.startTime = performance.now();
        this.elapsedBeforePause = 0;
        this.frame = 0;
        if (this.pausedAt !== null) this.pausedAt = this.startTime;
    }

    get paused() { return this.pausedAt !== null; }

    setPaused(paused) {
        if (paused && this.pausedAt === null) {
            this.pausedAt = performance.now();
        } else if (!paused && this.pausedAt !== null) {
            // Shift the start forward by the paused duration so time is continuous.
            this.startTime += performance.now() - this.pausedAt;
            this.pausedAt = null;
        }
    }

    _timeSeconds() {
        const now = this.pausedAt !== null ? this.pausedAt : performance.now();
        return (now - this.startTime) / 1000;
    }

    render() {
        const gl = this.gl;
        this._resize();
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        if (!this.program) {
            gl.clearColor(0.0, 0.0, 0.0, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            return;
        }

        const now = performance.now();
        const dt = (now - this.lastFrameTime) / 1000;
        this.lastFrameTime = now;

        // FPS (smoothed over ~0.5s windows).
        this._fpsFrames++;
        if (now - this._fpsLast >= 500) {
            this.fps = Math.round((this._fpsFrames * 1000) / (now - this._fpsLast));
            this._fpsFrames = 0;
            this._fpsLast = now;
        }

        gl.useProgram(this.program);
        gl.bindVertexArray(this._vao);

        const t = this._timeSeconds();
        if (this.uniforms.iResolution) gl.uniform3f(this.uniforms.iResolution, this.canvas.width, this.canvas.height, 1.0);
        if (this.uniforms.iTime) gl.uniform1f(this.uniforms.iTime, t);
        if (this.uniforms.iTimeDelta) gl.uniform1f(this.uniforms.iTimeDelta, dt);
        if (this.uniforms.iFrame) gl.uniform1i(this.uniforms.iFrame, this.frame);
        if (this.uniforms.iMouse) {
            // Shadertoy convention: zw sign encodes button state.
            const zw = this.mouse.down ? [this.mouse.clickX, this.mouse.clickY] : [-this.mouse.clickX, -this.mouse.clickY];
            gl.uniform4f(this.uniforms.iMouse, this.mouse.x, this.mouse.y, zw[0], zw[1]);
        }
        if (this.uniforms.iDate) {
            const d = new Date();
            const secs = d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds() / 1000;
            gl.uniform4f(this.uniforms.iDate, d.getFullYear(), d.getMonth() + 1, d.getDate(), secs);
        }

        gl.drawArrays(gl.TRIANGLES, 0, 3);
        if (this.pausedAt === null) this.frame++;
    }

    get timeSeconds() { return this._timeSeconds(); }
}

class ShaderCompileError extends Error {
    constructor(log) {
        super("shader compile error");
        this.log = log;
    }
}

export { ShaderPlayer };
