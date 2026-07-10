// app.js — wires the ShaderPlayer to the page UI: shader picker, transport
// controls (play/pause/reset), live HUD (time/frame/fps), and error display.

import { ShaderPlayer } from "./player.js";

const canvas = document.getElementById("gl");
const els = {
    picker: document.getElementById("shader-picker"),
    playPause: document.getElementById("play-pause"),
    reset: document.getElementById("reset"),
    reload: document.getElementById("reload"),
    time: document.getElementById("hud-time"),
    frame: document.getElementById("hud-frame"),
    fps: document.getElementById("hud-fps"),
    res: document.getElementById("hud-res"),
    desc: document.getElementById("shader-desc"),
    errors: document.getElementById("errors"),
};

let player;
try {
    player = new ShaderPlayer(canvas);
} catch (err) {
    showFatal(err.message);
    throw err;
}

let currentFile = null;

function showFatal(msg) {
    els.errors.textContent = msg;
    els.errors.classList.add("visible");
}

function showErrors(msg) {
    els.errors.textContent = msg;
    els.errors.classList.toggle("visible", !!msg);
}

async function fetchText(url) {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.text();
}

async function loadShader(file) {
    try {
        const src = await fetchText(`shaders/${file}`);
        const result = player.load(src);
        if (result.ok) {
            showErrors("");
            currentFile = file;
        } else {
            showErrors(`Compile failed in ${file}:\n\n${result.error}`);
        }
    } catch (err) {
        showErrors(
            `Could not load shaders/${file} (${err.message}).\n\n` +
            `If you opened this file directly (file://), the browser blocks fetch().\n` +
            `Serve the folder over HTTP instead:\n\n` +
            `    python3 -m http.server 8080\n\n` +
            `then open http://localhost:8080/`
        );
    }
}

async function init() {
    let manifest;
    try {
        manifest = JSON.parse(await fetchText("shaders/manifest.json"));
    } catch (err) {
        showErrors(
            `Could not load shaders/manifest.json (${err.message}).\n\n` +
            `Serve the folder over HTTP:  python3 -m http.server 8080  →  http://localhost:8080/`
        );
        requestAnimationFrame(loop);
        return;
    }

    els.picker.innerHTML = "";
    for (const s of manifest.shaders) {
        const opt = document.createElement("option");
        opt.value = s.file;
        opt.textContent = s.name;
        opt.dataset.description = s.description || "";
        els.picker.appendChild(opt);
    }

    els.picker.addEventListener("change", () => {
        const opt = els.picker.selectedOptions[0];
        els.desc.textContent = opt ? opt.dataset.description : "";
        loadShader(els.picker.value);
    });

    els.playPause.addEventListener("click", () => {
        player.setPaused(!player.paused);
        updatePlayPause();
    });
    els.reset.addEventListener("click", () => player.resetTime());
    els.reload.addEventListener("click", () => { if (currentFile) loadShader(currentFile); });

    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "SELECT") return;
        if (e.code === "Space") { e.preventDefault(); player.setPaused(!player.paused); updatePlayPause(); }
        else if (e.key === "r") player.resetTime();
    });

    if (manifest.shaders.length) {
        els.desc.textContent = manifest.shaders[0].description || "";
        await loadShader(manifest.shaders[0].file);
    }
    updatePlayPause();
    requestAnimationFrame(loop);
}

function updatePlayPause() {
    els.playPause.textContent = player.paused ? "▶ Play" : "⏸ Pause";
}

let hudLast = 0;
function loop(now) {
    player.render();
    if (now - hudLast > 100) {
        hudLast = now;
        els.time.textContent = player.timeSeconds.toFixed(2) + "s";
        els.frame.textContent = player.frame;
        els.fps.textContent = player.fps;
        els.res.textContent = `${canvas.width}×${canvas.height}`;
    }
    requestAnimationFrame(loop);
}

init();
