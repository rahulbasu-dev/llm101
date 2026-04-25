# Animated Transformer Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an animated "Visualize" tab to the Gradio UI that shows the transformer forward pass as a synchronized three-panel animation (architecture diagram, 6×6 attention grid, activation flow bars).

**Architecture:** Python collects all tensors in one hooked forward pass (`visualize_anim.py`), serializes to JSON, and injects into an HTML/CSS/JS template (`templates/visualize.html`). Gradio serves the result via `gr.HTML`. The JS animation engine steps through layers, updating all three panels in sync. Click interaction on the attention grid expands a detail panel — purely client-side, no round-trip.

**Tech Stack:** Python (PyTorch hooks, JSON), HTML/CSS/JS (Canvas API for heatmaps, CSS transitions for bars), Gradio `gr.HTML` component.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `visualize_anim.py` (new) | `collect_viz_data(model, tokenizer, text, config)` — runs a hooked forward pass across ALL layers, collects attention weights + hidden norms + shapes, returns a dict. |
| `templates/visualize.html` (new) | Self-contained HTML/CSS/JS template. Receives data via `{{VIZ_DATA_JSON}}` placeholder. Three-panel layout, animation engine, click-to-expand detail, playback controls. |
| `app.py` (modify) | New "Visualize" tab between Attention and Generate. Handler calls `collect_viz_data()`, reads template, injects JSON, returns HTML string. |
| `tests/test_visualize_anim.py` (new) | Unit tests for `collect_viz_data()` — structure, shapes, value ranges. |
| `CLAUDE.md` (modify) | Update tab count (7→8) and tab list. |

---

### Task 1: Data collection — `visualize_anim.py`

**Files:**
- Create: `visualize_anim.py`
- Test: `tests/test_visualize_anim.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualize_anim.py`:

```python
"""Tests for visualize_anim — tensor collection for the animated visualization."""

import torch
import pytest

from config import NanoLLMConfig
from tokenizer import BPETokenizer
from model import NanoLLM
from visualize_anim import collect_viz_data


def test_collect_viz_data_structure(tiny_model, trained_tokenizer, tiny_config):
    """collect_viz_data returns the expected JSON-serializable structure."""
    data = collect_viz_data(tiny_model, trained_tokenizer, "The cat sat", tiny_config)

    assert isinstance(data, dict)
    assert data["n_layers"] == tiny_config.n_layers  # 2
    assert data["n_heads"] == tiny_config.n_heads    # 2
    assert data["d_model"] == tiny_config.d_model    # 32
    assert data["d_head"] == tiny_config.d_head      # 16
    assert isinstance(data["tokens"], list)
    assert len(data["tokens"]) > 0
    assert isinstance(data["layers"], list)
    assert len(data["layers"]) == tiny_config.n_layers


def test_collect_viz_data_layer_contents(tiny_model, trained_tokenizer, tiny_config):
    """Each layer entry has attention weights, hidden norm, and shapes."""
    data = collect_viz_data(tiny_model, trained_tokenizer, "The cat sat", tiny_config)
    T = len(data["tokens"])

    for i, layer in enumerate(data["layers"]):
        assert layer["layer_idx"] == i

        # attn_weights: [n_heads][T][T], values in [0, 1]
        aw = layer["attn_weights"]
        assert len(aw) == tiny_config.n_heads
        assert len(aw[0]) == T
        assert len(aw[0][0]) == T
        # Check post-softmax range
        assert all(0.0 <= v <= 1.0 for row in aw[0] for v in row)

        # hidden_norm: positive scalar
        assert isinstance(layer["hidden_norm"], float)
        assert layer["hidden_norm"] > 0

        # shapes: dict with expected keys
        shapes = layer["shapes"]
        assert "input" in shapes
        assert "q" in shapes
        assert "k" in shapes
        assert "v" in shapes
        assert "attn_out" in shapes
        assert "ffn_out" in shapes
        assert "output" in shapes


def test_collect_viz_data_is_json_serializable(tiny_model, trained_tokenizer, tiny_config):
    """The returned dict must be JSON-serializable (no tensors, no numpy)."""
    import json
    data = collect_viz_data(tiny_model, trained_tokenizer, "hello", tiny_config)
    # This will raise TypeError if any value is a tensor/ndarray
    json_str = json.dumps(data)
    assert len(json_str) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `NANOLLM_ALLOW_CPU=1 python -m pytest tests/test_visualize_anim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'visualize_anim'`

- [ ] **Step 3: Write the implementation**

Create `visualize_anim.py`:

```python
"""Collect tensor data for the animated transformer visualization.

Runs a single hooked forward pass across ALL layers and collects:
  - attention weights (post-softmax, post-mask) for every layer and head
  - hidden state L2 norms after each layer's residual connection
  - tensor shape strings at each stage

Returns a JSON-serializable dict ready to inject into the HTML template.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from config import NanoLLMConfig
from tokenizer import BPETokenizer
from model import NanoLLM


def collect_viz_data(
    model: NanoLLM,
    tokenizer: BPETokenizer,
    text: str,
    config: NanoLLMConfig,
) -> dict:
    """Run a hooked forward pass and collect visualization data.

    Args:
        model: The NanoLLM model (eval mode, any device).
        tokenizer: Trained BPE tokenizer.
        text: Input text to visualize.
        config: Model config (for d_model, n_layers, etc.).

    Returns:
        JSON-serializable dict matching the spec's data structure.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # Tokenize
    token_ids = tokenizer.encode(text, add_special=False)
    # Truncate to max_seq_len
    token_ids = token_ids[:config.max_seq_len]
    token_labels = [tokenizer.decode_token(tid) for tid in token_ids]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    B, T = input_ids.shape

    # Storage for all layers
    all_layers_data = []

    # Install hooks on every layer (not just one target like ForwardCapture)
    hooks = []

    # Capture hidden state after each block's residual connection
    hidden_norms = {}

    def _make_block_hook(layer_idx):
        def hook(module, inp, out):
            # out is the block output (after both residual connections)
            norm = out.detach().float().norm(dim=-1).mean().item()
            hidden_norms[layer_idx] = norm
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(_make_block_hook(i)))

    # Capture attention weights from every layer
    attn_data = {}

    def _make_attn_hook(layer_idx):
        def hook(module, inp):
            x_norm = inp[0]
            b, t, c = x_norm.shape

            qkv = module.qkv_proj(x_norm)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(b, t, module.n_heads, module.d_head).transpose(1, 2)
            k = k.view(b, t, module.n_heads, module.d_head).transpose(1, 2)
            v = v.view(b, t, module.n_heads, module.d_head).transpose(1, 2)

            q_rot = module.rope(q, t, start_pos=0)
            k_rot = module.rope(k, t, start_pos=0)

            scale = 1.0 / math.sqrt(module.d_head)
            scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale
            masked = scores.masked_fill(
                module.causal_mask[:, :, :t, :t] == 0, float("-inf")
            )
            weights = F.softmax(masked, dim=-1)

            attn_data[layer_idx] = {
                "weights": weights.detach().cpu(),  # (B, n_heads, T, T)
                "shapes": {
                    "input": str(tuple(x_norm.shape)),
                    "qkv": str(tuple(qkv.shape)),
                    "q": str(tuple(q.shape)),
                    "k": str(tuple(k.shape)),
                    "v": str(tuple(v.shape)),
                },
            }
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_pre_hook(_make_attn_hook(i)))

    # Capture FFN output shapes
    ffn_shapes = {}

    def _make_ffn_hook(layer_idx):
        def hook(module, inp, out):
            ffn_shapes[layer_idx] = {
                "attn_out": str(tuple(inp[0].shape)),
                "ffn_out": str(tuple(out.shape)),
                "output": str(tuple(out.shape)),
            }
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.ffn.register_forward_hook(_make_ffn_hook(i)))

    # Run forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove all hooks
    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    # Assemble the result
    for i in range(config.n_layers):
        ad = attn_data[i]
        weights_tensor = ad["weights"][0]  # (n_heads, T, T)

        # Convert attention weights to nested Python lists
        attn_weights_list = []
        for head in range(config.n_heads):
            head_matrix = weights_tensor[head].tolist()
            attn_weights_list.append(head_matrix)

        shapes = {**ad["shapes"], **ffn_shapes.get(i, {})}

        all_layers_data.append({
            "layer_idx": i,
            "attn_weights": attn_weights_list,
            "hidden_norm": hidden_norms.get(i, 0.0),
            "shapes": shapes,
        })

    return {
        "tokens": token_labels,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_model": config.d_model,
        "d_head": config.d_head,
        "layers": all_layers_data,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `NANOLLM_ALLOW_CPU=1 python -m pytest tests/test_visualize_anim.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add visualize_anim.py tests/test_visualize_anim.py
git commit -m "feat: add visualize_anim.py — tensor collection for animated viz"
```

---

### Task 2: HTML/CSS/JS animation template

**Files:**
- Create: `templates/visualize.html`

This is the largest single file. It contains the three-panel layout, animation engine, canvas-based heatmaps, click interaction, and playback controls. All self-contained — no external dependencies.

- [ ] **Step 1: Create the templates directory**

```bash
mkdir -p templates
```

- [ ] **Step 2: Write the HTML template**

Create `templates/visualize.html` — a complete, self-contained HTML document that reads `VIZ_DATA` from an inline `<script>` tag (injected by Python). The structure:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
/* ── Reset & dark theme ── */
* { margin: 0; padding: 0; box-sizing: border-box; }
body, html { background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif; }

/* ── Controls bar ── */
.controls {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; background: #1e293b; border-radius: 8px; margin-bottom: 12px;
    flex-wrap: wrap;
}
.controls button {
    padding: 6px 16px; border: none; border-radius: 6px;
    background: #3b82f6; color: white; cursor: pointer; font-size: 14px;
    transition: background 0.2s;
}
.controls button:hover { background: #2563eb; }
.controls button.active { background: #f59e0b; color: #0f172a; }
.controls label { font-size: 13px; color: #94a3b8; display: flex; align-items: center; gap: 6px; }
.controls input[type=range] { width: 100px; accent-color: #3b82f6; }
.controls input[type=checkbox] { accent-color: #3b82f6; }
.progress { font-size: 13px; color: #93c5fd; margin-left: auto; font-weight: 600; }

/* ── Three-column layout ── */
.viz-container {
    display: grid; grid-template-columns: 220px 1fr 200px;
    gap: 12px; min-height: 500px;
}

/* ── Left: Architecture ── */
.arch-panel { display: flex; flex-direction: column; gap: 6px; }
.arch-block {
    padding: 8px 10px; border-radius: 6px; border: 2px solid #334155;
    background: #1e293b; transition: all 0.4s ease; font-size: 12px;
}
.arch-block.completed { border-color: #3b82f6; background: #1e3a5f; }
.arch-block.active {
    border-color: #f59e0b; background: #78350f;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.4);
}
.arch-block .label { font-weight: 700; color: #e2e8f0; }
.arch-block .shape { font-size: 10px; color: #94a3b8; font-family: monospace; }
.arch-block .sub {
    font-size: 10px; color: #64748b; margin-top: 2px;
    line-height: 1.4;
}
.arch-arrow { text-align: center; color: #475569; font-size: 16px; line-height: 1; }
.arch-special {
    padding: 6px 10px; border-radius: 6px; background: #0f172a;
    border: 1px solid #334155; font-size: 11px; text-align: center; color: #94a3b8;
}

/* ── Center: Attention grid ── */
.grid-panel { display: flex; flex-direction: column; }
.grid-header { font-size: 13px; font-weight: 700; color: #93c5fd; margin-bottom: 8px; text-align: center; }
.attn-grid {
    display: grid;
    gap: 4px;
    flex: 1;
}
.attn-cell {
    border-radius: 4px; background: #1e293b; cursor: pointer;
    position: relative; overflow: hidden;
    transition: box-shadow 0.3s, opacity 0.4s;
    border: 2px solid transparent;
    min-height: 50px;
}
.attn-cell canvas { width: 100%; height: 100%; display: block; }
.attn-cell.placeholder { opacity: 0.3; }
.attn-cell.filled { opacity: 1; }
.attn-cell.selected { border-color: #f59e0b; box-shadow: 0 0 8px rgba(245,158,11,0.5); }
.attn-cell .cell-label {
    position: absolute; bottom: 2px; right: 4px;
    font-size: 9px; color: #94a3b8; pointer-events: none;
}
.grid-labels { display: flex; justify-content: space-around; margin-top: 4px; }
.grid-labels span { font-size: 10px; color: #64748b; }
.row-label {
    font-size: 10px; color: #64748b; writing-mode: vertical-lr;
    transform: rotate(180deg); display: flex; align-items: center;
    justify-content: center; padding: 0 4px;
}

/* ── Detail panel (below grid, shown on click) ── */
.detail-panel {
    margin-top: 12px; padding: 16px; background: #1e293b;
    border-radius: 8px; border: 1px solid #334155;
    display: none; /* shown by JS */
}
.detail-panel.visible { display: flex; gap: 20px; align-items: flex-start; }
.detail-panel canvas { border-radius: 4px; }
.detail-info { font-size: 13px; line-height: 1.6; }
.detail-info .detail-title { font-weight: 700; color: #f59e0b; margin-bottom: 8px; }
.detail-info .shape-line { font-family: monospace; font-size: 12px; color: #93c5fd; }

/* ── Right: Activation flow ── */
.flow-panel { display: flex; flex-direction: column; gap: 4px; }
.flow-header { font-size: 13px; font-weight: 700; color: #93c5fd; margin-bottom: 8px; text-align: center; }
.flow-bar-row {
    display: flex; align-items: center; gap: 8px; height: 28px;
}
.flow-label { font-size: 11px; color: #94a3b8; width: 24px; text-align: right; font-family: monospace; }
.flow-track { flex: 1; height: 18px; background: #1e293b; border-radius: 4px; overflow: hidden; position: relative; }
.flow-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.6s ease, background-color 0.6s ease;
    width: 0%;
}
.flow-value { font-size: 10px; color: #64748b; width: 40px; font-family: monospace; }
</style>
</head>
<body>
<script>
// Data injected by Python
const VIZ_DATA = {{VIZ_DATA_JSON}};
</script>

<script>
// ═══════════════════════════════════════════════════════════
// Animation Engine
// ═══════════════════════════════════════════════════════════

const state = {
    currentLayer: -1,   // -1 = pre-animation
    playing: false,
    speed: 0.8,
    loop: false,
    timer: null,
    selectedCell: null, // {layer, head} or null
};

const N_LAYERS = VIZ_DATA.n_layers;
const N_HEADS = VIZ_DATA.n_heads;
const TOKENS = VIZ_DATA.tokens;
const T = TOKENS.length;

// ── Color helpers ──
function attnColor(value) {
    // Sequential blue: white(0) → dark blue(1)
    const r = Math.round(255 * (1 - value));
    const g = Math.round(255 * (1 - value * 0.8));
    const b = 255;
    return `rgb(${r},${g},${b})`;
}

function normColor(norm, maxNorm) {
    const ratio = Math.min(norm / maxNorm, 1);
    if (ratio < 0.5) return '#22c55e';       // green
    if (ratio < 0.75) return '#f59e0b';      // amber
    return '#ef4444';                          // red
}

// ── Canvas heatmap rendering ──
function drawMiniHeatmap(canvas, weights) {
    // weights: T x T array
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cellW = w / T;
    const cellH = h / T;
    for (let qi = 0; qi < T; qi++) {
        for (let ki = 0; ki < T; ki++) {
            ctx.fillStyle = attnColor(weights[qi][ki]);
            ctx.fillRect(ki * cellW, qi * cellH, cellW + 0.5, cellH + 0.5);
        }
    }
}

function drawDetailHeatmap(canvas, weights, size) {
    const ctx = canvas.getContext('2d');
    canvas.width = size;
    canvas.height = size;
    const cellW = size / T;
    const cellH = size / T;
    ctx.clearRect(0, 0, size, size);

    for (let qi = 0; qi < T; qi++) {
        for (let ki = 0; ki < T; ki++) {
            ctx.fillStyle = attnColor(weights[qi][ki]);
            ctx.fillRect(ki * cellW, qi * cellH, cellW + 0.5, cellH + 0.5);
        }
    }

    // Draw token labels
    ctx.fillStyle = '#e2e8f0';
    ctx.font = `${Math.max(9, Math.min(12, size / T - 2))}px monospace`;
    ctx.textAlign = 'center';
    for (let i = 0; i < T; i++) {
        const label = TOKENS[i].length > 6 ? TOKENS[i].slice(0, 5) + '…' : TOKENS[i];
        // Bottom (key labels)
        ctx.save();
        ctx.translate(i * cellW + cellW / 2, size - 2);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(label, 0, 0);
        ctx.restore();
    }
}

// ── Build DOM ──
function buildUI() {
    const root = document.body;

    // Controls
    const controls = document.createElement('div');
    controls.className = 'controls';
    controls.innerHTML = `
        <button id="btn-play">▶ Play</button>
        <button id="btn-step-prev">◀</button>
        <button id="btn-step-next">▶</button>
        <button id="btn-reset">⟳ Reset</button>
        <label>Speed: <input type="range" id="speed-slider" min="0.3" max="1.5" step="0.1" value="0.8">
            <span id="speed-val">0.8s</span></label>
        <label><input type="checkbox" id="loop-toggle"> Loop</label>
        <span class="progress" id="progress">Ready — ${T} tokens, ${N_LAYERS} layers × ${N_HEADS} heads</span>
    `;
    root.appendChild(controls);

    // Three-column container
    const container = document.createElement('div');
    container.className = 'viz-container';
    container.style.gridTemplateColumns = `220px 1fr 200px`;

    // ── Left: Architecture ──
    const archPanel = document.createElement('div');
    archPanel.className = 'arch-panel';

    // Embedding block
    archPanel.innerHTML = `
        <div class="arch-special" id="arch-embed">
            <div style="font-weight:700">Token Embedding</div>
            <div class="shape">(1, ${T}, ${VIZ_DATA.d_model})</div>
        </div>
        <div class="arch-arrow">↓</div>
    `;

    // Transformer blocks
    for (let i = 0; i < N_LAYERS; i++) {
        const shapes = VIZ_DATA.layers[i].shapes;
        archPanel.innerHTML += `
            <div class="arch-block" id="arch-block-${i}">
                <div class="label">Block ${i}</div>
                <div class="sub">
                    RMSNorm → Self-Attention → + residual<br>
                    RMSNorm → FFN (SwiGLU) → + residual
                </div>
                <div class="shape">${shapes.input || ''}</div>
            </div>
            ${i < N_LAYERS - 1 ? '<div class="arch-arrow">↓</div>' : ''}
        `;
    }

    // Final norm + LM head
    archPanel.innerHTML += `
        <div class="arch-arrow">↓</div>
        <div class="arch-special" id="arch-output">
            <div style="font-weight:700">RMSNorm → LM Head</div>
            <div class="shape">→ logits</div>
        </div>
    `;

    container.appendChild(archPanel);

    // ── Center: Attention grid ──
    const gridWrapper = document.createElement('div');
    gridWrapper.className = 'grid-panel';
    gridWrapper.innerHTML = `<div class="grid-header">Attention Weights — All Heads (rows = layers, cols = heads)</div>`;

    const grid = document.createElement('div');
    grid.className = 'attn-grid';
    grid.style.gridTemplateColumns = `repeat(${N_HEADS}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${N_LAYERS}, 1fr)`;

    for (let li = 0; li < N_LAYERS; li++) {
        for (let hi = 0; hi < N_HEADS; hi++) {
            const cell = document.createElement('div');
            cell.className = 'attn-cell placeholder';
            cell.id = `cell-${li}-${hi}`;
            cell.dataset.layer = li;
            cell.dataset.head = hi;

            const canvas = document.createElement('canvas');
            canvas.width = Math.max(T * 8, 50);
            canvas.height = Math.max(T * 8, 50);
            cell.appendChild(canvas);

            const label = document.createElement('div');
            label.className = 'cell-label';
            label.textContent = `L${li}H${hi}`;
            cell.appendChild(label);

            cell.addEventListener('click', () => onCellClick(li, hi));
            grid.appendChild(cell);
        }
    }

    gridWrapper.appendChild(grid);

    // Head labels below grid
    const headLabels = document.createElement('div');
    headLabels.className = 'grid-labels';
    for (let hi = 0; hi < N_HEADS; hi++) {
        const s = document.createElement('span');
        s.textContent = `Head ${hi}`;
        headLabels.appendChild(s);
    }
    gridWrapper.appendChild(headLabels);

    // Detail panel (hidden initially)
    const detail = document.createElement('div');
    detail.className = 'detail-panel';
    detail.id = 'detail-panel';
    detail.innerHTML = `
        <canvas id="detail-canvas" width="300" height="300"></canvas>
        <div class="detail-info" id="detail-info"></div>
    `;
    gridWrapper.appendChild(detail);

    container.appendChild(gridWrapper);

    // ── Right: Activation flow ──
    const flowPanel = document.createElement('div');
    flowPanel.className = 'flow-panel';
    flowPanel.innerHTML = `<div class="flow-header">Hidden State Norms</div>`;

    // Compute max norm for scaling
    const maxNorm = Math.max(...VIZ_DATA.layers.map(l => l.hidden_norm), 1);

    for (let i = 0; i < N_LAYERS; i++) {
        flowPanel.innerHTML += `
            <div class="flow-bar-row">
                <span class="flow-label">L${i}</span>
                <div class="flow-track">
                    <div class="flow-fill" id="flow-bar-${i}"></div>
                </div>
                <span class="flow-value" id="flow-val-${i}">—</span>
            </div>
        `;
    }
    container.appendChild(flowPanel);

    root.appendChild(container);

    // ── Wire up controls ──
    document.getElementById('btn-play').addEventListener('click', togglePlay);
    document.getElementById('btn-step-prev').addEventListener('click', stepPrev);
    document.getElementById('btn-step-next').addEventListener('click', stepNext);
    document.getElementById('btn-reset').addEventListener('click', reset);
    document.getElementById('speed-slider').addEventListener('input', (e) => {
        state.speed = parseFloat(e.target.value);
        document.getElementById('speed-val').textContent = state.speed.toFixed(1) + 's';
        if (state.playing) { clearInterval(state.timer); state.timer = setInterval(tick, state.speed * 1000); }
    });
    document.getElementById('loop-toggle').addEventListener('change', (e) => {
        state.loop = e.target.checked;
    });
}

// ── Animation tick ──
function tick() {
    const nextLayer = state.currentLayer + 1;
    if (nextLayer >= N_LAYERS) {
        if (state.loop) {
            reset();
            state.playing = true;
            state.timer = setInterval(tick, state.speed * 1000);
            document.getElementById('btn-play').textContent = '⏸ Pause';
            tick(); // start immediately
        } else {
            stopPlaying();
            updateProgress('Complete');
        }
        return;
    }
    showLayer(nextLayer);
}

function showLayer(layerIdx) {
    state.currentLayer = layerIdx;
    const layerData = VIZ_DATA.layers[layerIdx];
    const maxNorm = Math.max(...VIZ_DATA.layers.map(l => l.hidden_norm), 1);

    // Update architecture panel
    for (let i = 0; i < N_LAYERS; i++) {
        const block = document.getElementById(`arch-block-${i}`);
        block.classList.remove('active', 'completed');
        if (i < layerIdx) block.classList.add('completed');
        else if (i === layerIdx) block.classList.add('active');
    }

    // Fill attention grid row
    for (let hi = 0; hi < N_HEADS; hi++) {
        const cell = document.getElementById(`cell-${layerIdx}-${hi}`);
        cell.classList.remove('placeholder');
        cell.classList.add('filled');
        const canvas = cell.querySelector('canvas');
        drawMiniHeatmap(canvas, layerData.attn_weights[hi]);
    }

    // Extend activation bar
    const norm = layerData.hidden_norm;
    const pct = Math.min((norm / maxNorm) * 100, 100);
    const bar = document.getElementById(`flow-bar-${layerIdx}`);
    bar.style.width = pct + '%';
    bar.style.backgroundColor = normColor(norm, maxNorm);
    document.getElementById(`flow-val-${layerIdx}`).textContent = norm.toFixed(1);

    // Update selected cell detail if it's on this layer
    if (state.selectedCell && state.selectedCell.layer === layerIdx) {
        showDetail(layerIdx, state.selectedCell.head);
    }

    updateProgress(`Layer ${layerIdx}/${N_LAYERS - 1} — Block ${layerIdx}`);
}

function updateProgress(text) {
    document.getElementById('progress').textContent = text;
}

function togglePlay() {
    if (state.playing) {
        stopPlaying();
    } else {
        if (state.currentLayer >= N_LAYERS - 1) reset();
        state.playing = true;
        document.getElementById('btn-play').textContent = '⏸ Pause';
        state.timer = setInterval(tick, state.speed * 1000);
        tick(); // start immediately
    }
}

function stopPlaying() {
    state.playing = false;
    clearInterval(state.timer);
    state.timer = null;
    document.getElementById('btn-play').textContent = '▶ Play';
}

function stepNext() {
    stopPlaying();
    if (state.currentLayer < N_LAYERS - 1) {
        showLayer(state.currentLayer + 1);
    }
}

function stepPrev() {
    stopPlaying();
    if (state.currentLayer > 0) {
        // Hide current layer's grid row
        for (let hi = 0; hi < N_HEADS; hi++) {
            const cell = document.getElementById(`cell-${state.currentLayer}-${hi}`);
            cell.classList.remove('filled');
            cell.classList.add('placeholder');
            const canvas = cell.querySelector('canvas');
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        }
        // Reset activation bar
        const bar = document.getElementById(`flow-bar-${state.currentLayer}`);
        bar.style.width = '0%';
        document.getElementById(`flow-val-${state.currentLayer}`).textContent = '—';

        state.currentLayer--;

        // Update architecture highlighting
        for (let i = 0; i < N_LAYERS; i++) {
            const block = document.getElementById(`arch-block-${i}`);
            block.classList.remove('active', 'completed');
            if (i < state.currentLayer) block.classList.add('completed');
            else if (i === state.currentLayer) block.classList.add('active');
        }
        updateProgress(`Layer ${state.currentLayer}/${N_LAYERS - 1} — Block ${state.currentLayer}`);
    }
}

function reset() {
    stopPlaying();
    state.currentLayer = -1;

    // Reset architecture
    for (let i = 0; i < N_LAYERS; i++) {
        const block = document.getElementById(`arch-block-${i}`);
        block.classList.remove('active', 'completed');
    }

    // Reset grid
    for (let li = 0; li < N_LAYERS; li++) {
        for (let hi = 0; hi < N_HEADS; hi++) {
            const cell = document.getElementById(`cell-${li}-${hi}`);
            cell.classList.remove('filled', 'selected');
            cell.classList.add('placeholder');
            const canvas = cell.querySelector('canvas');
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    // Reset flow bars
    for (let i = 0; i < N_LAYERS; i++) {
        document.getElementById(`flow-bar-${i}`).style.width = '0%';
        document.getElementById(`flow-val-${i}`).textContent = '—';
    }

    // Hide detail
    document.getElementById('detail-panel').classList.remove('visible');
    state.selectedCell = null;

    updateProgress(`Ready — ${T} tokens, ${N_LAYERS} layers × ${N_HEADS} heads`);
}

// ── Click interaction ──
function onCellClick(layer, head) {
    // Only interact with filled cells
    const cell = document.getElementById(`cell-${layer}-${head}`);
    if (!cell.classList.contains('filled')) return;

    // Toggle selection
    if (state.selectedCell && state.selectedCell.layer === layer && state.selectedCell.head === head) {
        // Deselect
        cell.classList.remove('selected');
        document.getElementById('detail-panel').classList.remove('visible');
        state.selectedCell = null;
        // Remove architecture highlight
        for (let i = 0; i < N_LAYERS; i++) {
            const block = document.getElementById(`arch-block-${i}`);
            block.classList.remove('active');
            if (i <= state.currentLayer) {
                block.classList.add(i === state.currentLayer ? 'active' : 'completed');
            }
        }
        return;
    }

    // Clear previous selection
    document.querySelectorAll('.attn-cell.selected').forEach(c => c.classList.remove('selected'));

    // Select new cell
    cell.classList.add('selected');
    state.selectedCell = { layer, head };
    showDetail(layer, head);

    // Highlight corresponding architecture block
    for (let i = 0; i < N_LAYERS; i++) {
        const block = document.getElementById(`arch-block-${i}`);
        block.classList.remove('active', 'completed');
        if (i < layer) block.classList.add('completed');
        else if (i === layer) block.classList.add('active');
    }
}

function showDetail(layer, head) {
    const layerData = VIZ_DATA.layers[layer];
    const weights = layerData.attn_weights[head];
    const shapes = layerData.shapes;

    // Draw full-size heatmap
    const canvas = document.getElementById('detail-canvas');
    const size = Math.min(300, Math.max(200, T * 30));
    drawDetailHeatmap(canvas, weights, size);

    // Show info
    const info = document.getElementById('detail-info');
    info.innerHTML = `
        <div class="detail-title">Layer ${layer}, Head ${head} — attention weights (post-softmax + causal mask)</div>
        <div>Tokens: ${TOKENS.map(t => '<code>' + t + '</code>').join(' → ')}</div>
        <br>
        <div><strong>Tensor shapes at this layer:</strong></div>
        ${Object.entries(shapes).map(([k, v]) => `<div class="shape-line">${k}: ${v}</div>`).join('')}
    `;

    document.getElementById('detail-panel').classList.add('visible');
}

// ── Init ──
buildUI();
</script>
</body>
</html>
```

- [ ] **Step 3: Commit**

```bash
git add templates/visualize.html
git commit -m "feat: add HTML/JS animation template for transformer visualization"
```

---

### Task 3: Gradio tab integration in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the import at the top of app.py**

Add after the existing imports (around line 64):

```python
from visualize_anim import collect_viz_data
```

- [ ] **Step 2: Add the handler function**

Add after the existing `render_attention` handler (before the `build_ui` function). Find the section around line 560 (the Effects handlers) and add:

```python
# ═══════════════════════════════════════════════════════════════
# Tab 6: Visualize (animated forward pass)
# ═══════════════════════════════════════════════════════════════

_VIZ_TEMPLATE = None  # Lazy-loaded

def _get_viz_template() -> str:
    """Load the HTML template once."""
    global _VIZ_TEMPLATE
    if _VIZ_TEMPLATE is None:
        import os
        template_path = os.path.join(os.path.dirname(__file__), "templates", "visualize.html")
        with open(template_path, "r", encoding="utf-8") as f:
            _VIZ_TEMPLATE = f.read()
    return _VIZ_TEMPLATE


def render_visualization(prompt: str) -> str:
    """Collect tensors and return the animated HTML visualization."""
    import json
    model, tokenizer, config = _require_loaded()
    data = collect_viz_data(model, tokenizer, prompt, config)
    template = _get_viz_template()
    json_str = json.dumps(data)
    html = template.replace("{{VIZ_DATA_JSON}}", json_str)
    return html
```

- [ ] **Step 3: Add the Visualize tab in `build_ui()`**

Insert a new tab block between the Attention tab and the Generate tab. Find the line `# ── Tab 6: Generate` and insert before it:

```python
        # ── Tab 6: Visualize (animated forward pass) ──
        with gr.Tab("Visualize"):
            gr.Markdown(
                "### Animated forward pass — watch data flow through the transformer\n\n"
                "Enter a prompt and click **Visualize** to see a synchronized animation of:\n"
                "- **Left:** Architecture diagram — each layer lights up as data flows through\n"
                "- **Center:** 6×6 attention grid — every head's attention pattern, revealed layer by layer\n"
                "- **Right:** Activation flow — hidden state magnitude at each layer\n\n"
                "Click any cell in the attention grid to see the full-size heatmap and tensor shapes. "
                "Use the playback controls (Play/Pause/Step) to control the animation speed."
            )
            viz_prompt = gr.Textbox(
                label="Prompt", value="To be or not to be",
                lines=1, max_lines=2,
                info="Text to process. Shorter prompts (5-10 tokens) produce clearer visualizations.",
            )
            viz_btn = gr.Button("Visualize forward pass", variant="primary")
            viz_html = gr.HTML(label="Visualization")

            viz_btn.click(
                render_visualization,
                inputs=[viz_prompt],
                outputs=[viz_html],
            )
```

- [ ] **Step 4: Renumber subsequent tabs**

Update the comment for Generate from `# ── Tab 6:` to `# ── Tab 7:` and Benchmark from `# ── Tab 7:` to `# ── Tab 8:`.

- [ ] **Step 5: Update the header description**

Update the subtitle Markdown to include the Visualize tab:

```python
            "**Build Steps** → **Train** → **Effects** → **Train Reports** → "
            "**Attention** → **Visualize** → **Generate** → **Benchmark**."
```

- [ ] **Step 6: Verify import works**

Run: `NANOLLM_ALLOW_CPU=1 python -c "import app; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add Visualize tab — animated transformer forward pass"
```

---

### Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the app.py section**

In the `### app.py — Gradio UI reuses everything else` section, update the tab count from 7 to 8 and add the Visualize tab to the list:

Change:
```
`app.py` is the Gradio webinar console (7 tabs: Generate / Teach / Attention /
```
To:
```
`app.py` is the Gradio webinar console (8 tabs: Build Steps / Train / Effects /
Train Reports / Attention / Visualize / Generate / Benchmark).
```

Add a bullet for the new tab:
```
- **Visualize tab** uses `visualize_anim.py` to collect all attention weights
  and hidden norms in a single hooked forward pass, then renders an animated
  HTML/JS three-panel visualization via `gr.HTML`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Visualize tab (8 tabs)"
```

---

### Task 5: Full test run and manual verification

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `NANOLLM_ALLOW_CPU=1 PYTHONIOENCODING=utf-8 python -m pytest tests/ -v`
Expected: All tests pass (existing 52 + 3 new = 55)

- [ ] **Step 2: Manual UI test**

Run: `bash run.sh ui`

1. Navigate to the **Visualize** tab
2. Enter "The cat sat on" in the prompt
3. Click **Visualize forward pass**
4. Verify the three-panel layout appears
5. Click **▶ Play** — animation should sweep through all 6 layers
6. Click a cell in the attention grid — detail panel should expand below with full heatmap and shapes
7. Click the same cell again — detail panel should close
8. Toggle **Loop** on, click Play — animation should restart after completing
9. Use **◀ ▶** step buttons — should advance/retreat one layer

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete animated transformer visualization with tests"
```
