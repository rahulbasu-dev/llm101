# Animated Transformer Visualization — Design Spec

**Date:** 2026-04-23  
**Status:** Approved  
**Scope:** New "Visualize" tab in the Gradio UI (`app.py`)

## Overview

An animated, interactive visualization of the transformer forward pass that
combines three synchronized panels into a single view:

1. **Architecture diagram** — vertical pipeline showing data flowing through layers
2. **All-heads attention grid** — 6×6 mini-heatmaps (layers × heads)
3. **Activation flow** — per-layer hidden state norms as horizontal bars

The animation sweeps downward through the layers in sync across all three panels,
giving a complete picture of how the model processes a prompt.

## Delivery

New Gradio tab ("Visualize") in the existing `app.py` UI, rendered as embedded
HTML/CSS/JS via `gr.HTML`. Reuses the existing model singleton — no separate
loading.

### Tab position

After Attention, before Generate:

```
Build Steps | Train | Effects | Train Reports | Attention | Visualize | Generate | Benchmark
```

## Layout — Three Columns

All three panels share the same vertical axis (layer 0 at top, layer 5 at
bottom). The animation wave sweeps downward in sync.

### Left Panel: Architecture Diagram

- Vertical flow: `Embedding → Block 0 → Block 1 → ... → Block 5 → RMSNorm → LM Head`
- Each block expands to show `Attn Norm → Self-Attention → FFN Norm → FFN (SwiGLU)`
- **Current layer** glows amber during animation; completed layers stay blue;
  upcoming layers are grey
- Tensor shape annotations at each stage: `(B, T, 384)`, `(B, 6, T, 64)` for QKV, etc.

### Center Panel: All-Heads Attention Grid

- 6 rows (layers) × 6 columns (heads) = 36 mini-heatmaps
- Each cell is a small attention weight matrix (T×T), rendered as a colored
  thumbnail using a sequential blue colormap (white=0, dark blue=1, matching
  the Attention tab's existing style)
- Rows fill in top-to-bottom as the animation advances through layers
- Unfilled rows are dark/placeholder

### Right Panel: Activation Flow

- 6 horizontal bars, one per layer (L0–L5)
- Bar width = L2 norm of the hidden state after that layer
- Color coding: green (normal range) → amber (large) → red (very large)
- Bars extend with a smooth CSS transition as each layer completes
- Shows how the residual stream's magnitude evolves through the model

## Animation Controls

Rendered as a horizontal control bar between the prompt input and the
three-panel visualization:

- **Play / Pause** button (toggles)
- **Step ◀ ▶** — manually advance one layer at a time when paused
- **Speed slider** — 0.3s to 1.5s per layer (default: 0.8s)
- **Loop toggle** — when enabled, animation restarts from layer 0 after
  completing layer 5 (useful for demos/presentations)
- **Reset** button — returns to pre-animation state
- **Progress indicator** — text showing e.g. "Layer 3/6 — CausalSelfAttention"

## Click Interaction on the 6×6 Grid

Clicking any cell in the attention grid:

1. **Expands a detail panel** below the grid showing:
   - Full-size attention heatmap for that (layer, head) — same quality as the
     Attention tab's single-head view
   - Tensor shape annotations: Q, K, V dimensions for that head
   - Text label: "Layer 3, Head 2 — attention weights after softmax + causal mask"
2. **Highlights the corresponding layer** in the architecture panel (left) —
   that block gets a bright border/glow
3. Clicking another cell switches to that cell's detail; clicking the same cell
   again closes the detail panel

The detail panel is purely JS — no round-trip to Python. All 36 attention
matrices are already in the JSON payload.

## Data Flow

```
User types prompt → clicks "Visualize"
    │
    ▼
Python handler in app.py:
    1. Tokenize prompt with _TOKENIZER
    2. Run forward pass through _MODEL with ForwardCapture hooks
    3. Collect per-layer:
       - attention_weights[layer][head] → (T, T) float tensors
       - hidden_state_norms[layer] → scalar (L2 norm after residual)
       - tensor_shapes[layer] → dict of shape strings
    4. Serialize to JSON dict
    5. Inject JSON into the HTML template string
    6. Return HTML string to gr.HTML component
    │
    ▼
Browser renders HTML/CSS/JS:
    - Parses JSON data
    - Builds three-panel layout
    - Animation loop reads layer-by-layer from the data
    - Click handlers for grid interaction
```

## Files

### New files

| File | Purpose |
|------|---------|
| `visualize_anim.py` | `collect_viz_data(model, tokenizer, text, config)` → returns a dict with all tensors/shapes needed for the animation. Uses `ForwardCapture` hooks from `teach.py`. |
| `templates/visualize.html` | HTML/CSS/JS template with `{{VIZ_DATA_JSON}}` placeholder. Three-panel layout, animation engine, click interaction. Self-contained (no external deps). |

### Modified files

| File | Change |
|------|--------|
| `app.py` | New "Visualize" tab: prompt textbox + "Visualize" button → calls `collect_viz_data()` → injects into template → returns via `gr.HTML`. Tab positioned after Attention, before Generate. |
| `CLAUDE.md` | Update tab count and tab list. |

## JSON Data Structure

```json
{
  "tokens": ["The", "_cat", "_sat", "_on", "_the"],
  "n_layers": 6,
  "n_heads": 6,
  "d_model": 384,
  "d_head": 64,
  "layers": [
    {
      "layer_idx": 0,
      "attn_weights": [[0.1, 0.3, ...], ...],
      "hidden_norm": 12.5,
      "shapes": {
        "input": "(1, 5, 384)",
        "qkv": "(1, 5, 1152)",
        "q": "(1, 6, 5, 64)",
        "k": "(1, 6, 5, 64)",
        "v": "(1, 6, 5, 64)",
        "attn_out": "(1, 5, 384)",
        "ffn_out": "(1, 5, 384)",
        "output": "(1, 5, 384)"
      }
    }
  ]
}
```

`attn_weights` is a 3D array: `[head][query_pos][key_pos]`, values are
post-softmax, post-causal-mask floats in [0, 1].

## Animation Engine (JS)

- State: `{ currentLayer: 0, playing: false, speed: 0.8, loop: false }`
- `tick()`: called by `setInterval(tick, speed * 1000)` when playing
  - Highlights layer `currentLayer` in architecture panel
  - Fills row `currentLayer` of the attention grid with mini-heatmaps
  - Extends bar `currentLayer` in the activation panel
  - Increments `currentLayer`; if at end and loop=true, resets to 0
- Mini-heatmaps rendered as `<canvas>` elements (one per cell, T×T pixels
  scaled to ~50×50px). Canvas avoids DOM overhead for 36 small matrices.
- Full-size detail heatmap also rendered on `<canvas>` (T×T scaled to ~300×300px)
- CSS transitions on bar widths and opacity for smooth animation

## Constraints

- All data computed in one forward pass — no repeated model calls
- JSON payload size: ~36 matrices of T×T floats. For T=7 tokens, that's
  36 × 49 = 1,764 floats ≈ 14 KB. For T=50, ≈ 900 KB. Acceptable.
- No external JS libraries — vanilla JS + Canvas API only
- The HTML template must work inside Gradio's `gr.HTML` iframe/sandbox
- `ForwardCapture` from `teach.py` is reused — no new hooks in `model.py`

## Testing

- Unit test in `tests/test_integration.py`: call `collect_viz_data()` with the
  tiny test model and verify the JSON structure (correct number of layers,
  heads, attention matrix shapes)
- Manual test: run the UI, enter a prompt, verify animation plays through all
  6 layers with correct token labels

## Out of scope

- Export as GIF/video (future enhancement)
- 3D visualization
- Token-level activation inspection (beyond per-layer norms)
