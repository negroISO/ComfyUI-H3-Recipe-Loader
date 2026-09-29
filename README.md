# ComfyUI H3 Recipe Loader

Pick a MiniMax H3 checkpoint and get that checkpoint's correct sampler settings
automatically — steps, sampler, scheduler, and sigma shift fill in live on the
node, stay hand-editable, and a turbo LoRA can't be stacked on a model that is
already distilled.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/negroiso/ComfyUI-H3-Recipe-Loader
```

Restart ComfyUI, then hard-refresh the browser (Ctrl+Shift+R) so the JS
extension loads. Nodes appear under **`MiniMax H3/loaders`**.

No dependencies beyond ComfyUI itself.

## Why

Step count is not a taste setting on H3. Running a **non-distilled** checkpoint
at 4–5 steps produces heavy chromatic speckle — and a turbo LoRA does **not**
convert a non-distilled checkpoint into a few-step model. That single wrong
assumption is what this node exists to prevent.

The three cases you have to keep straight:

| Checkpoint kind | Steps | Turbo LoRA |
|---|---|---|
| Non-distilled finetune | 20–25 | **no** |
| Turbo/distilled (baked in) | 4–8 | **no** — already distilled |
| Raw base model | 20–25, or 4–8 with a LoRA | **yes**, if you want low steps |

## Why a sidecar and not `<model>.metadata.json`

ComfyUI LoRA Manager's `metadata.json` does **not** carry these settings in any
machine-readable form. Checked across a set of real H3 checkpoints:

| Source | Has steps? | sampler? | shift? |
|---|---|---|---|
| `civitai/model/description` | prose HTML only | prose only | prose only |
| `civitai/images[].meta` | sometimes (often `None`) | always `None` | never |
| safetensors header | no | no | no |

Two checkpoints have zero sample images. DaSiWa V2 declares `steps`/`sampler`/
`cfgScale` keys whose values are all `None`. **Nothing anywhere records
`shift`**, which H3 needs most. Regex-scraping the prose would be wrong or empty
for most models, so the node reads a small file we control instead.

## The sidecar

Put `<model>.settings.json` next to the checkpoint in
`models/diffusion_models/`:

```json
{
  "schema": "h3-recipe/1",
  "distilled": false,
  "use_turbo_lora": true,
  "steps": 20,
  "sampler": "res_multistep",
  "scheduler": "simple",
  "shift_video": 12.0,
  "shift_audio": 3.0,
  "notes": "shown in the info output",
  "source": "where this recipe came from"
}
```

Every field is optional. Missing fields fall back to safe defaults
(non-distilled, 20 steps) and are **named in the info output**, so a
half-filled sidecar never fails silently.

No sidecar at all? The node guesses from the safetensors header and filename
(`turbo`, `distill`, `lightning`, `lcm`, `4step`, …), reports what it guessed
and why, and marks `CONFIDENCE: LOW`.

## Nodes

**H3 Recipe Loader (UNET + settings)** — replaces `UNETLoader`. Outputs:

| Output | Wire into |
|---|---|
| `model` | your model chain |
| `steps` | `BasicScheduler.steps` |
| `sampler` | `KSamplerSelect.sampler_name` |
| `scheduler` | `BasicScheduler.scheduler` |
| `shift_video` / `shift_audio` | `MiniMaxH3SigmaShift` |
| `use_turbo_lora` | a boolean switch, if you want one |
| `turbo_lora_strength` | **the turbo LoRA's strength** |
| `info` | a preview/show-text node |

`turbo_lora_strength` is the safety mechanism: it emits **0.0** whenever the
recipe says this checkpoint must not get a turbo LoRA, which makes a wired
`LoraLoader` inert. A distilled checkpoint therefore cannot accidentally get a
turbo LoRA applied, even if the node stays connected.

### Live widgets

`steps`, `sampler`, `scheduler`, `shift_video` and `shift_audio` are **real
editable widgets on the node**, not just outputs. Change the checkpoint and they
repopulate from that model's sidecar immediately, so you can see what changed
before queueing anything.

- **`auto_apply_recipe`** — `auto (follow model)` refills the widgets on every
  model change; `manual (keep my values)` leaves your edits alone.
- **`apply recipe now`** — button that pulls the recipe on demand regardless of
  the toggle, so you can hand-tune, experiment, then snap back.
- **Status line** under the widgets, colour-coded green (sidecar complete),
  amber (guessed / low confidence), red (lookup failed):

```
8 steps  res_multistep/simple  shift 12/7  |  turbo LoRA: NO (strength 0)
```

Widget values are the source of truth at execution time, with the sidecar as
fallback — so API and headless runs work without supplying them.

**H3 Recipe Info (read-only)** — same report without loading weights. Use it to
confirm a sidecar is being picked up.

## Known-good recipes

Starting points measured on an RTX 5090 with SageAttention 3. Copy
`examples/EXAMPLE.settings.json` next to your checkpoint, rename it to
`<model>.settings.json`, and edit.

| Checkpoint | Turbo LoRA | Settings |
|---|---|---|
| `DasiwaMinimaxH3_dasiwaHybridV2_int8` | **NO** (non-distilled) | `res_multistep`/simple, 20 steps, shift 12/3 |
| `10Eros_Max_h3_TURBO-hybrid_beta5_int8` | **NO** (turbo baked in) | `res_multistep`/simple, 8 steps, shift 12/7 |
| `minimax_h3_ref2va_pruned_int8_convrot` | **YES** (raw base) | `euler`/simple, 8 steps, shift 12/3 |
| `minimax_h3_fl2va_pruned_int8_convrot` | **YES** (raw base) | same; first/last-frame model, **unmeasured** |

Relative cost for the same 4 s clip at 0.6 MP: 10Eros 8-step ≈ 56 s, DaSiWa
20-step ≈ 96 s, raw base + turbo LoRA 8-step ≈ 206 s. The finetunes are both
faster and better than stacking a LoRA on the raw base.

Recipes are per-author, not universal — always check the model card. PRs adding
verified sidecars for other checkpoints are welcome.

## Contributing

Issues and PRs welcome, particularly:

- verified `settings.json` recipes for checkpoints not listed above
- additional `_DISTILLED_TOKENS` heuristics for the no-sidecar fallback
- wiring the resolver at `_read_sidecar()` to a richer metadata source

## Later: moving to metadata.json

If LoRA Manager gains these fields, or a metadata injector becomes available,
point `_read_sidecar()` in `h3_recipe_loader.py` at that file. Nothing else in
the node changes — the resolution, fallback, and reporting logic are all
downstream of that one function.
