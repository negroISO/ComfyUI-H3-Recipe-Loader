"""H3 Recipe Loader — pick a checkpoint, get its correct sampler settings.

Why this exists
---------------
MiniMax H3 checkpoints disagree about step counts, and getting it wrong is not
a subtle quality loss: running a NON-distilled checkpoint at 4-5 steps produces
heavy chromatic speckle. A turbo LoRA does NOT convert a non-distilled model
into a few-step model.

The settings are published by each model's author, but NOT in machine-readable
form. `<model>.metadata.json` (written by ComfyUI LoRA Manager) carries the
recipe only as HTML prose inside `civitai/model/description`; there is no
steps/sampler field, sample-image `meta` is sparse or all-None, and nothing
anywhere records `shift`. So we read a small sidecar we control instead:

    models/diffusion_models/<model>.settings.json

Schema ("h3-recipe/1"):
    {
      "schema": "h3-recipe/1",
      "distilled": false,          # true => already few-step
      "use_turbo_lora": true,      # should a turbo LoRA be stacked?
      "steps": 20,
      "sampler": "res_multistep",
      "scheduler": "simple",
      "shift_video": 12.0,
      "shift_audio": 3.0,
      "notes": "free text shown in the info output",
      "source": "where the recipe came from"
    }

Every field is optional; missing ones fall back to a built-in default and are
reported in the info string so a half-filled sidecar never fails silently.

Migration path: when LoRA Manager (or a metadata injector) can carry these
fields, point `_read_sidecar` at that file instead — the rest of the node is
unchanged.
"""

from __future__ import annotations

import json
import logging
import os

import comfy.samplers
import comfy.sd
import folder_paths


# Emit the bare "COMBO" type, NOT an inline option list.
#
# ComfyUI matches link types by identity, not by contents: KSamplerSelect's
# sampler_name and BasicScheduler's scheduler are declared as the string
# "COMBO", so an output declared as a *list* of the same options is treated as
# a different type and the editor refuses the connection
# ("Connected nodes are using incompatible input and output types").
# The returned VALUE is still validated by the receiving node.
SAMPLER_T = "COMBO"
SCHEDULER_T = "COMBO"


def comfy_sampler_list():
    """Valid sampler names, for checking a sidecar's value is real."""
    return tuple(comfy.samplers.KSampler.SAMPLERS)


def comfy_scheduler_list():
    """Valid scheduler names, for checking a sidecar's value is real."""
    return tuple(comfy.samplers.KSampler.SCHEDULERS)


SCHEMA = "h3-recipe/1"
SIDECAR_SUFFIX = ".settings.json"

# Used when a sidecar is absent or a key is missing. Deliberately the SAFE
# choice: non-distilled many-step settings. Too many steps is slow; too few is
# broken output.
DEFAULTS = {
    "distilled": False,
    "use_turbo_lora": False,
    "steps": 20,
    "sampler": "res_multistep",
    "scheduler": "simple",
    "shift_video": 12.0,
    "shift_audio": 3.0,
    "notes": "",
    "source": "",
}

# Fallback heuristics for a checkpoint with no sidecar. Filename tokens and
# safetensors header keys that indicate a model is ALREADY distilled, and so
# must not get a turbo LoRA stacked on it.
_DISTILLED_TOKENS = ("turbo", "distill", "lightning", "lcm", "step4", "4step",
                     "8step", "step8")


def _detect_architecture(full_path: str):
    """Is this actually a MiniMax H3 checkpoint? Returns (verdict, why).

    verdict: "h3" | "other:<name>" | "unknown"

    Loading a non-H3 model into an H3 graph fails deep in the sampler with an
    opaque shape error (e.g. Krea2 wants a 4D image latent, H3 supplies a 5D
    video latent -> "not enough values to unpack (expected 4, got 3)"). Catch
    it at load time instead, where we can name the real problem.
    """
    try:
        from safetensors import safe_open
        with safe_open(full_path, framework="pt") as fh:
            header = fh.metadata() or {}
            arch = str(header.get("modelspec.architecture", "")).lower()
            if arch:
                if "minimax" in arch or "h3" in arch:
                    return "h3", "header architecture=%s" % arch
                return "other:%s" % arch, "header architecture=%s" % arch
            # No architecture tag: fall back to H3's distinctive tensors.
            keys = list(fh.keys())
    except Exception as exc:
        return "unknown", "could not read (%s)" % type(exc).__name__

    markers = sum(1 for k in keys
                  if "adaln_proj" in k or "token_refiner" in k)
    if markers:
        return "h3", "%d H3 marker tensors" % markers
    return "unknown", "no H3 marker tensors among %d keys" % len(keys)


def _sidecar_path(full_path: str) -> str:
    """`.../foo.safetensors` -> `.../foo.settings.json`"""
    base, _ = os.path.splitext(full_path)
    return base + SIDECAR_SUFFIX


def _read_sidecar(full_path: str):
    """Return (recipe_dict, status_string). Never raises."""
    path = _sidecar_path(full_path)
    if not os.path.isfile(path):
        return None, "no sidecar at %s" % os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:                        # malformed JSON, bad encoding
        return None, "sidecar unreadable (%s: %s)" % (type(exc).__name__, exc)
    if not isinstance(data, dict):
        return None, "sidecar is not a JSON object"
    schema = data.get("schema")
    if schema and schema != SCHEMA:
        return data, "sidecar schema %r != %r (reading anyway)" % (schema, SCHEMA)
    return data, "ok"


def _guess_from_model(full_path: str):
    """Heuristic distilled/not for a checkpoint with no sidecar.

    Checks the safetensors header first (authors sometimes tag it), then falls
    back to filename tokens. Returns (distilled, why).
    """
    name = os.path.basename(full_path).lower()
    try:
        from safetensors import safe_open
        with safe_open(full_path, framework="pt") as fh:
            header = fh.metadata() or {}
        blob = " ".join("%s=%s" % (k, v) for k, v in header.items()
                        if k != "_quantization_metadata").lower()
        for token in _DISTILLED_TOKENS:
            if token in blob:
                return True, "header mentions %r" % token
    except Exception:
        pass                                        # no header / not safetensors
    for token in _DISTILLED_TOKENS:
        if token in name:
            return True, "filename contains %r" % token
    return False, "no distilled marker in header or filename"


def _resolve(unet_name: str):
    """Load recipe for a checkpoint. Returns (recipe, info_lines, confident)."""
    full = folder_paths.get_full_path("diffusion_models", unet_name)
    if full is None:
        full = folder_paths.get_full_path("unet", unet_name) or unet_name

    recipe = dict(DEFAULTS)
    data, status = _read_sidecar(full)
    lines = ["model   : %s" % unet_name]

    arch, arch_why = _detect_architecture(full)
    wrong_arch = arch.startswith("other:")
    if wrong_arch:
        lines.append("  *** NOT A MINIMAX H3 CHECKPOINT (%s) ***" % arch_why)
        lines.append("  *** an H3 graph will fail in the sampler with a "
                     "shape error ***")
    elif arch == "unknown":
        lines.append("  (architecture unverified: %s)" % arch_why)

    if data:
        used, ignored = [], []
        for key in DEFAULTS:
            if key in data and data[key] is not None:
                recipe[key] = data[key]
                used.append(key)
        for key in data:
            if key not in DEFAULTS and key != "schema":
                ignored.append(key)
        lines.append("sidecar : %s  (%d field(s) applied)" % (status, len(used)))
        missing = [k for k in ("steps", "sampler", "shift_video")
                   if k not in used]
        if missing:
            lines.append("  !! missing %s -> using defaults" % ", ".join(missing))
        if ignored:
            lines.append("  (ignored unknown keys: %s)" % ", ".join(sorted(ignored)))
        # A typo'd sampler/scheduler would otherwise surface as an obscure
        # failure inside the sampler. Catch it here and fall back.
        if recipe["sampler"] not in comfy_sampler_list():
            lines.append("  !! sampler %r is not a known sampler -> using %r"
                         % (recipe["sampler"], DEFAULTS["sampler"]))
            recipe["sampler"] = DEFAULTS["sampler"]
            missing.append("sampler")
        if recipe["scheduler"] not in comfy_scheduler_list():
            lines.append("  !! scheduler %r is not a known scheduler -> using %r"
                         % (recipe["scheduler"], DEFAULTS["scheduler"]))
            recipe["scheduler"] = DEFAULTS["scheduler"]
            missing.append("scheduler")
        confident = not missing
    else:
        distilled, why = _guess_from_model(full)
        recipe["distilled"] = distilled
        if distilled:
            # Already few-step: do NOT stack a turbo LoRA, and use a low count.
            recipe["use_turbo_lora"] = False
            recipe["steps"] = 8
            recipe["sampler"] = "euler"
        else:
            recipe["use_turbo_lora"] = False
            recipe["steps"] = 20
            recipe["sampler"] = "res_multistep"
        lines.append("sidecar : %s" % status)
        lines.append("  GUESSED distilled=%s (%s)" % (distilled, why))
        lines.append("  -> write %s to make this exact"
                     % os.path.basename(_sidecar_path(full)))
        confident = False

    # Carried on the dict rather than the signature so existing callers and
    # the HTTP route keep working unchanged.
    recipe["_architecture"] = arch
    recipe["_wrong_architecture"] = wrong_arch
    if wrong_arch:
        confident = False
    return recipe, lines, confident


class H3RecipeLoader:
    """UNETLoader + per-checkpoint settings read from a sidecar JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),
                              {"tooltip": "Checkpoint. Its settings are read "
                                          "from <name>.settings.json beside it."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast",
                                  "fp8_e5m2"],),
            },
            "optional": {
                # auto_apply_recipe ON: the JS extension overwrites the four
                # widgets below whenever unet_name changes, so they always show
                # the live recipe. Turn it OFF to hand-tune without the next
                # model switch clobbering your values.
                "auto_apply_recipe": ("BOOLEAN", {
                    "default": True, "label_on": "auto (follow model)",
                    "label_off": "manual (keep my values)",
                    "tooltip": "ON: widgets refill from the sidecar on every "
                               "model change. OFF: your edits are kept."}),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 200,
                    "tooltip": "Auto-filled from the recipe. Edit freely with "
                               "auto_apply_recipe off."}),
                "sampler": (comfy_sampler_list(), {"default": "res_multistep"}),
                "scheduler": (comfy_scheduler_list(), {"default": "simple"}),
                "shift_video": ("FLOAT", {
                    "default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {
                    "default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "turbo_lora_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Strength emitted when the recipe says a turbo "
                               "LoRA SHOULD be used. Emits 0.0 when it says "
                               "not to, which disables a LoraLoader inline."}),
                # ref2va V2V (reference-driven video-to-video identity swap) is
                # broken by turbo LoRAs: the swap degrades into a slideshow of
                # the reference, cycling between source and reference, or no
                # transfer at all. Turning the turbo LoRA off fixes it. The
                # sidecar's use_turbo_lora is about STEP COUNT, so it cannot
                # know which task you are running — hence this switch.
                "task": (["t2v / i2v (default)", "ref2va v2v (no turbo LoRA)"], {
                    "default": "t2v / i2v (default)",
                    "tooltip": "Set to ref2va v2v when doing a reference-driven "
                               "video-to-video identity swap. Forces "
                               "turbo_lora_strength to 0.0 regardless of the "
                               "sidecar, because turbo LoRAs break that path."}),
            },
        }

    RETURN_TYPES = ("MODEL", "INT", SAMPLER_T, SCHEDULER_T,
                    "FLOAT", "FLOAT", "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("model", "steps", "sampler", "scheduler", "shift_video",
                    "shift_audio", "use_turbo_lora", "turbo_lora_strength",
                    "info")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/loaders"
    DESCRIPTION = ("Load an H3 checkpoint and emit the author's recommended "
                   "sampler settings from a <model>.settings.json sidecar.")

    def load(self, unet_name, weight_dtype, auto_apply_recipe=True,
             steps=None, sampler=None, scheduler=None, shift_video=None,
             shift_audio=None, turbo_lora_strength=1.0,
             task="t2v / i2v (default)"):
        import torch

        recipe, lines, confident = _resolve(unet_name)

        # The widgets are the source of truth at execution time: the JS keeps
        # them synced to the recipe while auto_apply_recipe is on, and they hold
        # the user's edits when it is off. Falling back to the recipe keeps the
        # node working headlessly (API calls with no widget values supplied).
        overrides = {"steps": steps, "sampler": sampler, "scheduler": scheduler,
                     "shift_video": shift_video, "shift_audio": shift_audio}
        changed = []
        for key, value in overrides.items():
            if value is None:
                continue
            if value != recipe[key]:
                changed.append("%s %s->%s" % (key, recipe[key], value))
            recipe[key] = value
        if changed:
            if auto_apply_recipe:
                # Widgets disagree with the sidecar while claiming to follow it.
                # Usually means the sidecar changed after the node was added.
                lines.append("  widgets differ from sidecar: %s"
                             % ", ".join(changed))
                lines.append("  (auto is ON - reselect the model to resync)")
            else:
                lines.append("  MANUAL overrides: %s" % ", ".join(changed))

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        full = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(full, model_options=model_options)

        steps = int(recipe["steps"])
        use_turbo = bool(recipe["use_turbo_lora"])
        is_v2v = task.startswith("ref2va v2v")
        if is_v2v and use_turbo:
            lines.append("  ref2va v2v selected -> turbo LoRA FORCED OFF "
                         "(sidecar said use it; turbo breaks v2v swaps)")
            use_turbo = False
        # Emitting 0.0 lets a LoraLoader stay wired but inert, so a distilled
        # checkpoint cannot accidentally get a turbo LoRA applied.
        out_strength = float(turbo_lora_strength) if use_turbo else 0.0

        lines.append("applied : %d steps, %s/%s, shift %s/%s"
                     % (steps, recipe["sampler"], recipe["scheduler"],
                        recipe["shift_video"], recipe["shift_audio"]))
        lines.append("turbo   : %s -> strength %s%s"
                     % ("USE a turbo LoRA" if use_turbo else "NO turbo LoRA",
                        out_strength, "" if use_turbo else "  (0.0 = inert)"))
        if recipe.get("distilled"):
            lines.append("          checkpoint is DISTILLED (few-step native)")
        if recipe.get("notes"):
            lines.append("notes   : %s" % recipe["notes"])
        if recipe.get("source"):
            lines.append("source  : %s" % recipe["source"])
        if recipe.get("_wrong_architecture"):
            # Loud, because the downstream failure is an opaque shape error
            # far from the real cause.
            logging.warning(
                "[H3 Recipe Loader] %s is a %r model, not MiniMax H3. An H3 "
                "graph will fail in the sampler (latent rank mismatch). Pick "
                "an H3 checkpoint.",
                unet_name, recipe.get("_architecture", "?").replace("other:", ""))
        elif not confident:
            lines.append("CONFIDENCE: LOW - verify against the model card.")

        return (model, steps, recipe["sampler"], recipe["scheduler"],
                float(recipe["shift_video"]), float(recipe["shift_audio"]),
                use_turbo, out_strength, "\n".join(lines))


class H3RecipeInfo:
    """Read-only: report a checkpoint's recipe without loading the weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
        }}

    RETURN_TYPES = ("INT", SAMPLER_T, SCHEDULER_T,
                    "FLOAT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("steps", "sampler", "scheduler", "shift_video",
                    "shift_audio", "use_turbo_lora", "info")
    FUNCTION = "peek"
    CATEGORY = "MiniMax H3/loaders"
    OUTPUT_NODE = True
    DESCRIPTION = ("Report the recipe for a checkpoint without loading it. "
                   "Useful for checking a sidecar is being picked up.")

    def peek(self, unet_name):
        recipe, lines, confident = _resolve(unet_name)
        if not confident:
            lines.append("CONFIDENCE: LOW - verify against the model card.")
        info = "\n".join(lines)
        return {
            "ui": {"text": [info]},
            "result": (int(recipe["steps"]), recipe["sampler"],
                       recipe["scheduler"], float(recipe["shift_video"]),
                       float(recipe["shift_audio"]),
                       bool(recipe["use_turbo_lora"]), info),
        }


# ---------------------------------------------------------------- HTTP route
# The JS extension calls this when unet_name changes so it can populate the
# node's widgets live. Registered defensively: if the server API ever moves,
# the nodes still work, they just lose the auto-fill.
try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web

    @PromptServer.instance.routes.get("/h3_recipe/lookup")
    async def _h3_recipe_lookup(request):
        name = request.query.get("unet_name", "")
        if not name:
            return _aiohttp_web.json_response(
                {"error": "missing unet_name"}, status=400)
        try:
            recipe, lines, confident = _resolve(name)
        except Exception as exc:                    # never 500 the editor
            return _aiohttp_web.json_response(
                {"error": "%s: %s" % (type(exc).__name__, exc)}, status=200)
        return _aiohttp_web.json_response({
            "steps": int(recipe["steps"]),
            "sampler": recipe["sampler"],
            "scheduler": recipe["scheduler"],
            "shift_video": float(recipe["shift_video"]),
            "shift_audio": float(recipe["shift_audio"]),
            "use_turbo_lora": bool(recipe["use_turbo_lora"]),
            "distilled": bool(recipe.get("distilled", False)),
            "wrong_architecture": bool(recipe.get("_wrong_architecture")),
            "architecture": recipe.get("_architecture", "unknown"),
            "confident": bool(confident),
            "info": "\n".join(lines),
        })
except Exception:                                   # pragma: no cover
    pass


NODE_CLASS_MAPPINGS = {
    "H3RecipeLoader": H3RecipeLoader,
    "H3RecipeInfo": H3RecipeInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RecipeLoader": "H3 Recipe Loader (UNET + settings)",
    "H3RecipeInfo": "H3 Recipe Info (read-only)",
}
