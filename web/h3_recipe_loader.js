import { app } from "/scripts/app.js";

// Live-fill the H3 Recipe Loader's settings widgets when the checkpoint
// changes, so the values are visible AND hand-editable on the node itself.
//
// auto_apply_recipe ON  -> widgets are overwritten on every model change
// auto_apply_recipe OFF -> the user's values are left alone

const NODE_NAME = "H3RecipeLoader";
const SYNCED = ["steps", "sampler", "scheduler", "shift_video", "shift_audio"];

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setWidget(node, name, value) {
    const w = findWidget(node, name);
    if (!w || value === undefined || value === null) return false;
    if (w.value === value) return false;
    w.value = value;
    // Mirror into the DOM element combo/number widgets keep, when present.
    if (w.inputEl) w.inputEl.value = value;
    w.callback?.(value);
    return true;
}

// One badge line under the widgets: what the recipe decided, and how sure.
function setStatus(node, text, tone) {
    node.__h3Status = text;
    node.__h3Tone = tone || "info";
    node.setDirtyCanvas(true, true);
}

async function applyRecipe(node, { force = false } = {}) {
    const unetW = findWidget(node, "unet_name");
    if (!unetW?.value) return;

    const autoW = findWidget(node, "auto_apply_recipe");
    const auto = autoW ? autoW.value !== false : true;
    if (!auto && !force) {
        setStatus(node, "manual - widgets not touched", "manual");
        return;
    }

    let data;
    try {
        const res = await fetch(
            `/h3_recipe/lookup?unet_name=${encodeURIComponent(unetW.value)}`
        );
        data = await res.json();
    } catch (err) {
        setStatus(node, `lookup failed: ${err}`, "error");
        return;
    }
    if (!data || data.error) {
        setStatus(node, `lookup error: ${data?.error ?? "unknown"}`, "error");
        return;
    }

    for (const key of SYNCED) setWidget(node, key, data[key]);

    node.__h3Info = data.info || "";
    const turbo = data.use_turbo_lora
        ? "turbo LoRA: USE"
        : "turbo LoRA: NO (strength 0)";
    const conf = data.confident ? "" : "  |  LOW CONFIDENCE";
    setStatus(
        node,
        `${data.steps} steps  ${data.sampler}/${data.scheduler}  ` +
            `shift ${data.shift_video}/${data.shift_audio}  |  ${turbo}${conf}`,
        data.confident ? "ok" : "warn"
    );
}

app.registerExtension({
    name: "h3.recipe.loader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;

            // Refill whenever the checkpoint changes.
            const unetW = findWidget(node, "unet_name");
            if (unetW) {
                const prev = unetW.callback;
                unetW.callback = function () {
                    const out = prev?.apply(this, arguments);
                    applyRecipe(node);
                    return out;
                };
            }

            // Flipping auto back ON should immediately resync.
            const autoW = findWidget(node, "auto_apply_recipe");
            if (autoW) {
                const prev = autoW.callback;
                autoW.callback = function (v) {
                    const out = prev?.apply(this, arguments);
                    if (v !== false) applyRecipe(node, { force: true });
                    else setStatus(node, "manual - widgets not touched", "manual");
                    return out;
                };
            }

            // Manual "pull the recipe now", regardless of the auto toggle.
            node.addWidget("button", "apply recipe now", null, () =>
                applyRecipe(node, { force: true })
            );

            node.size[0] = Math.max(node.size[0], 420);
            // Widgets do not exist yet on the very first frame.
            setTimeout(() => applyRecipe(node), 0);
            return r;
        };

        // Restoring a saved graph: show status without clobbering saved values.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const node = this;
            const autoW = findWidget(node, "auto_apply_recipe");
            const auto = autoW ? autoW.value !== false : true;
            setTimeout(() => {
                if (auto) applyRecipe(node);
                else setStatus(node, "manual - widgets not touched", "manual");
            }, 0);
            return r;
        };

        // Paint the status line under the node.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed || !this.__h3Status) return;
            const tone = {
                ok: "#7ec87e",
                warn: "#e0c060",
                error: "#e07070",
                manual: "#9aa8c0",
                info: "#9aa8c0",
            }[this.__h3Tone] ?? "#9aa8c0";
            ctx.save();
            ctx.font = "11px monospace";
            ctx.fillStyle = tone;
            ctx.fillText(this.__h3Status, 12, this.size[1] - 6);
            ctx.restore();
        };
    },
});
