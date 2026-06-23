# Resumable training for `train.py` and `train_sft.py`

## Problem

Both training scripts always construct a fresh model (and, for `train.py`, a
fresh optimizer/schedule) on every run. Re-running either script after a
crash, an interruption, or simply to extend training does not continue from
where it left off -- it starts over.

## Goals

- Running `train.py` or `train_sft.py` again after a previous run picks up
  training from the last saved checkpoint instead of starting from scratch.
- The learning-rate schedule and AdamW momentum continue smoothly across a
  resume (no warmup/momentum reset), for checkpoints written under the new
  format.
- `MAX_STEPS` is a fixed target step count, not "train N more steps". If the
  checkpoint already reached `MAX_STEPS`, the script reports this and exits
  without training further.
- Existing checkpoints (`stage5.pt`, `stage8.pt`, etc.), which only contain
  `model` + `config`, still load -- resuming from them falls back to a fresh
  optimizer and `step = 0` rather than erroring.

## Non-goals

- Not adding a "latest" checkpoint distinct from the "best" one. Checkpoints
  are still written only when val loss improves, same as today. Resuming
  may therefore redo up to `EVAL_INTERVAL` steps of training on different
  random batches -- accepted as harmless given training is already
  stochastic.
- Not changing `EVAL_INTERVAL`, batch sampling, or any other training
  hyperparameter.

## Checkpoint format

Both scripts' `torch.save` call gains two new keys:

```python
torch.save({
    "model": model.state_dict(),
    "config": cfg,
    "optimizer": optimizer.state_dict(),
    "step": step,
    "best_val": best_val,
}, ckpt_path)
```

`config` and `model` are unchanged in meaning. `optimizer` is
`optimizer.state_dict()` at the time of the save (matching the AdamW state
for the saved model weights). `step` is the training step just completed.
`best_val` is the val loss that triggered this save, persisted so resume
doesn't have to recompute it.

## Resume logic (applies to both scripts)

At startup, after the tokenizer/data are loaded and `cfg` is constructed
(or, for `train_sft.py`, after `BASE_CKPT_PATH` would normally be loaded):

1. **Checkpoint exists and has a `"step"` key** (new-format checkpoint):
   - Load `model` state dict (`strict=True` -- a config mismatch is a hard
     error, not silently ignored).
   - Build the optimizer, then load `optimizer.state_dict()` from the
     checkpoint.
   - `start_step = checkpoint["step"] + 1`.
   - `best_val = checkpoint["best_val"]`.
2. **Checkpoint exists but has no `"step"` key** (legacy checkpoint --
   today's `stage5.pt` / `stage8.pt`):
   - Load `model` state dict only.
   - Build a fresh optimizer (`step = 0` schedule state).
   - `start_step = 1`.
   - `best_val = estimate_loss(...)["val"]` computed once before the loop,
     so the next improving eval doesn't get spuriously treated as the first
     improvement over `inf`.
3. **Checkpoint does not exist**:
   - `train.py`: build a fresh model as it does today.
   - `train_sft.py`: build from `BASE_CKPT_PATH` as it does today.
   - `start_step = 1`, `best_val = inf`, fresh optimizer.

After resolving `start_step`:

- If `start_step > MAX_STEPS`: print
  `f"already trained to step {MAX_STEPS}, nothing to do"` and return without
  entering the training loop.
- Otherwise, the loop becomes `for step in range(start_step, MAX_STEPS + 1):`
  (currently `range(1, MAX_STEPS + 1)`). `lr_at(step)` needs no change --
  it's already a pure function of `(step, MAX_STEPS)`.

## `train_sft.py` specifics

`train_sft.py` has two distinct sources of initial weights: the SFT
checkpoint itself (`stage8.pt`, what we resume from) and the base
checkpoint (`stage5.pt`, what we fine-tune from if there's no SFT
checkpoint yet). The resume check targets `stage8.pt` only -- `stage5.pt` is
never treated as something to "resume" from, only as the fresh-start
source per case 3 above.

## Testing

- Manual: run `train.py` for a handful of steps (temporarily lower
  `MAX_STEPS`), interrupt, rerun, confirm it resumes near the interrupted
  step and the loss curve doesn't spike (no warmup/momentum reset).
- Manual: run with `MAX_STEPS` already satisfied by the checkpoint, confirm
  the "nothing to do" message and immediate exit.
- Manual: point at a legacy-format checkpoint (current `stage5.pt`),
  confirm it loads, falls back to step 0 / fresh optimizer, and trains
  without error.
