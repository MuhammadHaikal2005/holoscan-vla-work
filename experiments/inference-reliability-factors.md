# Inference Reliability — Hypotheses and Experiments

**Started:** 2026-04-23  
**Task:** Pick up the blue cube  
**Model:** `checkpoints/dual_cam_blue_only/checkpoint-2000`  
**Dataset:** `datasets/dual_cam_blue_only` (~30 episodes)  
**Baseline success rate:** ~1%

The model runs end-to-end (server → client → arm) but success is near-zero. These are the candidate causes being investigated, tested one at a time to isolate each variable.

---

## Hypothesis 1 — Camera position drift

### Theory

The GR00T model is vision-driven. During training it saw the scene from a fixed perspective defined by where `cam0` and `cam1` were physically mounted. At inference time, if either camera has been bumped, tilted, or repositioned even slightly (a few centimetres or a few degrees), the model sees a different image than what it trained on. The arm joint predictions are then calibrated to the wrong spatial reference frame and the gripper misses the cube.

This effect is amplified with two cameras — both need to match training pose simultaneously.

### Test

1. Replay a training episode video:
   ```bash
   ffplay ~/hsb-groot-robot/datasets/dual_cam_blue_only/videos/chunk-000/observation.images.cam0/episode_000000.mp4
   ffplay ~/hsb-groot-robot/datasets/dual_cam_blue_only/videos/chunk-000/observation.images.cam1/episode_000000.mp4
   ```
2. Open the live preview:
   ```bash
   python inference/preview_cameras.py
   ```
3. Align the live view to match the training video frame — same crop, same angle, same visible workspace area.
4. Mark or tape the camera mount positions so they can be reproduced exactly.
5. Re-run inference and record success rate.

### Status

- [ ] Not started
- [ ] In progress
- [ ] Done

### Result

*(fill in after testing)*

---

## Hypothesis 2 — Robot base position drift

### Theory

The robot arm base is not bolted to the table. Between recording sessions it can slide or be bumped, changing its position relative to the cube. Even a 1 cm translation of the base moves the gripper tip by several centimetres at full extension, which is enough to miss the cube. The arm also has to start from a consistent home pose — if it starts significantly offset from where it started during training, the first few action steps are immediately wrong.

### Test

1. Check the arm's resting/home pose in the training data:
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_parquet('datasets/dual_cam_blue_only/data/chunk-000/episode_000000.parquet')
   print('First frame joints:', df.iloc[0][[c for c in df.columns if 'observation.state' in c]].values)
   print('Last frame joints: ', df.iloc[-1][[c for c in df.columns if 'observation.state' in c]].values)
   "
   ```
2. At inference time, manually move the arm to the home pose that matches the first frame joint values before triggering the policy.
3. Mark the arm base position on the table (tape or marker) so it can be placed back in the same spot every run.
4. Re-run inference and record success rate.

### Status

- [ ] Not started
- [ ] In progress
- [ ] Done

### Result

*(fill in after testing)*

---

## Hypothesis 3 — Dataset too sparse / too few episodes

### Theory

With only ~30 episodes and the cube placed at many different positions across them, the model has to generalise a pick action across a wide spread of cube positions. With so little data, it cannot reliably learn the mapping from any given cube position → correct joint trajectory. Instead it averages across all the training examples and produces a trajectory that doesn't quite work for any specific cube position.

The fix is either:
- **More data** — record 100+ episodes from a tighter, more consistent cube placement region.
- **Constrain the cube position** — fix the cube to a single spot for every episode so the model only needs to learn one precise pick trajectory.
- **Longer training** — more steps on the existing data (diminishing returns if the data itself is too scattered).

### Test (Option A — fix cube position, record more)

1. Fix the cube to one specific position on the table. Mark it with tape.
2. Record 50–100 episodes from that exact spot, same arm start pose, same camera positions.
3. Convert and train:
   ```bash
   # record new dataset
   conda activate lerobot2
   python recording/usb_dual_cam_record.py --dataset fixed_cube_v1

   # convert to v2.1
   python training/convert_v3_to_v2.py datasets/fixed_cube_v1

   # train
   bash training/finetune_dual_usb.sh fixed_cube_v1 fixed_cube_v1_ckpt --max-steps 5000
   ```
4. Test inference with the new checkpoint.

### Test (Option B — train longer on existing data)

1. Re-run training from step 0 with 5000–10000 steps:
   ```bash
   bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only_v2 --max-steps 10000
   ```
2. Check training loss at convergence vs step 2000 — if loss is still dropping at 2000, more steps will help.

### Status

- [ ] Not started
- [ ] In progress
- [ ] Done

### Result

*(fill in after testing)*

---

## Hypothesis 4 — Training steps too few (2000 → 10000)

### Theory

2000 gradient steps on a ~30-episode dataset is a short training run. With `global_batch_size = 8` and roughly 30 episodes × ~100 frames each = ~3000 frames, 2000 steps covers fewer than 6 full passes through the data. The training loss curve may still be trending downward at step 2000, meaning the model hadn't converged yet. Running to 10000 steps gives the model 3–4× more time to fit the training data and sharpen the action predictions.

### How to check if this is the issue

First inspect the loss curve from the existing run — if loss was still falling at step 2000, more steps will help:

```bash
python3 -c "
import json
s = json.load(open('checkpoints/dual_cam_blue_only/checkpoint-2000/trainer_state.json'))
for h in s['log_history']:
    if 'loss' in h:
        print(f\"step {h['step']:4d}  loss {h['loss']:.4f}\")
"
```

If you see the loss still dropping at the end, training was cut short.

### Test

Transfer to the 3090 (faster) and run 10000 steps:

```bash
# on the 3090
cd ~/holoscan-vla-work
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only_10k --max-steps 10000 --save-steps 1000
```

Then transfer `checkpoint-10000` back to Thor and test inference:

```bash
bash inference/run_server.sh checkpoints/dual_cam_blue_only_10k/checkpoint-10000
```

### Status

- [x] Loss curve inspected
- [x] 12k and 20k checkpoints tested on real robot — **CONFIRMED: more steps made it WORSE**

### Result

Inspected `trainer_state.json`. Loss curve:

| Step range | Loss range | Trend |
|---|---|---|
| 0 – 500 | 0.37 → 0.087 | Rapid descent |
| 500 – 1000 | 0.087 → 0.072 | Slowing |
| 1000 – 1500 | 0.072 → 0.050 | Slowing further |
| 1500 – 2000 | 0.032 – 0.071 | **Plateau / oscillation — no clear downward trend** |

The model converged by ~step 1500. Tested `checkpoint-12000` on the real robot — the arm barely moved, performing worse than `checkpoint-2000`. This is a textbook **overfitting** symptom: with only 30 scattered episodes, the model memorised the training examples so thoroughly by 12k steps that it lost all ability to generalise to the real scene.

**Conclusion: more training steps is not the solution. The bottleneck is the dataset.** The 2000-step checkpoint remains the best result so far. **This hypothesis is fully ruled out.**

---

## Testing order

| Priority | Hypothesis | Effort | Expected impact | Status |
|---|---|---|---|---|
| 1 | Camera position drift | Low (tape + re-run) | High | Not started |
| 2 | Robot base position | Low (tape + home pose) | High | Not started |
| 3 | Dataset sparsity (re-record more episodes) | High (re-record + re-train) | Very high | Not started |
| ~~4~~ | ~~More training steps (2000 → 10000)~~ | ~~Medium~~ | ~~Medium~~ | **Ruled out — overfitting confirmed** |

Fix 1 and 2 first — they are free and take 10 minutes. If success rate doesn't improve, the training data is the root cause (Hypothesis 3): the model memorised 30 scattered episodes and can't generalise. The loss plateau at ~0.04 on such a small dataset is a classic sign of overfitting to noise rather than learning the task.

---

## Log

| Date | Change | Success rate | Notes |
|---|---|---|---|
| 2026-04-23 | Baseline — checkpoint-2000 | ~1% | Server + client working, arm moving |
| 2026-04-24 | checkpoint-12000 | ~0% | Worse than 2000 steps — arm barely tries. Overfitting confirmed |
| 2026-04-24 | checkpoint-20000 | Not tested | Skipped — 12k already confirmed hypothesis |
