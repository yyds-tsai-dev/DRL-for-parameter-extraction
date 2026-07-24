# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Uses `uv` (Python 3.11; `verilogae` requires Linux + Python 3.11, hence `requires-python >=3.11,<3.13`).

```bash
uv sync                                    # install deps (pytest is in the dev group)
uv run pytest                              # run all tests
uv run pytest tests/test_material_hardness_env.py                 # one file
uv run pytest tests/test_ppo_common.py::test_name                 # one test
uv run ruff check .                        # lint (ruff/mypy are dev deps; global mypy false-alarms)
uv run mypy .                              # type check (config in mypy.ini; arg-type errors disabled)
uv run python train_ppo.py --env hardness  # train (or --env eehemt); scripts/train_ppo.sh wraps this
uv run python scripts/run_model_inference.py --model <zip> --input <csv> --output <csv>  # batch inference
```

Configuration comes from `.env` (loaded via python-dotenv at startup); CLI flags override env vars. Nearly every hyperparameter (LR, NRMSE_THRESHOLD, HARDNESS_THRESHOLD, OBSERVATION_FILTER, CHECKPOINT_DIR, WANDB_API_KEY, ...) has an env-var default, so check `.env` before assuming a code default is the active value. Pass `--restore_path` to resume a Tune run.

## Architecture

One PPO training harness (Ray RLlib + Tune, W&B logging) drives two independent RL problems, dispatched by `--env`:

1. **`eehemt`** — EEHEMT device-model parameter extraction: fit simulated I-V curves to measured data. The Verilog-A model in `src/env/eehemt/` is compiled via `verilogae`.
2. **`hardness`** — material composition optimization: find an alloy composition whose predicted hardness (from an XGBoost committee model package) meets a threshold.

`train_ppo.py` is the single entrypoint. `--env` resolves through the problem registry (`src/problems/registry.py`; built-ins self-register on `import problems`), which binds each problem's training module (`src/training/eehemt_ppo.py` / `src/training/hardness_ppo.py`: env-specific CLI args, callback, custom evaluation function, checkpoint ranking). Shared args/resources/W&B wiring and the generic PPO chain live in `src/training/ppo_common.py`. Adding a problem needs no shared-code edits — see `docs/how-to-add-a-problem.md`.

Layer boundaries (kept deliberately stable — see `docs/adr/` and `.codebase-memory/adr.md`):

- `src/env/parameter_flow.py` — parameter specs, measured-curve loading, EEHEMT simulation, IR-drop solving (multi-start continuation), arcsinh-Huber metric scaling.
- `src/env/eehemt_env.py` — Gymnasium episode state, action application, observation assembly, reward/termination for the EEHEMT problem.
- `src/env/material_hardness_env.py` — single-step env: action → bounded-simplex projection (`utils/composition_projection.py`) → hardness prediction → reward `(predicted - threshold) / scale`, clipped.
- `src/env/inference_engine.py` + `src/env/committee.py` — loads model-package ZIPs (`training_config.json`, `features.json`, `committee_scalers.pkl`, `committee_models/`); exposed as `env.InferenceModel` / `env.predict`. Handles categorical encoding alignment automatically.
- `evaluation/` — `metrics.py` (NRMSE) plus per-env custom RLlib evaluation functions that save checkpoint artifacts under `result/`; shared RLlib eval plumbing in `rllib_plumbing.py`.
- `utils/` — per-env RLlib callbacks, plotting, logging config.

Model artifacts and measurement data are **not committed**: the hardness model ZIP goes in `src/env/hardness/`, input CSVs in `data/hardness/` (see the `PUT_*_HERE.txt` placeholders).

## Model dispatch(模型調度規則)

主模型是 Fable 5(effortLevel: max),定位是**調度器**,不是執行者。原則:能委派就委派,把 Fable 的用量留給規劃、整合與關鍵決策。

| 角色 | 綁定 | 用途 |
|------|------|------|
| 主模型(Fable 5) | `.claude/settings.json` | 拆解任務、指派子 agent、整合結果、最終審查、跨模組決策 |
| `deep-reasoner` | Opus, effort high | 深度除錯與根因分析、數值/演算法問題(IR-drop 收斂、獎勵尺度)、RLlib 設定取捨、ADR 級架構決策 |
| `fast-worker` | Sonnet, effort low | 規格明確的機械執行:批次編輯、跑 pytest/ruff/mypy、套用重構、文件同步 |
| Codex(`codex` plugin) | GPT / codex CLI | 同級工程師的第二視角:獨立 code review(`review`、`adversarial-review` skill)、設計交叉檢查、與 deep-reasoner 意見分歧時的仲裁參考 |

調度規則:

- **主模型不做大量讀檔與機械編輯**——搜尋/摸索用 Explore agent,分析用 deep-reasoner,執行用 fast-worker。
- 派給 fast-worker 的指令必須含**精確規格與驗收條件**(例:改哪些檔、跑 `uv run pytest` 須全綠);它被指示遇到矛盾會停下回報,不自行猜測。
- 派給 deep-reasoner 的是**問題與相關檔案路徑**,要求回傳結論、依據(file:line)與風險;不派機械工作給它。
- 互不相依的子任務**同一訊息並行派發**;重要變更可讓 deep-reasoner 與 Codex **平行**給兩份獨立意見,再由主模型整合定案。
- 子 agent 的產出仍受本檔 guardrails 約束(NRMSE 目標、hardness 動作空間等);主模型整合時負責最後把關。

## Project conventions and guardrails

- `CONTEXT.md` defines the ubiquitous language (I-V Curve, Curve Condition, NRMSE Objective, Tunable/Fixed Composition Fraction, Feasible Material Composition, Hardness Objective). Use these terms in code, docs, and discussion.
- **EEHEMT objective is NRMSE in linear current space.** Reward is `clip(-log10((NRMSE/100) + EPSILON), REWARD_MIN, REWARD_MAX)`; termination is `NRMSE < NRMSE_THRESHOLD`; checkpoints rank by lowest `env_runners/min_nrmse` (episode-best NRMSE). Arcsinh Huber loss is a diagnostic only — do not make it the primary reward or termination criterion.
- **Hardness action space controls only the six tunable fractions** (Al, Cr, Mn, Fe, Co, Ni), each within [0.05, 0.35], summing to 1.0. Cu and Mo are fixed at zero — do not add them to the action space. Uncertainty is a diagnostic, not part of the reward. Checkpoints rank by `env_runners/max_predicted_hardness`.
- `OBSERVATION_FILTER=NoFilter` is the default; RLlib's `MeanStdFilter` can produce observations outside the declared Gymnasium Box under the new RLlib API stack — do not make it the default without a contract-preserving wrapper.
- Observations must remain legal `float32` Gymnasium arrays with finite bounds (`OBS_ERR_BOUND`) for all reset modes.
- Architecture decisions live in `docs/adr/` (and `.codebase-memory/adr.md`); feature plans/designs in `docs/superpowers/plans/` and `docs/superpowers/specs/`. Record new decisions of that kind there.
- `tests/conftest.py` inserts the project root and `src/` into `sys.path`; packages live under `src/` but import names stay `env.*`/`training.*`/`problems.*`/`evaluation.*` (keeps Ray checkpoint restore compatible). Tests stub the inference model, so they run without the model ZIP or GPU.
