# DRL-on-parameter-extraction

[English](README.md)

這是一套以 PPO 為核心的訓練框架,專門處理「包著一個預測模型」的最佳化問題。
內建兩個問題:一是 EEHEMT 電晶體模型的參數萃取,讓模擬 I-V 曲線去逼近量測
資料;二是合金成分搜尋,找出預測硬度能過門檻的配方。框架本身不綁定任何問
題,每個問題向註冊表登記,訓練器再依名稱查表取用。

底層使用 Ray RLlib 與 Tune,訓練紀錄走 Weights & Biases。

## 環境安裝

需要 Linux 加 Python 3.11(編譯 EEHEMT 模型的 `verilogae` 只支援這個組
合)。相依套件由 `uv` 管理,定義在 `pyproject.toml`:

```bash
uv sync
```

模型檔與量測資料不進版控。硬度模型 ZIP 放 `env/hardness/`,輸入 CSV 放
`data/hardness/`,目錄裡的 `PUT_*_HERE.txt` 就是佔位提示。EEHEMT 量測資料
放 `data/eehemt/`。

## 訓練

`train_ppo.py` 是唯一入口,用 `--env` 選問題:

```bash
uv run python train_ppo.py --env hardness
uv run python train_ppo.py --env eehemt
```

設定值來自 `.env`(啟動時經 python-dotenv 載入),命令列參數會蓋過環境變
數。幾乎每個超參數都有環境變數預設值,所以先看 `.env`,不要假設程式碼裡
的預設值就是實際生效的值。要接續之前的 Tune 訓練,帶 `--restore_path`。
`scripts/train_ppo.sh` 包的也是同一條指令。

兩個問題各有自己的目標與 checkpoint 排名方式:

- `eehemt` 最小化線性電流空間的 NRMSE,checkpoint 依
  `env_runners/min_nrmse` 取最低。
- `hardness` 追求預測硬度超過門檻,checkpoint 依
  `env_runners/max_predicted_hardness` 取最高。committee 模型的不確定度只
  當診斷訊號記錄,不會進到 reward。

## 測試、lint 與型別檢查

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

一律透過 `uv run` 執行。全域安裝的 mypy 看不到專案 venv 裡的套件,會報出
假錯誤。測試會自行 stub 掉推論模型,所以沒有模型 ZIP、沒有 GPU 也能跑完
整套測試。

## 新增一個問題

框架靠 `problems/registry.py` 解析 `--env <名稱>`,問題專屬的東西全部裝在
`ProblemSpec` 裡。新增問題不需要動任何共享程式碼。這句話有可執行的證明:
`tests/test_toy_problem_extension.py` 只靠測試程式碼就註冊了一個完整的玩
具問題,建議搭配本節一起讀。

一個問題由五個部分組成:

1. 預測後端,負責把一個候選解變成預測值。實作 `env/backends.py` 的
   `PredictionBackend` 協定:`predict(features) -> PredictionResult` 加上
   `close()`。committee 模型包 ZIP 直接沿用 `CommitteePackageBackend` 即
   可;類神經網路或其他代理模型直接實作協定,不必沿用 committee 的 ZIP 格
   式,也不必裝 verilogae 工具鏈。不確定度只作診斷用,別放進 reward。
2. 目標,負責把預測值變成 reward 與成功判定。語意相符時直接沿用
   `env/objectives.py` 的 `ThresholdMaximizeObjective` 或
   `NRMSEMinimizeObjective`;不符就照同樣的形狀新增一個類別,帶
   `RANKED_METRIC`、`RANKED_ORDER` 與 reward、成功判定方法。episode 的控
   制(termination、truncation)屬於環境,不屬於目標,原因記錄在 ADR
   0003。
3. 一個 `gymnasium.Env`,其觀測值在所有 reset 模式下都要落在有限邊界的
   float32 Box 裡。後端要能經由 env-config 注入,測試才能替換成假件。
   `MaterialHardnessEnv` 的 `prediction_backend_cls` 與
   `EEHEMTEnv_Measure_VDS` 的 `simulator_factory` 是兩個現成的示範。
4. 訓練模組,對外提供 `add_env_args(parser, current_dir)`、
   `build_env_config(args)`、`build_ppo_config(args, *, num_learners,
   num_gpus_per_learner)`(委派給
   `training.ppo_common.build_base_ppo_config`)、
   `build_checkpoint_config()`,以及一個 `<名稱>_WANDB_PROJECT` 常數。參考
   實作是 `training/hardness_ppo.py`,大約 90 行。
5. 註冊:組一個 `ProblemSpec`,呼叫 `problems.registry.register(spec)`,寫
   法照 `problems/hardness.py`。`checkpoint_metric` 與 `checkpoint_order`
   從你的目標類別拿,讓指標名稱只存在一個地方。

收尾前的確認清單:

- committee ZIP 放 `env/<問題>/`,輸入資料放 `data/<問題>/`,兩者都不進版
  控,記得補 `PUT_*_HERE.txt` 佔位檔。
- 超參數預設值照既有寫法:`add_env_args` 裡用 `os.getenv` 當 fallback,並
  在 `.env` 裡留下說明。
- 測試經由注入縫替換後端,不需要模型檔或 GPU,參考 `tests/conftest.py` 與
  既有的環境測試。
- `uv run pytest && uv run ruff check . && uv run mypy .` 全綠。
- 動到層邊界的新決策要記進 `docs/adr/` 與 `.codebase-memory/adr.md`。

想把玩具範例升級成真的問題:照抄 `tests/test_toy_problem_extension.py` 的
形狀,換上你的後端與環境,把模組搬進 `problems/`,再從
`problems/__init__.py` 註冊。

## 本地模型推論

`env/` 底下的硬度預測程式也能脫離 RL 迴圈單獨使用,適合本地腳本、批次推
論或其他最佳化流程載入平台訓練出的模型包。

ZIP 模型包必須包含:

```text
training_config.json
features.json
committee_scalers.pkl
committee_models/
```

### Python API

```python
import pandas as pd

from env import InferenceModel

model = InferenceModel("env/hardness/XGB_model_selection_package.zip")

input_df = pd.DataFrame(
    [
        {
            "Structure": "BCC",
            "frac_Al": 0.2,
            "frac_Cr": 0.1,
        }
    ]
)

result_df = model.predict(input_df)
print(result_df)
```

輸出包含 `Predicted <target>` 與 `Uncertainty <target>` 欄位。若想要型別化
的介面、不想依賴 DataFrame 欄位名,可以把模型包裝進
`env.backends.CommitteePackageBackend`,呼叫 `predict(features)` 拿
`PredictionResult`。

### 函式 API

```python
from env import predict

result_df = predict(
    model_package_path="env/hardness/XGB_model_selection_package.zip",
    input_data="data/hardness/input.csv",
)
```

### 命令列

```bash
uv run python scripts/run_model_inference.py \
  --model env/hardness/XGB_model_selection_package.zip \
  --input data/hardness/input.csv \
  --output data/hardness/output.csv
```

### 類別型特徵

輸入欄位保持與訓練資料相同的原始形式即可。如果當初訓練流程用了 one-hot、
label 或 target 編碼,模型包會從 `training_config.json` 讀到這件事並自動
套用,不需要自己手動造出 `Structure_FCC` 這類欄位。

## 文件地圖

- `CONTEXT.md` 定義專案的共同語彙(I-V Curve、Curve Condition、Feasible
  Material Composition、Problem Spec 等),程式與討論都用這套詞。
- `docs/adr/` 存架構決策紀錄:ADR 0001 鎖定 EEHEMT 的 NRMSE 目標,ADR
  0002 是 IR-drop 求解策略,ADR 0003 是問題註冊表、預測後端與目標抽象。
- `docs/how-to-add-a-problem.md` 是上面擴充指南的獨立版本。
- `docs/superpowers/specs/` 與 `docs/superpowers/plans/` 保存形成目前架構
  的設計文件與實作計畫。
