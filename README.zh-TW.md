# DRL-on-parameter-extraction

[English](README.md)

這是一套以 PPO 為核心的訓練框架,專門處理「包著一個預測模型」的最佳化問題。內建兩個問題:一是 EEHEMT 電晶體模型的參數萃取,讓模擬 I-V 曲線去逼近量測資料;二是合金成分搜尋,找出預測硬度能過門檻的配方。框架本身不綁定任何問題,每個問題向註冊表登記,訓練器再依名稱查表取用。

底層使用 Ray RLlib 與 Tune,訓練紀錄走 Weights & Biases。

## 環境安裝

需要 Linux 加 Python 3.11(編譯 EEHEMT 模型的 `verilogae` 只支援這個組合)。相依套件由 `uv` 管理,定義在 `pyproject.toml`:

```bash
uv sync
```

硬度模型 ZIP 放 `src/env/hardness/`,輸入 CSV 放 `data/hardness/`,目錄裡的 `PUT_*_HERE.txt` 就是佔位提示。EEHEMT 量測資料放 `data/eehemt/`。

## 訓練

`train_ppo.py` 是唯一入口,用 `--env` 選問題:

```bash
uv run python train_ppo.py --env hardness
uv run python train_ppo.py --env eehemt
```

設定值來自 `.env`(啟動時經 python-dotenv 載入),命令列參數會蓋過環境變數。幾乎每個超參數都有環境變數預設值,所以先看 `.env`,不要假設程式碼裡的預設值就是實際生效的值。要接續之前的 Tune 訓練,帶 `--restore_path`。`scripts/train_ppo.sh` 包的也是同一條指令。

兩個問題各有自己的目標與 checkpoint 排名方式:

- `eehemt` 最小化線性電流空間的 NRMSE,checkpoint 依 `env_runners/min_nrmse` 取最低。
- `hardness` 追求預測硬度超過門檻,checkpoint 依 `env_runners/max_predicted_hardness` 取最高。committee 模型的不確定度只當診斷訊號記錄,不會進到 reward。

## 新增一個問題

目前的兩個問題都是外掛,只要問題長得像「找一個輸入,讓某個預測值夠好」就適用。寫好幾個小元件、取個名字註冊,`train_ppo.py --env <名稱>` 就能直接用,過程中不需要改動共用的訓練程式(`train_ppo.py`、`src/training/ppo_common.py`)。

需要的元件:

1. 預測後端:把一個候選解丟進去、吐出預測值的程式。如果你的模型和硬度模型一樣打包成 ZIP,現成的 `CommitteePackageBackend` 直接能用;其他模型就寫一個有 `predict` 和 `close` 方法的小類別,寫法看 `src/env/backends.py`。
2. 目標:把預測值換算成 reward、並判斷問題算不算解決的規則。`src/env/objectives.py` 裡有兩個現成的:「把某個值推過門檻」和「把誤差壓到門檻以下」,符合就能直接拿來用。都不合用的話,照同樣的形狀自己寫一個類別:`RANKED_METRIC` 與 `RANKED_ORDER` 兩個常數,加上算 reward 和判斷成功的方法。episode 何時結束是環境的事,不要寫進目標裡。
3. 環境:一個標準的 Gymnasium 環境,把後端和目標接起來。`src/env/material_hardness_env.py` 是可以照抄的範例。有兩個要求:觀測值在任何 reset 模式下都要落在有限邊界的 float32 Box 裡;後端要經由 env config 傳進來,測試才能換成虛擬物件。`MaterialHardnessEnv` 的 `prediction_backend_cls` 和 `EEHEMTEnv_Measure_VDS` 的 `simulator_factory` 是兩種現成寫法。
4. 訓練模組:宣告這個問題的命令列選項和 PPO 設定。可參考 `src/training/hardness_ppo.py`,大約 90 行。模組要提供 `add_env_args(parser, current_dir)`、`build_env_config(args)`、`build_ppo_config(args, *, num_learners, num_gpus_per_learner)`、`build_checkpoint_config()`,和一個 `<名稱>_WANDB_PROJECT` 常數。
5. 註冊:在 `src/problems/` 底下加一個小檔案,給問題取 `--env` 用的名字,寫法照 `problems/hardness.py`:組一個 `ProblemSpec`,呼叫 `problems.registry.register(spec)`。`checkpoint_metric` 和 `checkpoint_order` 直接從你的目標類別拿,指標名稱就只有一個出處。

`tests/test_toy_problem_extension.py` 就是用這幾個元件從零組出一個範例問題,是最小的完整範例。想把它升級成真實問題,照它的形狀換上你的後端和環境,把模組搬進 `src/problems/`,再從 `problems/__init__.py` 註冊。

完成前檢查一遍:

- 模型檔放 `src/env/<問題>/`、輸入資料放 `data/<問題>/`,兩個目錄都不進版控,記得補 `PUT_*_HERE.txt` 佔位檔。
- 超參數預設值照既有寫法:`add_env_args` 裡用 `os.getenv` 讀環境變數當預設值,並在 `.env` 裡留下說明。
- 測試把後端換成虛擬物件,不需要真的模型檔或 GPU,寫法參考 `tests/conftest.py` 和既有的環境測試。
- `uv run pytest`、`uv run ruff check .`、`uv run mypy .` 全部通過。

## 本地模型推論

硬度預測模型也能單獨使用,完全不碰強化學習。想直接拿預測結果時很方便:把一批候選成分寫成 CSV 丟進去,就能拿回每一筆的預測硬度。

需要一個訓練好的模型包,也就是一個 ZIP,內容長這樣:

```text
training_config.json
features.json
committee_scalers.pkl
committee_models/
```

### 命令列

最簡單的用法。指定模型 ZIP、輸入 CSV 和輸出位置:

```bash
uv run python scripts/run_model_inference.py \
  --model src/env/hardness/XGB_model_selection_package.zip \
  --input data/hardness/input.csv \
  --output data/hardness/output.csv
```

### Python API

在自己的腳本或 notebook 裡用:

```python
import pandas as pd

from env import InferenceModel

model = InferenceModel("src/env/hardness/XGB_model_selection_package.zip")

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

輸出有 `Predicted <target>` 和 `Uncertainty <target>` 兩個欄位。

也有一個直接吃 CSV 的捷徑函式:

```python
from env import predict

result_df = predict(
    model_package_path="src/env/hardness/XGB_model_selection_package.zip",
    input_data="data/hardness/input.csv",
)
```

### 類別型欄位

輸入欄位保持訓練資料原本的樣子就好。像 `Structure` 這種欄位,直接放 `"BCC"` 之類的值即可。如果當初訓練時對這類欄位做過編碼(one-hot 等),模型包自己知道,會套用同一套編碼,不用手動造出 `Structure_FCC` 這種欄位。

## 測試、lint 與型別檢查

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

一律透過 `uv run` 執行。全域安裝的 mypy 看不到專案 venv 裡的套件,會報出假錯誤。測試會自行 stub 掉推論模型,所以沒有模型 ZIP、沒有 GPU 也能跑完整套測試。
