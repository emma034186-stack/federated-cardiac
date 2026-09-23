# Development Log — Federated Cardiac MRI Segmentation

紀錄開發過程中遇到的所有問題、根本原因與解法。

---

## Phase 1：資料讀取與前處理

### BUG-01｜MONAI RandHorizontalFlip 版本不相容
- **問題：** `AttributeError` 或 transform 執行失敗
- **原因：** 使用的 MONAI 版本已移除 `RandHorizontalFlip`，改為統一的 `RandFlip`
- **解法：** 將所有 `RandHorizontalFlip` 改為 `RandFlip(spatial_axis=1)`

---

### BUG-02｜Info.cfg 用 configparser 讀取失敗
- **問題：** `MissingSectionHeaderError`
- **原因：** ACDC 的 `Info.cfg` 沒有 `[section]` header，不符合 INI 格式規範
- **解法：** 改用手動 `open()` 逐行解析 `key: value`

---

### BUG-03｜MONAI transform 輸出 MetaTensor，PyTorch 無法直接使用
- **問題：** `TypeError` 或後續運算結果異常
- **原因：** 新版 MONAI transform pipeline 回傳 `MetaTensor` 而非一般 `np.ndarray`
- **解法：** transform 輸出後加 `np.array(x)` 強制轉換

---

### BUG-04｜PyTorch 對 non-writable NumPy array 的警告
- **問題：** `UserWarning: The given NumPy array is not writable`（出現在 `acdc_dataset.py:97`）
- **原因：** MONAI 或 NIfTI reader 回傳唯讀陣列，`torch.from_numpy()` 不接受
- **狀態：** 已 suppress warning，功能正常（`copy()` 的 overhead 可接受）

---

## Phase 2：聯邦學習框架整合

### BUG-05｜Client get/set parameters 不一致
- **問題：** 每輪聚合後模型效果沒有累積，Dice 不改善
- **原因：** `parameters()` 只回傳需要梯度的參數，缺少 Batch Normalization 的 running stats（`running_mean`, `running_var`）；server 聚合的是不完整的模型狀態
- **解法：** 改用 `state_dict()` 傳輸完整模型狀態；傳輸前轉 `float32`，載入後還原原始 dtype

---

### BUG-06｜MPS device mismatch 導致 RuntimeError
- **問題：** `Expected all tensors to be on the same device`
- **原因：** 從 server 接收的 parameters 預設在 CPU，但模型在 MPS（Apple Silicon GPU）
- **解法：** 載入 state_dict 時加 `.to(DEVICE)` 確保所有 tensor 在同一裝置

---

### BUG-07｜BN running stats 未交換導致 Global Dice = 0
- **問題：** 聯邦聚合後全局模型 Dice 接近 0，但各 client 本地 Dice 正常
- **原因：** 只聚合了可學習參數（weight/bias），Batch Normalization 的 running stats 沒有參與 FedAvg，導致 inference 時 normalization 完全錯誤
- **解法：** 使用完整 `state_dict()`（包含 BN running stats）進行聚合與廣播

---

## Phase 3：差分隱私（DP）整合

### BUG-08｜Flower DifferentialPrivacyServerSideFixedClipping 導致 Dice 卡在 ~0.025
- **問題：** 20 輪訓練後 Dice 全程維持在 0.025，完全不收斂
- **嘗試一：** `DP_CLIPPING_NORM = 1.0` → 無效
- **嘗試二：** `DP_CLIPPING_NORM = 5.0` → 仍無效
- **根本原因：** Flower 的 server-side DP wrapper 對**每個 client 傳來的完整模型參數向量**做 L2 norm clipping，而非對梯度更新量做 clipping。U-Net（~1.9M 參數）的參數向量 L2 norm 實測為 **~60**，設定 `clipping_norm=1.0` 等於把模型壓縮到原本的 1/60，幾乎歸零
- **解法：** 放棄 Flower DP wrapper，改寫自訂 `_DPFedAvg` 繼承 `FedAvg`，在 `aggregate_fit()` 聚合後對每個參數元素加 Gaussian 噪聲（`noise_scale=0.005`）
- **結果：** DP 版本成功收斂，Best Dice = 0.6787，符合「精度略低但仍勝 single-site」的預期
- **更正（2026-09-23）：** 此自訂版本只在聚合後加噪聲，**沒有**對各 client 的更新量做裁剪，也沒有隱私預算會計，因此**不構成形式化的差分隱私保證、沒有對應的 ε**（先前圖表上的「ε≈10」標示有誤，已移除）。另外 Best Dice 是以測試集挑選最佳輪次，會高估；最終輪 Dice 為 0.3422，訓練全程劇烈震盪，並非「成功收斂」。形式化 DP-FedAvg（per-client clipping + 校準噪聲 + RDP 會計）列為後續工作。

---

### BUG-09｜UnboundLocalError：ndarrays_to_parameters 在 if 區塊內 import
- **問題：** `UnboundLocalError: cannot access local variable 'ndarrays_to_parameters'`
- **原因：** 原本 `ndarrays_to_parameters` 在函數頂層使用，但 refactor 時不小心把 import 移進 `if use_dp:` 區塊內
- **解法：** 將 `from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays` 移到 module 頂層 import

---

## Phase 4：版本控制

### BUG-10｜git push rejected（fetch first）
- **問題：** `error: failed to push some refs — Updates were rejected`
- **原因：** GitHub 上有直接編輯（透過網頁），本地 commit 落後 remote
- **解法：** `git pull --rebase` 後再 `git push`

---

## Phase 5：資料管線與設定修正（2026-09-23）

### BUG-11｜ES 期影格寫死為 `_frame12`，80% 病人的 ES 切片沒被讀入
- **問題：** `load_patient_frames()` 固定讀 `_frame01`（ED）與 `_frame12`（ES），找不到檔案就靜默略過
- **原因：** ACDC 每位病人的 ED/ES 影格編號不同，記錄在 `Info.cfg` 的 `ED:`／`ES:` 欄位。統計 100 位訓練病人：只有 20 位 ES = 12，另有 1 位 ED ≠ 1 → 約 80 位病人只貢獻了 ED 期切片，資料量少了將近一半，而且 ES 期（心肌最厚、形狀差異最大）的樣本嚴重不足
- **解法：** 新增 `read_patient_info()` 解析整份 `Info.cfg`，依 `ED`／`ES` 欄位組出檔名

---

### BUG-12｜資料增強對影像與標註各自抽隨機參數，標註與影像錯位
- **問題：** `ACDCSliceDataset.__getitem__()` 對 `img` 和 `gt` 各呼叫一次 `aug_transform`
- **原因：** MONAI 的 `RandFlip`／`RandRotate`／`RandZoom` 每次呼叫都重新抽隨機參數 → 影像與標註拿到不同的翻轉、旋轉角度與縮放，訓練時標註經常對不上影像；另外標註用雙線性插值旋轉，類別值會被插成錯誤的中間類別
- **解法：** 改用字典版轉換 `RandFlipd`／`RandRotated`／`RandZoomd`（`keys=["img","gt"]`），同一次呼叫對兩者套用相同參數；標註改用 `nearest` 插值
- **影響：** 這很可能是先前 Dice 偏低（0.58～0.74，ACDC 常見水準 0.85 以上）的主因

---

### BUG-13｜`HOSPITAL_GROUPS` 與實際切分不一致
- **問題：** `config.py` 與 README 寫醫院 B 為 `HCM, DCM`，但實際上 B 沒有任何 DCM 病人
- **原因：** `build_noniid_splits()` 依序比對、配對到第一間醫院就 `break`，DCM 全部被分到醫院 A
- **解法：** 將醫院 B 的設定改為 `["HCM"]`，使設定、README 與實際行為一致（切分結果不變）

---

### 其他｜Windows + CUDA 環境下 client 無法使用 GPU
- **原因：** Ray 會對 `num_gpus=0` 的 actor 隱藏 GPU；原本在 Apple Silicon（MPS）上開發沒遇到
- **解法：** 有 CUDA 時每個 client 申請 `num_gpus=0.3`

---

### 其他｜Windows 上 Ray 共享記憶體失敗，新增 `--backend inprocess`
- **問題：** Windows（分頁檔 2 GB）執行 `start_simulation` 時 raylet 崩潰：`CreateFileMapping() failed. GetLastError() = 1455 / 1450`，主程式卡住不動
- **解法：** `simulate.py` 新增 `--backend inprocess`：同一程序內依序訓練三個 client，沿用相同的 `CardiacClient`、Flower `FedAvg.aggregate_fit` 與伺服器端評估；每輪重新建立 client（與 Flower 模擬相同，optimizer 狀態不跨輪）。預設仍為 `ray`
- 另外把標註切片改存為 `uint8`（原為 `int64`），降低記憶體用量，不影響計算結果

---

## 實驗最終結果（2026-09-23，修正 BUG-11～13 後重跑）

環境：RTX 4060 Laptop GPU，PyTorch 2.5.1 + CUDA 12.1，Flower 1.13.1；聯邦學習三組以 `--backend inprocess` 執行。
回報**最終輪**測試 Dice（RV／Myo／LV 平均），不以測試集挑最佳輪次。

| 實驗 | 最終輪 Dice | 修正前（最終輪） | 備註 |
|------|-------------|------------------|------|
| Single-site baseline | **0.8322** | 0.5805 | Hospital A only，50 epochs；與 Non-IID 使用相同測試病例 |
| FedAvg Non-IID | **0.9072** | 0.6799 | 20 rounds |
| FedAvg IID | **0.9188** | 0.7368 | 20 rounds；測試病例與其他組不同，僅供參考 |
| FedAvg Non-IID + noise (σ=0.005) | **0.0000** | 0.3422 | 20 輪全為 0，完全無法學習 |

- 聯邦學習（Non-IID）較單院基準高 0.075（相同測試集）
- 加噪聲版本在修正資料管線後完全失效。推測原因：噪聲加在**所有** state_dict 陣列上，包括 BatchNorm 的 running_mean／running_var，且沒有裁剪；變異數被擾動（甚至可能變負）會直接破壞推論。正確作法應只對可訓練參數的**更新量**裁剪加噪（DP-FedAvg），列為後續工作
- 修正前的數字僅保留作對照，不應再引用
