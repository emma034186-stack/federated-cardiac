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
- **解法：** 放棄 Flower DP wrapper，改寫自訂 `_DPFedAvg` 繼承 `FedAvg`，在 `aggregate_fit()` 聚合後對每個參數元素加 Gaussian 噪聲（`noise_scale=0.005`），效果符合 DP 設計目的
- **結果：** DP 版本成功收斂，Best Dice = 0.6787，符合「精度略低但仍勝 single-site」的預期

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

## 實驗最終結果

| 實驗 | Best Dice | 備註 |
|------|-----------|------|
| Single-site baseline | 0.5805 | Hospital A only，50 epochs |
| FedAvg IID | **0.7849** | Round 10，20 rounds |
| FedAvg Non-IID | 0.7214 | Round 5，20 rounds |
| FedAvg Non-IID + DP | 0.6787 | Round 8，noise_scale=0.005 |
