## 1. Environment Implementation
- [ ] 1.1 實作 Cliff Walking 環境（網格 4x12，起點與終點設定，懸崖區罰扣 100 點並回到起點）。
- [ ] 1.2 設定行動空間（上下左右）與獎勵機制（每步 -1）。

## 2. Agent Implementation
- [ ] 2.1 實作 ε-greedy 策略函數。
- [ ] 2.2 實作 Q-Learning Agent 及狀態-動作價值函數更新。
- [ ] 2.3 實作 SARSA Agent 及狀態-動作價值函數更新。

## 3. Training Loop
- [ ] 3.1 實作主要訓練腳本，設定學習率 (0.1)、折扣因子 (0.9)、ε (0.1)。
- [ ] 3.2 執行至少 500 回合訓練並紀錄每個 Episode 的 Total Reward。

## 4. Evaluation & Visualization
- [ ] 4.1 建立 `result` 資料夾以儲存所有生成的圖表與報告。
- [ ] 4.2 繪製並比較 Q-Learning 與 SARSA 的累積獎勵曲線，並儲存至 `result/` 資料夾內。
- [ ] 4.3 視覺化兩種演算法最終學習到的路徑與策略行為，並將圖片出圖至 `result/` 內。
- [ ] 4.4 撰寫總結報告與理論比較結果（收斂速度、穩定性等），與結果圖表一併整理。