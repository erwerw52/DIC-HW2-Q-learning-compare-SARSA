# Analysis Report: Q-Learning vs SARSA in Cliff Walking

## 學習表現 (Learning Performance)
透過相同環境 (Cliff Walking 4x12) 與參數 (`α`=0.1, `γ`=0.9, `ε`=0.3) 訓練後，從 `result/reward_curves.png` 可觀察到差異：
- **收斂速度**：SARSA 與 Q-learning 皆在前期即獲得快速提升，且 SARSA 整體因為較早選擇避開危險，曲線通常會快速趨平。
- **收斂後獎勵**：SARSA 每回合的累積獎勵顯著高於 Q-Learning。因為 Q-Learning 在更新時總是忽略這 30% (`ε`=0.3) 的隨機探索並假設不會出錯，因此不時仍會墜崖；反觀 SARSA 學到的安全路徑，有效避免了高額 (-100) 懲罰的發生頻率。

## 策略行為 (Policy Behavior)
圖表 `result/policy_sb_style.png` 參考了經典的 Sutton & Barto (S&B) 圖例風格，展示了兩種算法最終學到的貪婪決策表與對應的虛線行走軌跡：
- **Q-Learning (Off-policy)**：傾向**冒險 (Risk-seeking)**。
  - 它直接找出全長最短的路徑（沿著懸崖上方 Row 2），從圖中可以看到虛線緊貼在淺藍色的 Cliff 區上方。這在零探索狀況下是最佳解，但在訓練時極易受隨機影響而摔下懸崖。
- **SARSA (On-policy)**：傾向**保守 (Risk-averse)**。
  - 圖中的策略不僅顯示出向上的安全箭頭，且其虛線路徑會繞行至距離懸崖最遠的頂部 (Row 0)。因為該演算法考量到了 $\epsilon$-greedy 所帶來的潛在危險風險，選擇了一條相對漫長但絕對安全的路徑。

## 總結
1. **追求理論最佳：** 如果最後部署的情境沒有隨機性 ($ε = 0$)，應選 **Q-Learning** 以取得極致最短路徑效能。
2. **追求訓練時的安全：** 若是在機器人系統或不容許犯大錯的實體環境中，推薦使用 **SARSA** 的保守策略，以避免承受極大的損失與風險。