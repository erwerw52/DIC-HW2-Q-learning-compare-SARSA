## ADDED Requirements

### Requirement: Cliff Walking Environment
The system SHALL provide a Cliff Walking Gridworld environment for RL agents to interact with.

#### Scenario: Agent drops into the cliff
- **WHEN** 代理人走入位於最後一列但非終點的「懸崖」區域
- **THEN** 系統將給予 -100 的懲罰，並將代理人送回起點位置

#### Scenario: Agent reaches the goal
- **WHEN** 代理人抵達終點位置
- **THEN** 系統將結束此回合 (Episode ends)

### Requirement: Reinforcement Learning Agents
The system SHALL implement tabular Q-Learning and SARSA agents to learn in the environment.

#### Scenario: Q-learning update
- **WHEN** 代理人採取行動，進入新狀態並觀察到獎勵
- **THEN** 代理人透過 Off-policy 公式 (利用下一狀態最佳行動的值) 更新目前的 Q-value

#### Scenario: SARSA update  
- **WHEN** 代理人採取行動，進入新狀態並觀察到獎勵，接著再選出下一動作
- **THEN** 代理人透過 On-policy 公式 (利用實際選出的下一動作的 Q-value) 更新目前的 Q-value

### Requirement: Analytics and Plotting
The system SHALL record and plot the cumulative rewards and final learned paths for both agents.

#### Scenario: Plot cumulative rewards
- **WHEN** 訓練流程結束後
- **THEN** 系統應輸出每回合累積獎勵比較圖，清楚標示 Q-Learning 與 SARSA的表現差異