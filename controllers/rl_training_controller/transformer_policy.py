import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import math


class PositionalEncoding(nn.Module):
    """位置編碼模組"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """單個Transformer層"""
    
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=False  # 使用 [seq_len, batch, features] 格式
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [seq_len, batch_size, d_model]
            mask: [seq_len, seq_len] 注意力遮罩
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class SequenceEmbedding(nn.Module):
    """序列嵌入模組：將狀態-動作-獎勵序列轉換為統一的嵌入"""
    
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # 分別的嵌入層
        self.state_embed = nn.Linear(state_dim, hidden_size)
        self.action_embed = nn.Linear(action_dim, hidden_size)
        self.reward_embed = nn.Linear(1, hidden_size)  # 獎勵是標量
        
        # 類型嵌入（區分狀態、動作、獎勵）
        self.type_embed = nn.Embedding(3, hidden_size)  # 0:state, 1:action, 2:reward
        
        # 時間步嵌入
        self.timestep_embed = nn.Embedding(5000, hidden_size)  # 支援最多1000步
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, states, actions, rewards, timesteps=None):
        """
        Args:
            states: [seq_len, state_dim] 或 [batch_size, seq_len, state_dim]
            actions: [seq_len, action_dim] 或 [batch_size, seq_len, action_dim]
            rewards: [seq_len] 或 [batch_size, seq_len]
            timesteps: [seq_len] 或 [batch_size, seq_len] 可選的時間步
        
        Returns:
            embedded_sequence: [seq_len * 3, batch_size, hidden_size]
                按照 [s_0, a_0, r_0, s_1, a_1, r_1, ...] 的順序排列
        """
        # 自動處理單樣本和批次輸入
        if len(states.shape) == 2:  # 單樣本 [seq_len, state_dim]
            states = states.unsqueeze(0)    # [1, seq_len, state_dim]
            actions = actions.unsqueeze(0)  # [1, seq_len, action_dim]
            rewards = rewards.unsqueeze(0)  # [1, seq_len]
            single_sample = True
        else:
            single_sample = False
            
        batch_size, seq_len = states.shape[:2]
        
        # 嵌入各個模態
        state_embeds = self.state_embed(states)     # [batch, seq_len, hidden]
        action_embeds = self.action_embed(actions)  # [batch, seq_len, hidden]
        reward_embeds = self.reward_embed(rewards.unsqueeze(-1))  # [batch, seq_len, hidden]
        
        # 加入類型嵌入
        type_ids = torch.tensor([0, 1, 2], device=states.device)  # state, action, reward
        type_embeds = self.type_embed(type_ids)  # [3, hidden]
        
        state_embeds = state_embeds + type_embeds[0]
        action_embeds = action_embeds + type_embeds[1]
        reward_embeds = reward_embeds + type_embeds[2]
        
        # 加入時間步嵌入（如果提供）
        if timesteps is not None:
            time_embeds = self.timestep_embed(timesteps)  # [batch, seq_len, hidden]
            state_embeds = state_embeds + time_embeds
            action_embeds = action_embeds + time_embeds
            reward_embeds = reward_embeds + time_embeds
        
        # 交錯排列：[s_0, a_0, r_0, s_1, a_1, r_1, ...]
        # 轉換為 [seq_len, batch, hidden] 格式
        state_embeds = state_embeds.transpose(0, 1)   # [seq_len, batch, hidden]
        action_embeds = action_embeds.transpose(0, 1) # [seq_len, batch, hidden]
        reward_embeds = reward_embeds.transpose(0, 1) # [seq_len, batch, hidden]
        
        # 交錯合併
        sequence_embeds = []
        for t in range(seq_len):
            sequence_embeds.append(state_embeds[t])   # s_t
            sequence_embeds.append(action_embeds[t])  # a_t
            sequence_embeds.append(reward_embeds[t])  # r_t
        
        # 堆疊為 [seq_len * 3, batch, hidden]
        embedded_sequence = torch.stack(sequence_embeds, dim=0)
        
        return self.dropout(embedded_sequence)


class TransformerPolicyNetwork(nn.Module):
    """
    Transformer Policy Network for PPO
    參考Decision Transformer的序列建模方式
    """
    
    def __init__(
        self,
        state_dim=6,           # 狀態維度
        action_dim=6,          # 動作維度
        sequence_length=50,    # 序列長度
        hidden_size=128,       # 隱藏層維度
        n_layer=3,             # Transformer層數
        n_head=2,              # 注意力頭數
        dropout=0.1,           # Dropout率
        action_range=1.0       # 動作範圍 [-action_range, action_range]
    ):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎮 使用設備: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.action_range = action_range
        
        # 序列嵌入模組
        self.embedding = SequenceEmbedding(state_dim, action_dim, hidden_size)
        
        # 位置編碼
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=sequence_length * 3)
        
        # Transformer層
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # 層標準化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Policy頭（輸出動作的均值和標準差）
        self.policy_mean = nn.Linear(hidden_size, action_dim)
        self.policy_logstd = nn.Linear(hidden_size, action_dim)
        
        # Value頭（狀態價值估計）
        self.value_head = nn.Linear(hidden_size, 1)
        
        # 初始化權重
        self.apply(self._init_weights)
        self.to(self.device)
        
        print(f"✅ Transformer Policy Network 初始化完成")
        print(f"📊 參數: state_dim={state_dim}, action_dim={action_dim}")
        print(f"🔧 架構: seq_len={sequence_length}, hidden={hidden_size}, layers={n_layer}, heads={n_head}")
        print(f"📈 總參數量: {self.count_parameters():,}")
    
    def _init_weights(self, module):
        """初始化權重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def count_parameters(self):
        """計算模型參數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _ensure_device(self, tensor):
        """確保張量在正確設備上"""
        if isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, dtype=torch.float32).to(self.device)
        elif isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        else:
            return torch.tensor(tensor, dtype=torch.float32).to(self.device)
    
    def create_causal_mask(self, seq_len):
        """創建因果遮罩，確保當前位置只能看到過去的信息"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        return ~mask
    
    def forward(self, states, actions, rewards, return_dict=False):
        """
        前向傳播
        
        Args:
            states: [seq_len, state_dim] 或 [batch_size, seq_len, state_dim]
            actions: [seq_len, action_dim] 或 [batch_size, seq_len, action_dim]
            rewards: [seq_len] 或 [batch_size, seq_len]
            return_dict: 是否返回詳細字典
        
        Returns:
            如果return_dict=False: (action_mean, action_logstd, value)
            如果return_dict=True: 包含詳細信息的字典
        """
        states = self._ensure_device(states)
        actions = self._ensure_device(actions)
        rewards = self._ensure_device(rewards)
        # 自動處理單樣本輸入
        if len(states.shape) == 2:  # 單樣本 [seq_len, state_dim]
            states = states.unsqueeze(0)    # [1, seq_len, state_dim]
            actions = actions.unsqueeze(0)  # [1, seq_len, action_dim]
            rewards = rewards.unsqueeze(0)  # [1, seq_len]
            single_sample = True
        else:
            single_sample = False
            
        batch_size, seq_len = states.shape[:2]
        
        # 序列嵌入
        embedded_sequence = self.embedding(states, actions, rewards)
        # embedded_sequence: [seq_len * 3, batch_size, hidden_size]
        
        # 位置編碼
        embedded_sequence = self.pos_encoding(embedded_sequence)
        
        # 創建因果遮罩
        total_seq_len = embedded_sequence.shape[0]
        causal_mask = self.create_causal_mask(total_seq_len).to(embedded_sequence.device)
        
        # 通過Transformer層
        hidden_states = embedded_sequence
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, mask=causal_mask)
        
        # 層標準化
        hidden_states = self.layer_norm(hidden_states)
        
        # 提取狀態對應的隱藏狀態（序列中的狀態位置：0, 3, 6, 9, ...）
        state_indices = torch.arange(0, total_seq_len, 3)  # [0, 3, 6, 9, ...]
        state_hidden = hidden_states[state_indices]  # [seq_len, batch_size, hidden_size]
        
        # 轉換回 [batch_size, seq_len, hidden_size]
        state_hidden = state_hidden.transpose(0, 1)
        
        # 只使用最後一個狀態的隱藏狀態來預測動作和價值
        last_hidden = state_hidden[:, -1, :]  # [batch_size, hidden_size]
        
        # Policy輸出（動作的均值和對數標準差）
        action_mean = self.policy_mean(last_hidden)
        action_logstd = self.policy_logstd(last_hidden)
        
        # 限制動作範圍
        action_mean = torch.tanh(action_mean) * self.action_range
        action_logstd = torch.clamp(action_logstd, min=-20, max=2)
        
        # Value輸出
        value = self.value_head(last_hidden).squeeze(-1)  # [batch_size] 或 [1]
        
        # 如果是單樣本輸入，移除批次維度
        if single_sample:
            action_mean = action_mean.squeeze(0)    # [action_dim]
            action_logstd = action_logstd.squeeze(0) # [action_dim]
            value = value.squeeze(0)                # scalar
        
        if return_dict:
            return {
                'action_mean': action_mean,
                'action_logstd': action_logstd,
                'action_std': torch.exp(action_logstd),
                'value': value,
                'hidden_states': hidden_states,
                'state_hidden': state_hidden,
                'last_hidden': last_hidden
            }
        else:
            return action_mean, action_logstd, value
    
    def get_action_and_value(self, states, actions, rewards, action=None):
        """
        獲取動作和價值（用於PPO訓練）
        
        Args:
            states: [seq_len, state_dim] 或 [batch_size, seq_len, state_dim]
            actions: [seq_len, action_dim] 或 [batch_size, seq_len, action_dim]
            rewards: [seq_len] 或 [batch_size, seq_len]
            action: [action_dim] 或 [batch_size, action_dim] 可選，用於計算特定動作的概率
        
        Returns:
            sampled_action: [action_dim] 或 [batch_size, action_dim] 採樣的動作
            log_prob: scalar 或 [batch_size] 動作的對數概率
            entropy: scalar 或 [batch_size] 策略熵
            value: scalar 或 [batch_size] 狀態價值
        """
        states = self._ensure_device(states)
        actions = self._ensure_device(actions)
        rewards = self._ensure_device(rewards)
        if action is not None:
            action = self._ensure_device(action)
        action_mean, action_logstd, value = self.forward(states, actions, rewards)
        action_std = torch.exp(action_logstd)
        
        # 創建正態分佈
        dist = Normal(action_mean, action_std)
        
        # 如果沒有提供特定動作，則採樣
        if action is None:
            sampled_action = dist.sample()
        else:
            sampled_action = action
        
        # 計算對數概率
        log_prob = dist.log_prob(sampled_action).sum(dim=-1)  # 對動作維度求和
        
        # 計算熵
        entropy = dist.entropy().sum(dim=-1)
        
        # 限制動作範圍
        sampled_action = torch.clamp(sampled_action, -self.action_range, self.action_range)
        
        return sampled_action, log_prob, entropy, value
    
    def get_value(self, states, actions, rewards):
        """
        僅獲取狀態價值（用於優勢函數計算）
        
        Args:
            states: [seq_len, state_dim] 或 [batch_size, seq_len, state_dim]
            actions: [seq_len, action_dim] 或 [batch_size, seq_len, action_dim]
            rewards: [seq_len] 或 [batch_size, seq_len]
        
        Returns:
            value: scalar 或 [batch_size] 狀態價值
        """
        states = self._ensure_device(states)
        actions = self._ensure_device(actions)
        rewards = self._ensure_device(rewards)
        _, _, value = self.forward(states, actions, rewards)
        return value
    
    def get_action_distribution(self, states, actions, rewards):
        """
        獲取動作分佈（用於策略評估）
        
        Args:
            states: [seq_len, state_dim] 或 [batch_size, seq_len, state_dim]
            actions: [seq_len, action_dim] 或 [batch_size, seq_len, action_dim]
            rewards: [seq_len] 或 [batch_size, seq_len]
        
        Returns:
            dist: torch.distributions.Normal 動作分佈
        """
        states = self._ensure_device(states)
        actions = self._ensure_device(actions)
        rewards = self._ensure_device(rewards)
        action_mean, action_logstd, _ = self.forward(states, actions, rewards)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)


class TransformerPolicyWrapper:
    """
    Transformer Policy的包裝器，提供便利的介面用於與環境互動
    """
    
    def __init__(self, policy_network, device=None):
        if device is None:
            self.device = policy_network.device
        else:
            self.device = device
        self.policy = policy_network.to(device)
        
        # 用於推理時的序列緩存
        self.reset_sequence_cache()
    
    def reset_sequence_cache(self):
        """重置序列緩存"""
        seq_len = self.policy.sequence_length
        state_dim = self.policy.state_dim
        action_dim = self.policy.action_dim
        
        self.states_cache = torch.zeros(1, seq_len, state_dim).to(self.device)
        self.actions_cache = torch.zeros(1, seq_len, action_dim).to(self.device)
        self.rewards_cache = torch.zeros(1, seq_len).to(self.device)
        self.cache_index = 0
    
    def update_sequence(self, state, action=None, reward=0.0):
        """更新序列緩存"""
        if self.cache_index < self.policy.sequence_length:
            # 填充階段
            self.states_cache[0, self.cache_index] = torch.tensor(state).to(self.device)
            if action is not None:
                self.actions_cache[0, self.cache_index] = torch.tensor(action).to(self.device)
            self.rewards_cache[0, self.cache_index] = reward
            self.cache_index += 1
        else:
            # 滑動窗口階段
            self.states_cache = torch.roll(self.states_cache, -1, dims=1)
            self.actions_cache = torch.roll(self.actions_cache, -1, dims=1)
            self.rewards_cache = torch.roll(self.rewards_cache, -1, dims=1)
            
            self.states_cache[0, -1] = torch.tensor(state).to(self.device)
            if action is not None:
                self.actions_cache[0, -1] = torch.tensor(action).to(self.device)
            self.rewards_cache[0, -1] = reward
    
    def predict_action(self, state, deterministic=False):
        """
        預測動作（用於環境互動）
        
        Args:
            state: [state_dim] 當前狀態
            deterministic: 是否使用確定性動作
        
        Returns:
            action: [action_dim] 動作
        """
        # 更新狀態到序列中
        self.update_sequence(state)
        
        self.policy.eval()
        with torch.no_grad():
            if deterministic:
                action_mean, _, _ = self.policy(
                    self.states_cache.squeeze(0),   # [seq_len, state_dim]
                    self.actions_cache.squeeze(0),  # [seq_len, action_dim]
                    self.rewards_cache.squeeze(0)   # [seq_len]
                )
                action = action_mean.cpu().numpy()  # [action_dim]
            else:
                sampled_action, _, _, _ = self.policy.get_action_and_value(
                    self.states_cache.squeeze(0),   # [seq_len, state_dim]
                    self.actions_cache.squeeze(0),  # [seq_len, action_dim]
                    self.rewards_cache.squeeze(0)   # [seq_len]
                )
                action = sampled_action.cpu().numpy()  # [action_dim]
        
        # 更新動作到序列中
        self.update_sequence(state, action)
        
        return action
    
    def get_value(self, state):
        """獲取狀態價值"""
        self.policy.eval()
        with torch.no_grad():
            value = self.policy.get_value(
                self.states_cache.squeeze(0),   # [seq_len, state_dim]
                self.actions_cache.squeeze(0),  # [seq_len, action_dim]
                self.rewards_cache.squeeze(0)   # [seq_len]
            )
        return value.cpu().item()  # 返回標量
    
    def get_sequence_data(self):
        """獲取當前序列數據（供訓練器使用）"""
        return {
            'states': self.states_cache.squeeze(0),   # [seq_len, state_dim]
            'actions': self.actions_cache.squeeze(0), # [seq_len, action_dim]  
            'rewards': self.rewards_cache.squeeze(0)  # [seq_len]
        }


# 測試和使用範例
if __name__ == "__main__":
    # 創建測試數據（單樣本）
    sequence_length = 50
    state_dim = 6
    action_dim = 6
    
    # 單樣本測試數據
    states = torch.randn(sequence_length, state_dim)   # [50, 6]
    actions = torch.randn(sequence_length, action_dim) # [50, 6]
    rewards = torch.randn(sequence_length)             # [50]
    
    print("🧪 測試 Transformer Policy Network (單樣本模式)...")
    
    # 創建模型
    policy = TransformerPolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        sequence_length=sequence_length,
        hidden_size=128,
        n_layer=3,
        n_head=2,
        dropout=0.1,
        action_range=1.0
    )
    
    # 測試前向傳播
    print("\n📈 測試前向傳播...")
    action_mean, action_logstd, value = policy(states, actions, rewards)
    print(f"Action mean shape: {action_mean.shape}")     # [6]
    print(f"Action logstd shape: {action_logstd.shape}") # [6]
    print(f"Value shape: {value.shape}")                 # scalar
    
    # 測試動作採樣
    print("\n🎯 測試動作採樣...")
    sampled_action, log_prob, entropy, value = policy.get_action_and_value(states, actions, rewards)
    print(f"Sampled action shape: {sampled_action.shape}") # [6]
    print(f"Log prob shape: {log_prob.shape}")             # scalar
    print(f"Entropy shape: {entropy.shape}")               # scalar
    print(f"Value shape: {value.shape}")                   # scalar
    
    # 測試包裝器
    print("\n🔧 測試策略包裝器...")
    wrapper = TransformerPolicyWrapper(policy)
    
    test_state = np.random.randn(state_dim)
    action = wrapper.predict_action(test_state)
    value = wrapper.get_value(test_state)
    
    print(f"Predicted action: {action}")
    print(f"State value: {value}")
    
    print("\n✅ Transformer Policy Network 測試完成！")