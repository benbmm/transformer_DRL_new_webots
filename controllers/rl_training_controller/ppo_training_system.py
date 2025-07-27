import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from collections import deque
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import json
import pickle

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Weights & Biases 未安裝，僅使用 TensorBoard")


@dataclass
class PPOConfig:
    """PPO訓練配置"""
    # 環境配置
    max_episode_steps: int = 2000
    sequence_length: int = 50
    
    # 網路配置
    hidden_size: int = 128
    n_layer: int = 3
    n_head: int = 2
    dropout: float = 0.1
    action_range: float = 1.0
    
    # PPO配置
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # 訓練配置
    total_timesteps: int = 1000000
    episodes_per_update: int = 4    # 每次更新收集的episode數
    update_epochs: int = 4          # 每次數據的更新epochs
    gae_lambda: float = 0.95
    gamma: float = 0.99
    
    # 學習率調度
    anneal_lr: bool = True
    
    # 早停和保存
    target_reward: float = 0.8      # 目標平均獎勵
    save_frequency: int = 50        # 每50次更新保存一次
    eval_frequency: int = 20        # 每20次更新評估一次
    
    # 日誌配置
    log_frequency: int = 10         # 每10次更新記錄一次
    use_wandb: bool = False
    project_name: str = "hexapod_balance"
    run_name: str = "transformer_ppo"


class GAE:
    """Generalized Advantage Estimation"""
    
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def compute_advantages(self, rewards, values, dones, next_value):
        """
        計算GAE優勢函數
        
        Args:
            rewards: [seq_len] 獎勵序列
            values: [seq_len] 價值序列  
            dones: [seq_len] 終止標誌
            next_value: scalar 下一個狀態的價值
        
        Returns:
            advantages: [seq_len] 優勢函數
            returns: [seq_len] 回報
        """
        seq_len = len(rewards)
        advantages = torch.zeros(seq_len)
        returns = torch.zeros(seq_len)
        
        # 從後往前計算
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
            # Return
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns


class ExperienceBuffer:
    """經驗緩衝區"""
    
    def __init__(self, config):
        self.config = config
        self.clear()
    
    def clear(self):
        """清空緩衝區"""
        self.states_seq = []      # 狀態序列
        self.actions_seq = []     # 動作序列
        self.rewards_seq = []     # 獎勵序列
        self.values_seq = []      # 價值序列
        self.log_probs_seq = []   # 對數概率序列
        self.dones_seq = []       # 終止標誌序列
        
        # Episode級別的數據
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_infos = []
    
    def add_episode(self, episode_data):
        """添加一個完整的episode"""
        self.states_seq.append(episode_data['states'])
        self.actions_seq.append(episode_data['actions'])
        self.rewards_seq.append(episode_data['rewards'])
        self.values_seq.append(episode_data['values'])
        self.log_probs_seq.append(episode_data['log_probs'])
        self.dones_seq.append(episode_data['dones'])
        
        # Episode統計
        self.episode_rewards.append(episode_data['total_reward'])
        self.episode_lengths.append(episode_data['length'])
        self.episode_infos.append(episode_data['info'])
    
    def get_batch_data(self):
        """獲取批次訓練數據"""
        if len(self.states_seq) == 0:
            return None
            
        return {
            'states_seq': self.states_seq,
            'actions_seq': self.actions_seq,
            'rewards_seq': self.rewards_seq,
            'values_seq': self.values_seq,
            'log_probs_seq': self.log_probs_seq,
            'dones_seq': self.dones_seq,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_infos': self.episode_infos
        }
    
    def size(self):
        """返回緩衝區中的episode數量"""
        return len(self.states_seq)


class PPOTrainer:
    """PPO訓練器"""
    
    def __init__(self, env, policy, config):
        self.env = env
        self.policy = policy
        self.config = config
        
        # 訓練組件
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.gae = GAE(gamma=config.gamma, gae_lambda=config.gae_lambda)
        self.experience_buffer = ExperienceBuffer(config)
        
        # 訓練狀態
        self.global_step = 0
        self.update_count = 0
        self.best_avg_reward = float('-inf')
        
        # 設置日誌
        self.setup_logging()
        
        # 策略包裝器（用於環境交互）
        from transformer_policy import TransformerPolicyWrapper
        self.policy_wrapper = TransformerPolicyWrapper(self.policy)
        
        print(f"✅ PPO訓練器初始化完成")
        print(f"📊 配置: {self.config}")
    
    def setup_logging(self):
        """設置日誌系統"""
        # 創建輸出目錄
        self.output_dir = f"runs/{self.config.run_name}_{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir)
        
        # Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.__dict__
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # 保存配置
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"📝 日誌保存到: {self.output_dir}")
    
    def collect_episode(self):
        """收集一個完整的episode"""
        # 重置環境和策略包裝器
        state = self.env.reset()
        self.policy_wrapper.reset_sequence_cache()
        
        # Episode數據
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        total_reward = 0
        step_count = 0
        episode_info = {}
        
        while True:
            # 獲取策略輸出
            self.policy.eval()
            with torch.no_grad():
                # 獲取當前序列數據
                seq_data = self.policy_wrapper.get_sequence_data()
                
                # 獲取動作和價值
                action, log_prob, entropy, value = self.policy.get_action_and_value(
                    seq_data['states'], 
                    seq_data['actions'], 
                    seq_data['rewards']
                )
                
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().item()
                value = value.cpu().item()
            
            # 執行動作
            next_state, reward, done, info = self.env.step(action)
            
            # 記錄數據
            states.append(state.copy())
            actions.append(action.copy())
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            # 更新策略包裝器
            self.policy_wrapper.update_sequence(next_state, action, reward)
            
            # 統計
            total_reward += reward
            step_count += 1
            self.global_step += 1
            
            # 檢查終止條件
            if done or step_count >= self.config.max_episode_steps:
                episode_info = {
                    'reason': info.get('reason', 'max_steps'),
                    'final_imu': info.get('imu_data', (0, 0)),
                    'final_gps': info.get('gps_data', (0, 0, 0))
                }
                break
            
            state = next_state
        
        # 轉換為張量
        episode_data = {
            'states': torch.tensor(np.array(states), dtype=torch.float32),
            'actions': torch.tensor(np.array(actions), dtype=torch.float32),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'values': torch.tensor(values, dtype=torch.float32),
            'log_probs': torch.tensor(log_probs, dtype=torch.float32),
            'dones': torch.tensor(dones, dtype=torch.float32),
            'total_reward': total_reward,
            'length': step_count,
            'info': episode_info
        }
        
        return episode_data
    
    def collect_experiences(self):
        """收集訓練數據"""
        self.experience_buffer.clear()
        
        for episode_idx in range(self.config.episodes_per_update):
            episode_data = self.collect_episode()
            self.experience_buffer.add_episode(episode_data)
            
            # 簡單進度顯示
            if episode_idx % max(1, self.config.episodes_per_update // 4) == 0:
                print(f"  收集episode {episode_idx+1}/{self.config.episodes_per_update}, "
                      f"獎勵: {episode_data['total_reward']:.3f}, "
                      f"長度: {episode_data['length']}")
        
        return self.experience_buffer.get_batch_data()
    
    def compute_advantages_and_returns(self, batch_data):
        """計算所有episode的優勢函數和回報"""
        all_advantages = []
        all_returns = []
        
        for i in range(len(batch_data['states_seq'])):
            rewards = batch_data['rewards_seq'][i]
            values = batch_data['values_seq'][i]
            dones = batch_data['dones_seq'][i]
            
            # 計算下一個狀態的價值（用於bootstrap）
            if dones[-1]:
                next_value = 0.0  # Episode結束，下一個狀態價值為0
            else:
                # Episode未結束，估計下一個狀態的價值
                # 這裡簡化為使用最後一個狀態的價值
                next_value = values[-1].item()
            
            # 計算GAE
            advantages, returns = self.gae.compute_advantages(rewards, values, dones, next_value)
            
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        return all_advantages, all_returns
    
    def ppo_update(self, batch_data):
        """執行PPO更新"""
        # 計算優勢函數和回報
        all_advantages, all_returns = self.compute_advantages_and_returns(batch_data)
        
        # 準備訓練數據
        train_data = []
        for i in range(len(batch_data['states_seq'])):
            # 構建序列數據
            seq_len = len(batch_data['states_seq'][i])
            
            # 創建填充的序列（確保長度為sequence_length）
            padded_states = torch.zeros(self.config.sequence_length, 6)
            padded_actions = torch.zeros(self.config.sequence_length, 6)
            padded_rewards = torch.zeros(self.config.sequence_length)
            
            # 填充實際數據
            actual_len = min(seq_len, self.config.sequence_length)
            padded_states[:actual_len] = batch_data['states_seq'][i][:actual_len]
            padded_actions[:actual_len] = batch_data['actions_seq'][i][:actual_len]
            padded_rewards[:actual_len] = batch_data['rewards_seq'][i][:actual_len]
            
            # 為每個時間步創建訓練數據
            for t in range(actual_len):
                train_data.append({
                    'states_seq': padded_states,
                    'actions_seq': padded_actions,
                    'rewards_seq': padded_rewards,
                    'action': batch_data['actions_seq'][i][t],
                    'old_log_prob': batch_data['log_probs_seq'][i][t],
                    'advantage': all_advantages[i][t],
                    'return': all_returns[i][t],
                    'old_value': batch_data['values_seq'][i][t]
                })
        
        # 標準化優勢函數
        advantages = torch.tensor([data['advantage'] for data in train_data])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for i, data in enumerate(train_data):
            data['advantage'] = advantages[i]
        
        # 多次更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        
        for epoch in range(self.config.update_epochs):
            # 隨機打亂數據
            np.random.shuffle(train_data)
            
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy_loss = 0
            epoch_kl_div = 0
            
            for data in train_data:
                self.policy.train()
                
                # 前向傳播
                action, log_prob, entropy, value = self.policy.get_action_and_value(
                    data['states_seq'],
                    data['actions_seq'],
                    data['rewards_seq'],
                    action=data['action']
                )
                
                # 計算損失
                # 1. Policy Loss (PPO Clipped)
                ratio = torch.exp(log_prob - data['old_log_prob'])
                surr1 = data['advantage'] * ratio
                surr2 = data['advantage'] * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                policy_loss = -torch.min(surr1, surr2)
                
                # 2. Value Loss
                value_loss = 0.5 * (value - data['return']) ** 2
                
                # 3. Entropy Loss
                entropy_loss = -entropy
                
                # 4. Total Loss
                loss = policy_loss + self.config.value_loss_coef * value_loss + self.config.entropy_coef * entropy_loss
                
                # 反向傳播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # 統計
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                
                # KL散度（用於監控）
                kl_div = data['old_log_prob'] - log_prob
                epoch_kl_div += kl_div.item()
            
            # Epoch統計
            num_samples = len(train_data)
            total_policy_loss += epoch_policy_loss / num_samples
            total_value_loss += epoch_value_loss / num_samples
            total_entropy_loss += epoch_entropy_loss / num_samples
            total_kl_div += epoch_kl_div / num_samples
        
        # 返回訓練統計
        num_epochs = self.config.update_epochs
        return {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy_loss': total_entropy_loss / num_epochs,
            'kl_div': total_kl_div / num_epochs,
            'num_samples': len(train_data)
        }
    
    def evaluate_policy(self, num_episodes=5):
        """評估策略性能"""
        print("🧪 評估策略性能...")
        
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            episode_data = self.collect_episode()
            eval_rewards.append(episode_data['total_reward'])
            eval_lengths.append(episode_data['length'])
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        print(f"  評估結果: 平均獎勵={avg_reward:.3f}, 平均長度={avg_length:.1f}")
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'rewards': eval_rewards,
            'lengths': eval_lengths
        }
    
    def save_model(self, suffix=""):
        """保存模型"""
        save_path = os.path.join(self.output_dir, f"model{suffix}.pt")
        
        save_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'update_count': self.update_count,
            'best_avg_reward': self.best_avg_reward
        }
        
        torch.save(save_data, save_path)
        print(f"💾 模型已保存: {save_path}")
        
        return save_path
    
    def load_model(self, checkpoint_path):
        """載入模型"""
        checkpoint = torch.load(checkpoint_path)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.update_count = checkpoint['update_count']
        self.best_avg_reward = checkpoint['best_avg_reward']
        
        print(f"📁 模型已載入: {checkpoint_path}")
    
    def update_learning_rate(self):
        """更新學習率"""
        if self.config.anneal_lr:
            frac = 1.0 - self.global_step / self.config.total_timesteps
            new_lr = frac * self.config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
    
    def log_metrics(self, metrics):
        """記錄訓練指標"""
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.update_count)
        
        # Weights & Biases
        if self.use_wandb:
            wandb.log(metrics, step=self.update_count)
    
    def train(self):
        """主訓練循環"""
        print("🚀 開始訓練...")
        print(f"目標: {self.config.total_timesteps:,} 總步數")
        
        start_time = time.time()
        
        while self.global_step < self.config.total_timesteps:
            self.update_count += 1
            
            # 收集經驗
            print(f"\n📊 更新 {self.update_count} - 收集經驗...")
            batch_data = self.collect_experiences()
            
            # PPO更新
            print("🔄 執行PPO更新...")
            update_stats = self.ppo_update(batch_data)
            
            # 更新學習率
            self.update_learning_rate()
            
            # 計算統計信息
            avg_reward = np.mean(batch_data['episode_rewards'])
            avg_length = np.mean(batch_data['episode_lengths'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 日誌記錄
            if self.update_count % self.config.log_frequency == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = self.global_step / elapsed_time
                
                metrics = {
                    'train/avg_reward': avg_reward,
                    'train/avg_length': avg_length,
                    'train/policy_loss': update_stats['policy_loss'],
                    'train/value_loss': update_stats['value_loss'],
                    'train/entropy_loss': update_stats['entropy_loss'],
                    'train/kl_div': update_stats['kl_div'],
                    'train/learning_rate': current_lr,
                    'train/global_step': self.global_step,
                    'train/steps_per_sec': steps_per_sec
                }
                
                self.log_metrics(metrics)
                
                print(f"📈 更新 {self.update_count}: 獎勵={avg_reward:.3f}, "
                      f"長度={avg_length:.1f}, SPS={steps_per_sec:.1f}")
            
            # 定期評估
            if self.update_count % self.config.eval_frequency == 0:
                eval_stats = self.evaluate_policy()
                
                eval_metrics = {
                    'eval/avg_reward': eval_stats['avg_reward'],
                    'eval/avg_length': eval_stats['avg_length']
                }
                self.log_metrics(eval_metrics)
                
                # 保存最佳模型
                if eval_stats['avg_reward'] > self.best_avg_reward:
                    self.best_avg_reward = eval_stats['avg_reward']
                    self.save_model("_best")
            
            # 定期保存
            if self.update_count % self.config.save_frequency == 0:
                self.save_model(f"_update_{self.update_count}")
            
            # 早停檢查
            if avg_reward >= self.config.target_reward:
                print(f"🎯 達到目標獎勵 {self.config.target_reward}! 訓練完成!")
                break
        
        # 訓練完成
        print("\n✅ 訓練完成!")
        self.save_model("_final")
        
        # 最終評估
        final_eval = self.evaluate_policy(num_episodes=10)
        print(f"🏆 最終評估: 平均獎勵={final_eval['avg_reward']:.3f}")
        
        # 清理
        self.writer.close()
        if self.use_wandb:
            wandb.finish()


def main():
    """主函數"""
    print("🤖 六足機器人平衡訓練 - PPO + Transformer")
    
    # 配置
    config = PPOConfig(
        # 基本配置
        total_timesteps=500000,
        episodes_per_update=4,
        update_epochs=4,
        
        # 網路配置
        hidden_size=128,
        n_layer=3,
        n_head=2,
        
        # PPO配置
        learning_rate=3e-4,
        clip_coef=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        
        # 日誌配置
        use_wandb=False,  # 設為True啟用Weights & Biases
        run_name=f"hexapod_ppo_{int(time.time())}"
    )
    
    # 創建環境
    print("🌍 初始化環境...")
    from hexapod_balance_env import HexapodBalanceEnv
    env = HexapodBalanceEnv(
        max_episode_steps=config.max_episode_steps,
        sequence_length=config.sequence_length
    )
    
    # 創建策略網路
    print("🧠 初始化Transformer策略網路...")
    from transformer_policy import TransformerPolicyNetwork
    policy = TransformerPolicyNetwork(
        state_dim=6,
        action_dim=6,
        sequence_length=config.sequence_length,
        hidden_size=config.hidden_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        dropout=config.dropout,
        action_range=config.action_range
    )
    
    # 創建訓練器
    trainer = PPOTrainer(env, policy, config)
    
    # 開始訓練
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⏹️  訓練被中斷")
        trainer.save_model("_interrupted")
    except Exception as e:
        print(f"\n❌ 訓練出錯: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_model("_error")
    finally:
        print("🧹 清理資源...")
        env.close()


if __name__ == "__main__":
    main()