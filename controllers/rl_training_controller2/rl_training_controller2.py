"""
Webots 控制器程式 - 六足機器人 TrXL-PPO 訓練
檔案位置: controllers/rl_training_controller/rl_training_controller.py
"""

import sys
import os
import time  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 導入環境和 TrXL-PPO 相關模組
from hexapod_env import HexapodBalanceEnv
from ppo_trxl import Agent, Args
from dataclasses import dataclass
from collections import deque
import tyro
from torch.utils.tensorboard import SummaryWriter

@dataclass 
class WebotsTrXLArgs(Args):
    """Webots 專用的 TrXL-PPO 參數"""
    # 覆蓋原參數以適應單環境
    num_envs: int = 1                    # 單環境
    num_steps: int = 2048               # 增加步數補償並行度
    batch_size: int = 2048              # 1 * 2048  
    minibatch_size: int = 256           # 調整小批次大小
    num_minibatches: int = 8            # 2048 / 256 = 8
    
    # 調整訓練參數
    total_timesteps: int = 10000000     # 降低總訓練步數
    init_lr: float = 3e-4               # 略微調高學習率
    final_lr: float = 1e-5
    update_epochs: int = 4              # 增加更新輪數
    
    # Transformer 參數
    trxl_memory_length: int = 64        # 適合機器人任務的記憶長度
    trxl_dim: int = 256                 # 降低維度節省計算
    trxl_num_layers: int = 2            # 減少層數
    
    # Webots 專用參數
    max_episode_steps: int = 2000       # episode 最大步數
    save_interval: int = 100            # 模型保存間隔(iterations)
    eval_interval: int = 50             # 評估間隔(iterations)
    webots_mode: str = "training"       # "training" 或 "evaluation"


class SingleEnvMemoryManager:
    """單環境記憶體管理器 - 簡化版本"""
    
    def __init__(self, max_episode_steps, memory_length, num_layers, embed_dim):
        self.max_episode_steps = max_episode_steps
        self.memory_length = memory_length
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        # 當前episode的記憶體
        self.current_episode_memory = torch.zeros(
            max_episode_steps, num_layers, embed_dim
        )
        self.current_episode_step = 0
        
        # 生成記憶體遮罩和索引（一次性計算）
        self._setup_memory_patterns()
        
    def _setup_memory_patterns(self):
        """設置記憶體遮罩和索引模式"""
        # 生成下三角遮罩
        self.memory_masks = []
        for i in range(self.memory_length):
            mask = torch.zeros(self.memory_length, dtype=torch.bool)
            mask[:i] = True  # 只能看到之前的步驟
            self.memory_masks.append(mask)
        
        # 生成滑動窗口索引
        self.memory_indices = []
        for step in range(self.max_episode_steps):
            # 計算當前步驟的記憶體窗口
            start_idx = max(0, step - self.memory_length + 1)
            end_idx = step + 1
            
            # 創建索引數組
            indices = torch.arange(start_idx, end_idx)
            
            # 如果不足memory_length，用0填充
            if len(indices) < self.memory_length:
                padding = torch.zeros(self.memory_length - len(indices), dtype=torch.long)
                indices = torch.cat([padding, indices])
            
            self.memory_indices.append(indices)
        
        self.memory_indices = torch.stack(self.memory_indices)
    
    def get_memory_window(self, step):
        """獲取指定步驟的記憶體窗口"""
        step = min(step, self.max_episode_steps - 1)
        indices = self.memory_indices[step]
        mask = self.memory_masks[min(step, self.memory_length - 1)]
        
        # 提取記憶體窗口
        memory_window = self.current_episode_memory[indices]  # [memory_length, num_layers, embed_dim]
        
        return memory_window.unsqueeze(0), mask.unsqueeze(0), indices.unsqueeze(0)
    
    def update_memory(self, step, new_memory):
        """更新指定步驟的記憶體"""
        if step < self.max_episode_steps:
            self.current_episode_memory[step] = new_memory.squeeze(0)
            self.current_episode_step = step
    
    def reset(self):
        """重置episode記憶體"""
        self.current_episode_memory.zero_()
        self.current_episode_step = 0


class SingleEnvRolloutBuffer:
    """單環境rollout緩衝區"""
    
    def __init__(self, num_steps, obs_shape, action_dim, memory_length, num_layers, embed_dim):
        self.num_steps = num_steps
        self.action_dim = action_dim
        
        # 基本數據緩衝區
        self.observations = torch.zeros((num_steps,) + obs_shape)
        self.actions = torch.zeros((num_steps, action_dim))
        self.log_probs = torch.zeros((num_steps, action_dim))
        self.rewards = torch.zeros(num_steps)
        self.values = torch.zeros(num_steps)
        self.dones = torch.zeros(num_steps, dtype=torch.bool)
        
        # 記憶體相關緩衝區
        self.memory_windows = torch.zeros((num_steps, memory_length, num_layers, embed_dim))
        self.memory_masks = torch.zeros((num_steps, memory_length), dtype=torch.bool)
        self.memory_indices = torch.zeros((num_steps, memory_length), dtype=torch.long)
        
        # Episode邊界追蹤
        self.episode_starts = torch.zeros(num_steps, dtype=torch.bool)
        self.episode_ids = torch.zeros(num_steps, dtype=torch.long)
        
        self.step = 0
        self.current_episode_id = 0
    
    def add(self, obs, action, log_prob, reward, value, done, 
            memory_window, memory_mask, memory_indices, episode_start=False):
        """添加一步數據"""
        if self.step >= self.num_steps:
            raise IndexError("Buffer is full")
        
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        
        self.memory_windows[self.step] = memory_window.squeeze(0)
        self.memory_masks[self.step] = memory_mask.squeeze(0)
        self.memory_indices[self.step] = memory_indices.squeeze(0)
        
        self.episode_starts[self.step] = episode_start
        self.episode_ids[self.step] = self.current_episode_id
        
        if done:
            self.current_episode_id += 1
        
        self.step += 1
    
    def get_batch_data(self):
        """獲取批次數據用於訓練"""
        return {
            'observations': self.observations[:self.step],
            'actions': self.actions[:self.step],
            'log_probs': self.log_probs[:self.step],
            'rewards': self.rewards[:self.step],
            'values': self.values[:self.step],
            'dones': self.dones[:self.step],
            'memory_windows': self.memory_windows[:self.step],
            'memory_masks': self.memory_masks[:self.step],
            'memory_indices': self.memory_indices[:self.step],
            'episode_starts': self.episode_starts[:self.step],
            'episode_ids': self.episode_ids[:self.step],
        }
    
    def reset(self):
        """重置緩衝區"""
        self.step = 0
        self.current_episode_id = 0


class WebotsTrainer:
    """Webots 環境下的 TrXL-PPO 訓練器 - 修正版本"""
    
    def __init__(self, args):
        self.args = args
        
        # 設置設備
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        
        # 創建環境
        self.env = HexapodBalanceEnv(max_episode_steps=args.max_episode_steps)
        
        # 檢查環境兼容性
        self._validate_environment()
        
        # 創建智能體
        if hasattr(self.env.action_space, 'shape'):
            action_space_shape = self.env.action_space.shape
        else:
            action_space_shape = (self.env.action_space.n,)

        self.agent = Agent(
            args, 
            self.env.observation_space, 
            action_space_shape,
            args.max_episode_steps
        ).to(self.device)
        
        # 創建優化器
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=args.init_lr)
        
        # 修正：創建記憶體管理器
        self.memory_manager = SingleEnvMemoryManager(
            args.max_episode_steps,
            args.trxl_memory_length,
            args.trxl_num_layers,
            args.trxl_dim
        )
        
        # 修正：創建rollout緩衝區
        self.rollout_buffer = SingleEnvRolloutBuffer(
            args.num_steps,
            self.env.observation_space.shape,
            action_space_shape[0],  # 連續動作維度
            args.trxl_memory_length,
            args.trxl_num_layers,
            args.trxl_dim
        )
        
        # 訓練狀態
        self.global_step = 0
        self.iteration = 0
        self.episode_count = 0
        
        # 監控統計
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_infos = deque(maxlen=100)
        
        # TensorBoard
        run_name = f"webots_hexapod_trxl_ppo_{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        
        print("🚀 Webots TrXL-PPO 訓練器初始化完成")
        print(f"📊 環境: {self.env.observation_space} -> {self.env.action_space}")
        print(f"🧠 智能體: {args.trxl_dim}d Transformer, {args.trxl_num_layers} layers")
        print(f"💾 設備: {self.device}")
        print(f"📈 TensorBoard: runs/{run_name}")

    def _validate_environment(self):
        """驗證環境兼容性"""
        # 檢查觀測空間
        assert len(self.env.observation_space.shape) == 1, "需要向量觀測空間"
        
        # 檢查動作空間  
        assert hasattr(self.env.action_space, 'shape'), "需要連續動作空間"
        
        # 測試環境
        obs, info = self.env.reset()
        assert obs.shape == self.env.observation_space.shape, "觀測維度不匹配"
        
        # 測試動作
        test_action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(test_action)
        
        print("✅ 環境兼容性驗證通過")

    def collect_rollout(self):
        """收集一個 rollout 的經驗 - 修正版本"""
        # 重置rollout緩衝區
        self.rollout_buffer.reset()
        
        # 獲取當前觀測
        if not hasattr(self, 'current_obs'):
            obs, info = self.env.reset()
            self.current_obs = torch.tensor(obs, dtype=torch.float32)
            self.current_done = False
            self.memory_manager.reset()
            episode_start = True
        else:
            episode_start = False
        
        # 收集經驗
        for step in range(self.args.num_steps):
            self.global_step += 1
            
            # 獲取記憶體窗口
            memory_window, memory_mask, memory_indices = self.memory_manager.get_memory_window(
                self.memory_manager.current_episode_step
            )
            
            # 智能體決策
            with torch.no_grad():
                obs_tensor = self.current_obs.unsqueeze(0)
                action, log_prob, _, value, new_memory = self.agent.get_action_and_value(
                    obs_tensor,
                    memory_window,
                    memory_mask,
                    memory_indices
                )
                
                # 更新記憶體
                self.memory_manager.update_memory(
                    self.memory_manager.current_episode_step, 
                    new_memory
                )
            
            # 執行動作
            obs, reward, terminated, truncated, info = self.env.step(action[0].cpu().numpy())
            done = terminated or truncated
            
            # 存儲數據到緩衝區
            self.rollout_buffer.add(
                obs=self.current_obs,
                action=action[0],
                log_prob=log_prob[0],
                reward=reward,
                value=value[0],
                done=done,
                memory_window=memory_window,
                memory_mask=memory_mask,
                memory_indices=memory_indices,
                episode_start=episode_start
            )
            
            # 更新狀態
            self.current_obs = torch.tensor(obs, dtype=torch.float32)
            self.current_done = done
            episode_start = False
            
            # Episode 管理
            if done:
                # 記錄 episode 結果
                self.episode_count += 1
                episode_info = {
                    'episode_reward': sum([self.rollout_buffer.rewards[i].item() 
                                         for i in range(max(0, step - self.memory_manager.current_episode_step), step + 1)]),
                    'episode_length': self.memory_manager.current_episode_step + 1,
                    'reason': info.get('reason', ''),
                    'stability_reward': info.get('stability_reward', 0),
                    'penalty': info.get('penalty', 0)
                }
                self.episode_infos.append(episode_info)
                
                # 重置環境和記憶體
                obs, info = self.env.reset()
                self.current_obs = torch.tensor(obs, dtype=torch.float32)
                self.current_done = False
                self.memory_manager.reset()
                episode_start = True
            else:
                self.memory_manager.current_episode_step += 1
        
        return self.rollout_buffer.get_batch_data()

    def compute_advantages(self, rollout_data):
        """計算 GAE advantages - 修正版本"""
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']
        
        # 計算下一個值
        with torch.no_grad():
            memory_window, memory_mask, memory_indices = self.memory_manager.get_memory_window(
                self.memory_manager.current_episode_step
            )
            
            next_value = self.agent.get_value(
                self.current_obs.unsqueeze(0),
                memory_window,
                memory_mask,
                memory_indices
            )[0]
        
        # GAE 計算
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - float(self.current_done)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1].float()
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        
        return advantages, returns

    def update_policy(self, rollout_data, advantages, returns):
        """更新策略 - 修正版本"""
        # 準備訓練數據
        b_obs = rollout_data['observations']
        b_actions = rollout_data['actions']
        b_log_probs = rollout_data['log_probs']
        b_values = rollout_data['values']
        b_memory_windows = rollout_data['memory_windows']
        b_memory_masks = rollout_data['memory_masks']
        b_memory_indices = rollout_data['memory_indices']
        
        batch_size = len(b_obs)
        
        # 學習率退火
        frac = 1.0 - (self.global_step - self.args.num_steps) / self.args.total_timesteps
        frac = max(0.0, frac)
        lr = frac * self.args.init_lr + (1 - frac) * self.args.final_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        # 熵系數退火
        ent_coef = frac * self.args.init_ent_coef + (1 - frac) * self.args.final_ent_coef
        
        clipfracs = []
        
        for epoch in range(self.args.update_epochs):
            # 修正：正確的批次處理
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.args.minibatch_size):
                end = min(start + self.args.minibatch_size, batch_size)
                mb_inds = indices[start:end]
                
                # 提取minibatch數據
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_log_probs = b_log_probs[mb_inds]
                mb_values = b_values[mb_inds]
                mb_memory_windows = b_memory_windows[mb_inds]
                mb_memory_masks = b_memory_masks[mb_inds]
                mb_memory_indices = b_memory_indices[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                
                # 前向傳播
                _, new_log_probs, entropy, new_values, _ = self.agent.get_action_and_value(
                    mb_obs,
                    mb_memory_windows,
                    mb_memory_masks,
                    mb_memory_indices,
                    mb_actions
                )
                
                # 策略損失
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # 修正：處理連續動作的log_probs
                logratio = new_log_probs - mb_log_probs
                ratio = torch.exp(logratio)
                
                # 為每個動作維度計算優勢
                mb_advantages_expanded = mb_advantages.unsqueeze(1).expand_as(ratio)
                
                # PPO 截斷
                pg_loss1 = -mb_advantages_expanded * ratio
                pg_loss2 = -mb_advantages_expanded * torch.clamp(
                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 價值損失
                v_loss = ((new_values - mb_returns) ** 2).mean()
                
                # 熵損失
                entropy_loss = entropy.mean()
                
                # 總損失
                loss = pg_loss - ent_coef * entropy_loss + self.args.vf_coef * v_loss
                
                # 反向傳播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                
                # 記錄統計
                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        
        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(), 
            'entropy_loss': entropy_loss.item(),
            'clipfrac': np.mean(clipfracs),
            'learning_rate': lr,
            'entropy_coef': ent_coef
        }

    def train(self):
        """主訓練循環"""
        print("🚀 開始訓練...")
        start_time = time.time()
        
        num_iterations = self.args.total_timesteps // self.args.batch_size
        
        for iteration in range(1, num_iterations + 1):
            self.iteration = iteration
            
            # 收集經驗
            rollout_data = self.collect_rollout()
            
            # 計算 advantages
            advantages, returns = self.compute_advantages(rollout_data)
            
            # 更新策略
            update_info = self.update_policy(rollout_data, advantages, returns)
            
            # 記錄統計
            self._log_training_stats(update_info, start_time)
            
            # 保存檢查點
            if iteration % self.args.save_interval == 0:
                self._save_checkpoint(iteration)
            
            # 評估
            if iteration % self.args.eval_interval == 0:
                self._evaluate()
        
        print("✅ 訓練完成!")
        self._save_checkpoint(iteration, final=True)

    def _log_training_stats(self, update_info, start_time):
        """記錄訓練統計"""
        if len(self.episode_infos) > 0:
            # Episode 統計
            episode_rewards = [info['episode_reward'] for info in self.episode_infos]
            episode_lengths = [info['episode_length'] for info in self.episode_infos]
            
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            
            # 打印進度
            sps = int(self.global_step / (time.time() - start_time))
            print(f"迭代 {self.iteration:4d} | 步數 {self.global_step:8d} | "
                  f"獎勵 {avg_reward:6.2f} | 長度 {avg_length:4.0f} | "
                  f"SPS {sps:4d} | 策略損失 {update_info['policy_loss']:.3f}")
            
            # TensorBoard 記錄
            self.writer.add_scalar("episode/reward_mean", avg_reward, self.global_step)
            self.writer.add_scalar("episode/length_mean", avg_length, self.global_step)
            self.writer.add_scalar("episode/count", len(self.episode_infos), self.global_step)
            
        # 訓練統計
        for key, value in update_info.items():
            self.writer.add_scalar(f"training/{key}", value, self.global_step)
        
        self.writer.add_scalar("performance/sps", sps, self.global_step)

    def _save_checkpoint(self, iteration, final=False):
        """保存檢查點"""
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }
        
        filename = f"checkpoint_final.pt" if final else f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, filename)
        print(f"💾 檢查點已保存: {filename}")

    def _evaluate(self):
        """評估智能體性能"""
        print("🔍 進行評估...")
        eval_episodes = 5
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(eval_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # 重置評估記憶體
            eval_memory_manager = SingleEnvMemoryManager(
                self.args.max_episode_steps,
                self.args.trxl_memory_length,
                self.args.trxl_num_layers,
                self.args.trxl_dim
            )
            
            while not done and episode_length < self.args.max_episode_steps:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # 獲取記憶體窗口
                memory_window, memory_mask, memory_indices = eval_memory_manager.get_memory_window(
                    eval_memory_manager.current_episode_step
                )
                
                with torch.no_grad():
                    action, _, _, _, new_memory = self.agent.get_action_and_value(
                        obs_tensor, memory_window, memory_mask, memory_indices
                    )
                    eval_memory_manager.update_memory(
                        eval_memory_manager.current_episode_step, new_memory
                    )
                
                obs, reward, terminated, truncated, info = self.env.step(action[0].cpu().numpy())
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                eval_memory_manager.current_episode_step += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_length = np.mean(eval_lengths)
        
        print(f"📊 評估結果: 平均獎勵 {avg_eval_reward:.2f}, 平均長度 {avg_eval_length:.1f}")
        
        self.writer.add_scalar("evaluation/reward_mean", avg_eval_reward, self.global_step)
        self.writer.add_scalar("evaluation/length_mean", avg_eval_length, self.global_step)


def main():
    """主函數"""
    # 解析參數
    args = tyro.cli(WebotsTrXLArgs)
    
    # 設置隨機種子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # 創建並運行訓練器
    trainer = WebotsTrainer(args)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  訓練被中斷")
        trainer._save_checkpoint(trainer.iteration, final=True)
    except Exception as e:
        print(f"❌ 訓練錯誤: {e}")
        raise
    finally:
        trainer.writer.close()
        trainer.env.close()


if __name__ == "__main__":
    main()