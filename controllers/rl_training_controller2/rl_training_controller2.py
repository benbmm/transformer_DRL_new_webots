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


class WebotsTrainer:
    """Webots 環境下的 TrXL-PPO 訓練器"""
    
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
        
        # 記憶體相關
        self.current_memory = torch.zeros(
            (1, args.max_episode_steps, args.trxl_num_layers, args.trxl_dim), 
            dtype=torch.float32
        )
        self.current_episode_step = 0
        
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
        """收集一個 rollout 的經驗"""
        # 儲存 rollout 數據
        observations = torch.zeros((self.args.num_steps, 1) + self.env.observation_space.shape)
        actions = torch.zeros((self.args.num_steps, 1, self.env.action_space.shape[0]))
        rewards = torch.zeros((self.args.num_steps, 1))
        dones = torch.zeros((self.args.num_steps, 1))
        log_probs = torch.zeros((self.args.num_steps, 1, self.env.action_space.shape[0]))
        values = torch.zeros((self.args.num_steps, 1))
        
        # 簡化的記憶體處理
        stored_memories = []
        stored_memory_masks = torch.zeros((self.args.num_steps, 1, self.args.trxl_memory_length), dtype=torch.bool)
        stored_memory_indices = torch.zeros((self.args.num_steps, 1, self.args.trxl_memory_length), dtype=torch.long)
        
        # 生成簡化的記憶體遮罩 (只需要一次)
        memory_mask = torch.tril(torch.ones((self.args.trxl_memory_length, self.args.trxl_memory_length)), diagonal=-1)
        # 獲取當前觀測
        if not hasattr(self, 'current_obs'):
            obs, info = self.env.reset()
            self.current_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            self.current_done = torch.tensor([False])
            self.current_episode_step = 0
            self.current_memory = torch.zeros(
                (1, self.args.max_episode_steps, self.args.trxl_num_layers, self.args.trxl_dim)
            )
        
        # 收集經驗
        for step in range(self.args.num_steps):
            self.global_step += 1
            
            # 儲存當前觀測和done
            observations[step] = self.current_obs
            dones[step] = self.current_done.float()
            
            # 準備記憶體窗口
            episode_step = min(self.current_episode_step, len(memory_indices) - 1)
            stored_memory_masks[step] = memory_mask[
                min(episode_step, self.args.trxl_memory_length - 1)
            ].unsqueeze(0)
            stored_memory_indices[step] = memory_indices[episode_step].unsqueeze(0)
            
            # 提取記憶體窗口
            memory_window_indices = stored_memory_indices[step][0]  # [memory_length]
            # 確保索引在有效範圍內
            valid_indices = torch.clamp(memory_window_indices, 0, self.current_memory.shape[1] - 1)
            memory_window = self.current_memory[0, valid_indices].unsqueeze(0)  # [1, memory_length, num_layers, dim]
            
            # 智能體決策
            with torch.no_grad():
                action, logprob, _, value, new_memory = self.agent.get_action_and_value(
                    self.current_obs,
                    memory_window,
                    stored_memory_masks[step],
                    stored_memory_indices[step]
                )
                
                # 更新記憶體
                if self.current_episode_step < self.current_memory.shape[1]:
                    self.current_memory[0, self.current_episode_step] = new_memory[0]
                
                # 儲存數據
                actions[step] = action
                log_probs[step] = logprob
                values[step] = value
            
            # 執行動作
            obs, reward, terminated, truncated, info = self.env.step(action[0].cpu().numpy())
            done = terminated or truncated
            
            # 更新狀態
            rewards[step] = torch.tensor([reward])
            self.current_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            self.current_done = torch.tensor([done])
            
            # Episode 管理
            if done:
                # 記錄 episode 結果
                self.episode_count += 1
                episode_info = {
                    'episode_reward': sum([rewards[i].item() for i in range(max(0, step-self.current_episode_step), step+1)]),
                    'episode_length': self.current_episode_step + 1,
                    'reason': info.get('reason', ''),
                    'stability_reward': info.get('stability_reward', 0),
                    'penalty': info.get('penalty', 0)
                }
                self.episode_infos.append(episode_info)
                
                # 保存當前記憶體
                stored_memories.append(self.current_memory[0].clone())
                
                # 重置環境
                obs, info = self.env.reset()
                self.current_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                self.current_done = torch.tensor([False])
                self.current_episode_step = 0
                self.current_memory = torch.zeros(
                    (1, self.args.max_episode_steps, self.args.trxl_num_layers, self.args.trxl_dim)
                )
            else:
                self.current_episode_step += 1
            
            # 為下一步準備記憶體引用
            if step == 0 or done:
                stored_memories.append(self.current_memory[0].clone())
        
        # 確保記憶體列表長度正確
        while len(stored_memories) < self.args.num_steps:
            stored_memories.append(self.current_memory[0].clone())
        
        stored_memories = stored_memories[:self.args.num_steps]
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'log_probs': log_probs,
            'values': values,
            'stored_memories': torch.stack(stored_memories),
            'stored_memory_masks': stored_memory_masks,
            'stored_memory_indices': stored_memory_indices
        }

    def compute_advantages(self, rollout_data):
        """計算 GAE advantages"""
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']
        
        # 計算下一個值
        with torch.no_grad():
            memory_window_indices = rollout_data['stored_memory_indices'][-1]
            # 安全的記憶體索引提取
            max_valid_idx = self.current_memory.shape[1] - 1
            safe_indices = torch.clamp(memory_window_indices[0], 0, max_valid_idx)
            memory_window = self.current_memory[:, safe_indices]
            
            next_value = self.agent.get_value(
                self.current_obs,
                memory_window,
                rollout_data['stored_memory_masks'][-1],
                memory_window_indices
            )
        
        # GAE 計算
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.args.num_steps)):
            if t == self.args.num_steps - 1:
                nextnonterminal = 1.0 - self.current_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        
        return advantages, returns

    def update_policy(self, rollout_data, advantages, returns):
        """更新策略"""
        # 展平數據
        b_obs = rollout_data['observations'].reshape(-1, *rollout_data['observations'].shape[2:])
        b_logprobs = rollout_data['log_probs'].reshape(-1, *rollout_data['log_probs'].shape[2:])
        b_actions = rollout_data['actions'].reshape(-1, *rollout_data['actions'].shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rollout_data['values'].reshape(-1)
        b_memory_masks = rollout_data['stored_memory_masks'].reshape(-1, *rollout_data['stored_memory_masks'].shape[2:])
        b_memory_indices = rollout_data['stored_memory_indices'].reshape(-1, *rollout_data['stored_memory_indices'].shape[2:])
        stored_memories = rollout_data['stored_memories']
        
        # 學習率退火
        frac = 1.0 - (self.global_step - self.args.num_steps) / self.args.total_timesteps
        frac = max(0.0, frac)  # 確保不會是負數
        lr = frac * self.args.init_lr + (1 - frac) * self.args.final_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        # 熵系數退火
        ent_coef = frac * self.args.init_ent_coef + (1 - frac) * self.args.final_ent_coef
        
        clipfracs = []
        
        for epoch in range(self.args.update_epochs):
            b_inds = torch.randperm(self.args.batch_size)
            
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # 修復記憶體窗口索引問題
                try:
                    # 獲取記憶體索引 - 需要確保形狀匹配
                    mb_memory_indices = b_memory_indices[mb_inds]  # [minibatch_size, memory_length]
                    
                    # 創建記憶體窗口
                    batch_size = len(mb_inds)
                    memory_length = mb_memory_indices.shape[1]
                    embed_dim = stored_memories.shape[-1]
                    num_layers = stored_memories.shape[-2]
                    
                    # 為每個 minibatch 樣本創建記憶體窗口
                    mb_memory_windows = torch.zeros(
                        batch_size, memory_length, num_layers, embed_dim,
                        device=stored_memories.device, dtype=stored_memories.dtype
                    )
                    
                    # 逐個樣本處理記憶體窗口
                    for i, sample_idx in enumerate(mb_inds):
                        # 計算這個樣本來自哪個 step
                        step_idx = sample_idx // 1  # 因為是單環境，每個step只有1個樣本
                        
                        # 確保 step_idx 在有效範圍內
                        step_idx = min(step_idx, stored_memories.shape[0] - 1)
                        
                        # 獲取該步驟的記憶體
                        step_memory = stored_memories[step_idx]  # [max_episode_steps, num_layers, embed_dim]
                        
                        # 獲取記憶體索引
                        indices = mb_memory_indices[i]  # [memory_length]
                        
                        # 確保索引在有效範圍內
                        indices = torch.clamp(indices, 0, step_memory.shape[0] - 1)
                        
                        # 提取記憶體窗口
                        mb_memory_windows[i] = step_memory[indices]  # [memory_length, num_layers, embed_dim]
                    
                except Exception as e:
                    print(f"記憶體窗口處理錯誤: {e}")
                    # 使用備用方案：創建零記憶體
                    batch_size = len(mb_inds)
                    memory_length = self.args.trxl_memory_length
                    mb_memory_windows = torch.zeros(
                        batch_size, memory_length, self.args.trxl_num_layers, self.args.trxl_dim,
                        device=self.device
                    )
                
                # 前向傳播
                _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                    b_obs[mb_inds], 
                    mb_memory_windows,
                    b_memory_masks[mb_inds], 
                    b_memory_indices[mb_inds], 
                    b_actions[mb_inds]
                )
                
                # 策略損失
                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # 處理連續動作的形狀
                if len(mb_advantages.shape) == 1:
                    mb_advantages = mb_advantages.unsqueeze(1)
                if len(ratio.shape) == 2 and ratio.shape[1] == 1:
                    mb_advantages = mb_advantages.expand_as(ratio)

                # PPO 截斷
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # 價值損失
                v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
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
            eval_memory = torch.zeros(
                (1, self.args.max_episode_steps, self.args.trxl_num_layers, self.args.trxl_dim)
            )
            eval_step = 0
            
            while not done and episode_length < self.args.max_episode_steps:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # 準備記憶體窗口
                memory_indices = torch.arange(
                    max(0, eval_step - self.args.trxl_memory_length + 1),
                    eval_step + 1
                ).unsqueeze(0)
                
                if len(memory_indices[0]) < self.args.trxl_memory_length:
                    padding = self.args.trxl_memory_length - len(memory_indices[0])
                    memory_indices = torch.cat([
                        torch.zeros(1, padding, dtype=torch.long),
                        memory_indices
                    ], dim=1)
                
                memory_window = eval_memory[:, memory_indices[0]]
                memory_mask = torch.tril(torch.ones((self.args.trxl_memory_length, self.args.trxl_memory_length)), diagonal=-1)
                memory_mask = memory_mask[min(eval_step, self.args.trxl_memory_length - 1)].unsqueeze(0)
                
                with torch.no_grad():
                    action, _, _, _, new_memory = self.agent.get_action_and_value(
                        obs_tensor, memory_window, memory_mask, memory_indices
                    )
                    eval_memory[0, eval_step] = new_memory[0]
                
                obs, reward, terminated, truncated, info = self.env.step(action[0].cpu().numpy())
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                eval_step += 1
            
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