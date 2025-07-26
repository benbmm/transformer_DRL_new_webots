"""
六足機器人地形適應控制系統 - PPO訓練版本
使用Transformer + PPO在傾斜平台上訓練平衡控制
"""

import sys
import time
import math
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from transformers import GPT2Config, GPT2Model
from collections import deque
from controller import Supervisor, Robot

class HexapodTransformer(nn.Module):
    """六足機器人地形適應Transformer - Policy Network"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用GPT2架構作為backbone
        transformer_config = GPT2Config(
            vocab_size=1,
            n_positions=config['sequence_length'],
            n_embd=config['hidden_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head']
        )
        
        self.transformer = GPT2Model(transformer_config)
        
        # 輸入投影層 (state + action + reward)
        input_dim = config['state_dim'] + config['action_dim'] + config['reward_dim']
        self.input_projection = nn.Linear(input_dim, config['hidden_size'])
        
        # Actor網絡：輸出動作均值和標準差
        self.actor_mean = nn.Linear(config['hidden_size'], config['action_dim'])
        self.actor_logstd = nn.Parameter(torch.zeros(config['action_dim']))
        
        # Critic網絡：輸出狀態值
        self.critic = nn.Linear(config['hidden_size'], 1)
        
    def forward(self, state_sequence, action_sequence, reward_sequence):
        batch_size, seq_len = state_sequence.shape[:2]
        
        # 組合輸入
        combined_input = torch.cat([state_sequence, action_sequence, reward_sequence], dim=-1)
        
        # 輸入投影
        embeddings = self.input_projection(combined_input)
        
        # Transformer處理
        transformer_outputs = self.transformer(inputs_embeds=embeddings)
        last_hidden = transformer_outputs.last_hidden_state[:, -1, :]
        
        return last_hidden
    
    def get_action_and_value(self, state_sequence, action_sequence, reward_sequence, action=None):
        hidden = self.forward(state_sequence, action_sequence, reward_sequence)
        
        # Actor輸出
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        # Critic輸出
        value = self.critic(hidden)
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

class PPOTrainer:
    """PPO訓練器"""
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'])
        
        # 訓練參數
        self.clip_coef = config.get('clip_coef', 0.2)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # 經驗緩存
        self.batch_size = config.get('batch_size', 64)
        self.num_epochs = config.get('num_epochs', 4)
        
    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
        """計算GAE優勢函數"""
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, rewards, logprobs, values, dones):
        """PPO更新 - 保持序列數據的時間順序"""
        # 計算下一個狀態的值（這裡簡化為0）
        next_value = torch.zeros(1)
        
        # 計算優勢函數和回報
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 標準化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = len(states)
        total_pg_loss = 0
        total_v_loss = 0
        total_entropy_loss = 0
        
        # 多輪更新
        for epoch in range(self.num_epochs):
            # ✅ 保持序列順序：連續取樣而非隨機打亂
            # 對於Transformer序列模型，時間順序很重要
            for start in range(0, batch_size, self.batch_size):
                end = min(start + self.batch_size, batch_size)
                
                if start >= end:
                    continue
                
                # ✅ 使用連續索引保持時序
                mb_indices = list(range(start, end))
                
                # 選擇小批量數據（保持時間順序）
                mb_states = [states[i] for i in mb_indices]
                mb_actions = [actions[i] for i in mb_indices]
                mb_rewards = [rewards[i] for i in mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_oldlogprobs = logprobs[mb_indices]
                
                # 準備序列數據
                state_seq = torch.stack([s[0] for s in mb_states])
                action_seq = torch.stack([s[1] for s in mb_states])
                reward_seq = torch.stack([s[2] for s in mb_states])
                
                # 前向傳播
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    state_seq, action_seq, reward_seq, 
                    torch.stack([a for a in mb_actions])
                )
                
                # PPO損失
                logratio = newlogprob - mb_oldlogprobs
                ratio = logratio.exp()
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = F.mse_loss(newvalue.flatten(), mb_returns)
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        num_updates = self.num_epochs * ((batch_size + self.batch_size - 1) // self.batch_size)
        return {
            'pg_loss': total_pg_loss / num_updates,
            'v_loss': total_v_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates
        }

class HexapodExperimentalController:
    """六足機器人實驗平台控制器 - 包含PPO訓練"""
    
    def __init__(self):
        # 基本設置
        self.MAX_STEPS = 2000
        self.NUM_LEGS = 6
        self.timestep = 20  # Webots時間步長
        
        # 初始化Webots Supervisor
        self.robot = Supervisor()
        self.current_step = 0
        
        # 控制參數
        self.body_height_offset = 0.5
        self.control_start_step = 100
        
        # Transformer配置
        self.transformer_config = {
            'sequence_length': 50,
            'state_dim': 6,
            'action_dim': 6,  # 只控制6個膝關節
            'reward_dim': 1,
            'hidden_size': 128,
            'n_layer': 3,
            'n_head': 2,
            'learning_rate': 3e-4,
            'batch_size': 32,
            'num_epochs': 4,
            'clip_coef': 0.2,
            'vf_coef': 0.5,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5
        }
        
        # 初始化神經網絡
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = HexapodTransformer(self.transformer_config).to(self.device)
        self.trainer = PPOTrainer(self.agent, self.transformer_config)
        
        # 序列緩存
        self.state_buffer = deque(maxlen=self.transformer_config['sequence_length'])
        self.action_buffer = deque(maxlen=self.transformer_config['sequence_length'])
        self.reward_buffer = deque(maxlen=self.transformer_config['sequence_length'])
        
        # 訓練數據收集
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_logprobs = []
        self.episode_values = []
        self.episode_dones = []
        
        # 初始化各種組件
        self.init_motors()
        self.init_sensors()
        self.init_platform_control()
        self._initialize_buffers()
        self.create_output_directories()
        
        # 訓練統計
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # 終止相關
        self.episode_terminated = False
        self.termination_reason = None
        
        print("六足機器人實驗平台控制器已初始化")
        print(f"使用設備: {self.device}")
        print(f"控制頻率: {1000/self.timestep} Hz")
    
    def init_motors(self):
        """初始化所有馬達"""
        leg_mapping = {
            1: ('R0', '右前腿'), 2: ('R1', '右中腿'), 3: ('R2', '右後腿'),
            4: ('L2', '左後腿'), 5: ('L1', '左中腿'), 6: ('L0', '左前腿')
        }
        
        joint_names = ['0', '1', '2']
        joint_descriptions = ['髖關節', '膝關節', '踝關節']
        
        self.motors = {}
        
        print("=== 馬達初始化 ===")
        for leg_idx in range(1, self.NUM_LEGS + 1):
            if leg_idx not in leg_mapping:
                continue
                
            leg_name, leg_desc = leg_mapping[leg_idx]
            self.motors[leg_idx] = {}
            
            for j, joint_name in enumerate(joint_names):
                joint_idx = j + 1
                motor_name = f"{leg_name}{joint_name}"
                joint_desc = joint_descriptions[j]
                
                try:
                    motor = self.robot.getDevice(motor_name)
                    if motor is None:
                        print(f"  ⚠️  找不到馬達 {motor_name}")
                        continue
                    
                    motor.setPosition(0.0)
                    motor.setVelocity(motor.getMaxVelocity())
                    
                    self.motors[leg_idx][joint_idx] = motor
                    print(f"  ✓  腿{leg_idx} {joint_desc} -> {motor_name}")
                    
                except Exception as e:
                    print(f"  ❌ 初始化馬達 {motor_name} 失敗: {e}")
    
    def init_sensors(self):
        """初始化感測器"""
        # IMU感測器
        try:
            self.imu_device = self.robot.getDevice("inertialunit1")
            if self.imu_device is None:
                print("❌ 找不到IMU感測器")
                return
            self.imu_device.enable(self.timestep)
            print("✅ IMU感測器已啟用")
        except Exception as e:
            print(f"❌ IMU初始化失敗: {e}")
            self.imu_device = None
        
        # GPS感測器（用於位置追蹤）
        try:
            self.gps_device = self.robot.getDevice("gps")
            if self.gps_device is None:
                print("⚠️ 未找到GPS感測器，將使用Supervisor節點")
                self.gps_device = None
            else:
                self.gps_device.enable(self.timestep)
                print("✅ GPS感測器已啟用")
        except Exception as e:
            print(f"❌ GPS初始化失敗: {e}")
            self.gps_device = None
    
    def init_platform_control(self):
        """初始化實驗平台控制"""
        try:
            # 獲取experimental_platform節點
            self.platform_node = self.robot.getFromDef("experimental_platform")
            if self.platform_node is None:
                print("❌ 找不到experimental_platform節點")
                return
            
            # 獲取platform_motor
            children_field = self.platform_node.getField("children")
            children_count = children_field.getCount()
            
            self.platform_motor_joint = None
            for i in range(children_count):
                child = children_field.getMFNode(i)
                if child is not None and child.getDef() == "platform_motor":
                    self.platform_motor_joint = child
                    break
            
            if self.platform_motor_joint is None:
                print("❌ 找不到platform_motor")
                return
            
            # 獲取position控制接口
            joint_params_field = self.platform_motor_joint.getField("jointParameters")
            joint_params_node = joint_params_field.getSFNode()
            self.platform_position_field = joint_params_node.getField("position")
            
            print("✅ 實驗平台控制已初始化")
            
        except Exception as e:
            print(f"❌ 平台控制初始化失敗: {e}")
            self.platform_position_field = None
    
    def _initialize_buffers(self):
        """初始化序列緩存"""
        seq_len = self.transformer_config['sequence_length']
        for _ in range(seq_len):
            self.state_buffer.append(np.zeros(6))
            self.action_buffer.append(np.zeros(6))
            self.reward_buffer.append(np.array([0.0]))
    
    def create_output_directories(self):
        """建立輸出目錄"""
        self.output_dir = "experimental_platform_training"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def control_platform(self):
        """控制實驗平台運動"""
        if self.platform_position_field is None:
            return
        
        current_time = self.robot.getTime()
        # 正弦波運動：0.2 * sin(π * t)
        sine_angle = 0.2 * math.sin(1.0 * math.pi * current_time)
        self.platform_position_field.setSFFloat(sine_angle)
    
    def get_robot_position(self):
        """獲取機器人位置"""
        try:
            if self.gps_device is not None:
                return np.array(self.gps_device.getValues())
            else:
                # 使用Supervisor獲取位置
                robot_node = self.robot.getSelf()
                if robot_node:
                    position = robot_node.getPosition()
                    return np.array(position)
                return np.array([0, 0, 0])
        except:
            return np.array([0, 0, 0])
    
    def get_imu_data(self):
        """獲取IMU數據並轉換為6維腳部方向分量"""
        if self.imu_device is None:
            return np.zeros(6)
        
        try:
            roll_pitch_yaw = self.imu_device.getRollPitchYaw()
            pitch, roll, yaw = roll_pitch_yaw[1], roll_pitch_yaw[0], roll_pitch_yaw[2]
            
            # 計算6維腳部方向分量
            sqrt_half = math.sqrt(0.5)
            e1 = (pitch + roll) * sqrt_half  # 前右腳
            e2 = roll                        # 右中腳
            e3 = (-pitch + roll) * sqrt_half # 後右腳
            e4 = (-pitch - roll) * sqrt_half # 後左腳
            e5 = -roll                       # 左中腳
            e6 = (pitch - roll) * sqrt_half  # 前左腳
            
            return np.array([e1, e2, e3, e4, e5, e6])
            
        except Exception as e:
            print(f"IMU讀取錯誤: {e}")
            return np.zeros(6)
    
    def get_raw_imu_data(self):
        """獲取原始IMU數據（用於獎勵計算）"""
        if self.imu_device is None:
            return np.array([0, 0, 0])
        
        try:
            roll_pitch_yaw = self.imu_device.getRollPitchYaw()
            return np.array([roll_pitch_yaw[0], roll_pitch_yaw[1], roll_pitch_yaw[2]])  # roll, pitch, yaw
        except:
            return np.array([0, 0, 0])
    
    def calculate_reward(self, raw_imu_data):
        """計算獎勵函數"""
        roll, pitch, yaw = raw_imu_data
        
        # 穩定性獎勵
        stability_reward = math.exp(-((abs(pitch) + abs(roll)) / 2) ** 2 / (0.1 ** 2))
        
        # 跌倒懲罰
        fall_penalty = 0
        if abs(pitch) >= 0.524 or abs(roll) >= 0.524:
            fall_penalty = -1
        
        # 位置懲罰（檢查是否超出平台範圍）
        position = self.get_robot_position()
        position_penalty = 0
        if not (-0.2 < position[0] < 0.2 and -0.18 < position[1] < 0.18):
            position_penalty = -1
        
        total_reward = stability_reward + fall_penalty + position_penalty
        return total_reward
    
    def check_termination_conditions(self, raw_imu_data):
        """檢查終止條件"""
        roll, pitch, yaw = raw_imu_data
        
        # 角度過大
        if abs(pitch) > 0.524 or abs(roll) > 0.524:
            self.episode_terminated = True
            self.termination_reason = "角度過大"
            return True
        
        # 超出平台範圍
        position = self.get_robot_position()
        if not (-0.2 < position[0] < 0.2 and -0.18 < position[1] < 0.18):
            self.episode_terminated = True
            self.termination_reason = "超出平台"
            return True
        
        # 達到最大步數
        if self.current_step >= self.MAX_STEPS:
            self.episode_terminated = True
            self.termination_reason = "達到最大步數"
            return True
        
        return False
    
    def get_transformer_action(self):
        """獲取Transformer動作"""
        if len(self.state_buffer) < self.transformer_config['sequence_length']:
            return np.zeros(6)
        
        try:
            # 準備序列數據
            state_seq = torch.FloatTensor(np.array(list(self.state_buffer))).unsqueeze(0).to(self.device)
            action_seq = torch.FloatTensor(np.array(list(self.action_buffer))).unsqueeze(0).to(self.device)
            reward_seq = torch.FloatTensor(np.array(list(self.reward_buffer))).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, logprob, entropy, value = self.agent.get_action_and_value(
                    state_seq, action_seq, reward_seq
                )
            
            # 儲存訓練數據
            self.episode_states.append((state_seq.cpu(), action_seq.cpu(), reward_seq.cpu()))
            self.episode_actions.append(action.cpu())
            self.episode_logprobs.append(logprob.cpu())
            self.episode_values.append(value.cpu())
            
            return action.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Transformer推理錯誤: {e}")
            return np.zeros(6)
    
    def apply_fixed_angles_with_corrections(self, corrections):
        """應用固定角度和Transformer修正"""
        # 基礎固定角度
        base_angles = {
            'hip': 0.0,      # 髖關節固定為0
            'knee': 0.5,     # 膝關節基礎角度
            'ankle': 0.5     # 踝關節基礎角度
        }
        
        for leg_idx in range(1, self.NUM_LEGS + 1):
            if leg_idx not in self.motors:
                continue
            
            # 確定左右腿的方向係數
            direction = 1 if leg_idx <= 3 else -1  # 右腿為正，左腿為負
            
            for joint_idx in range(1, 4):  # 1:髖, 2:膝, 3:踝
                if joint_idx not in self.motors[leg_idx]:
                    continue
                
                if joint_idx == 1:  # 髖關節
                    motor_angle = base_angles['hip']
                elif joint_idx == 2:  # 膝關節
                    motor_angle = base_angles['knee'] * direction
                    # 加入Transformer修正
                    correction_idx = leg_idx - 1
                    if correction_idx < len(corrections):
                        motor_angle += corrections[correction_idx]
                else:  # 踝關節
                    motor_angle = base_angles['ankle'] * direction
                
                # 發送到馬達
                try:
                    if self.current_step >= self.control_start_step:
                        self.motors[leg_idx][joint_idx].setPosition(motor_angle)
                except Exception as e:
                    print(f"馬達控制錯誤 (腿{leg_idx}, 關節{joint_idx}): {e}")
    
    def save_training_checkpoint(self):
        """保存訓練檢查點"""
        checkpoint = {
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'config': self.transformer_config
        }
        
        checkpoint_path = os.path.join(self.models_dir, f"checkpoint_episode_{self.episode_count}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if hasattr(self, 'episode_total_reward') and self.episode_total_reward > self.best_reward:
            self.best_reward = self.episode_total_reward
            best_model_path = os.path.join(self.models_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
    
    def load_training_checkpoint(self, checkpoint_path):
        """載入訓練檢查點"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_count = checkpoint['episode']
            self.total_steps = checkpoint['total_steps']
            self.best_reward = checkpoint['best_reward']
            print(f"✅ 載入檢查點：Episode {self.episode_count}")
        except Exception as e:
            print(f"❌ 載入檢查點失敗: {e}")
    
    def reset_episode(self):
        """重置episode"""
        print(f"\n=== Episode {self.episode_count} 結束 ===")
        if hasattr(self, 'episode_total_reward'):
            print(f"總獎勵: {self.episode_total_reward:.4f}")
        if self.termination_reason:
            print(f"終止原因: {self.termination_reason}")
        print(f"步數: {self.current_step}")
        
        # 執行PPO訓練
        if len(self.episode_states) > 0:
            self.train_ppo()
        
        # 重載環境
        try:
            self.robot.worldReload()
            time.sleep(0.1)  # 等待重載完成
        except Exception as e:
            print(f"環境重載失敗: {e}")
        
        # 重置狀態
        self.current_step = 0
        self.episode_terminated = False
        self.termination_reason = None
        self.episode_count += 1
        
        # 清空episode數據
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_logprobs.clear()
        self.episode_values.clear()
        self.episode_dones.clear()
        
        # 重新初始化緩存
        self._initialize_buffers()
        
        print(f"開始 Episode {self.episode_count}")
    
    def train_ppo(self):
        """執行PPO訓練"""
        if len(self.episode_states) < 10:  # 需要足夠的數據
            return
        
        try:
            # 轉換數據格式
            states = self.episode_states
            actions = self.episode_actions
            rewards = torch.tensor(self.episode_rewards)
            logprobs = torch.stack(self.episode_logprobs)
            values = torch.stack(self.episode_values).squeeze()
            dones = torch.tensor(self.episode_dones)
            
            # 執行PPO更新
            loss_info = self.trainer.update(states, actions, rewards, logprobs, values, dones)
            
            print(f"PPO更新 - PG損失: {loss_info['pg_loss']:.4f}, "
                  f"V損失: {loss_info['v_loss']:.4f}, "
                  f"熵損失: {loss_info['entropy_loss']:.4f}")
            
        except Exception as e:
            print(f"PPO訓練錯誤: {e}")
    
    def run(self):
        """主運行循環"""
        print("開始PPO訓練...")
        
        # 檢查是否有現有檢查點
        latest_checkpoint = os.path.join(self.models_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_checkpoint):
            self.load_training_checkpoint(latest_checkpoint)
        
        self.episode_total_reward = 0
        
        while self.robot.step(self.timestep) != -1:
            # 控制實驗平台
            self.control_platform()
            
            # 獲取感測器數據
            imu_data = self.get_imu_data()
            raw_imu_data = self.get_raw_imu_data()
            
            # 更新狀態緩存
            self.state_buffer.append(imu_data)
            
            # 檢查終止條件
            if self.check_termination_conditions(raw_imu_data):
                self.episode_dones.append(1.0)
                self.reset_episode()
                continue
            else:
                self.episode_dones.append(0.0)
            
            # 獲取Transformer動作
            transformer_corrections = self.get_transformer_action()
            
            # 更新動作緩存
            self.action_buffer.append(transformer_corrections)
            
            # 計算獎勵
            current_reward = self.calculate_reward(raw_imu_data)
            self.reward_buffer.append(np.array([current_reward]))
            self.episode_rewards.append(current_reward)
            self.episode_total_reward += current_reward
            
            # 應用控制指令
            self.apply_fixed_angles_with_corrections(transformer_corrections)
            
            # 更新計數器
            self.current_step += 1
            self.total_steps += 1
            
            # 定期輸出訓練信息
            if self.current_step % 200 == 0:
                print(f"Episode {self.episode_count}, Step {self.current_step}, "
                      f"當前獎勵: {current_reward:.4f}, "
                      f"累積獎勵: {self.episode_total_reward:.4f}")
                print(f"IMU: Roll={raw_imu_data[0]:.3f}, Pitch={raw_imu_data[1]:.3f}")
                print(f"修正量範圍: {np.min(transformer_corrections):.3f} ~ {np.max(transformer_corrections):.3f}")
            
            # 定期保存檢查點
            if self.total_steps % 2000 == 0:
                self.save_training_checkpoint()
                print(f"已保存檢查點 (總步數: {self.total_steps})")

# 主程序入口
if __name__ == "__main__":
    try:
        controller = HexapodExperimentalController()
        controller.run()
    except KeyboardInterrupt:
        print("\n訓練被用戶中斷")
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序結束")