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
    def __init__(self, agent, config, device):
        self.agent = agent
        self.config = config
        self.device = device  # ✅ 添加device屬性
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
        """計算GAE優勢函數 - 修正設備分配問題"""
        # ✅ 確保所有張量都在同一設備上
        device = rewards.device
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = torch.tensor(0.0).to(device)
        
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
        """PPO更新 - 修正設備分配問題"""
        # ✅ 確保所有張量都在正確設備上
        next_value = torch.zeros(1).to(self.device)
        
        # 將輸入數據移動到正確設備
        if isinstance(rewards, np.ndarray):
            rewards = torch.FloatTensor(rewards).to(self.device)
        else:
            rewards = rewards.to(self.device)
            
        if isinstance(logprobs, np.ndarray):
            logprobs = torch.FloatTensor(logprobs).to(self.device)
        else:
            logprobs = logprobs.to(self.device)
            
        if isinstance(values, np.ndarray):
            values = torch.FloatTensor(values).to(self.device)
        else:
            values = values.to(self.device)
            
        if isinstance(dones, np.ndarray):
            dones = torch.FloatTensor(dones).to(self.device)
        else:
            dones = dones.to(self.device)
        
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
                
                # ✅ 準備序列數據並確保在正確設備上
                state_seq = torch.stack([s[0].squeeze(0) for s in mb_states]).to(self.device)
                action_seq = torch.stack([s[1].squeeze(0) for s in mb_states]).to(self.device)
                reward_seq = torch.stack([s[2].squeeze(0) for s in mb_states]).to(self.device)
                mb_actions_tensor = torch.stack([a for a in mb_actions]).to(self.device)
                
                # 前向傳播
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    state_seq, action_seq, reward_seq, mb_actions_tensor
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
        
        # ✅ 控制參數 - 參考controller2
        self.body_height_offset = 0.5
        self.control_start_step = 100
        self.knee_clamp_positive = True       # 膝關節限制為正值
        self.use_knee_signal_for_ankle = True # 踝關節使用膝關節信號
        
        # ✅ 初始化處理過的訊號記錄
        self.processed_signals = {}
        for leg_idx in range(1, self.NUM_LEGS + 1):
            self.processed_signals[leg_idx] = {}
            for joint_idx in range(1, 4):  # 1:髖, 2:膝, 3:踝
                self.processed_signals[leg_idx][joint_idx] = np.zeros(self.MAX_STEPS + 1)
        
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
        self.trainer = PPOTrainer(self.agent, self.transformer_config, self.device)  # ✅ 傳遞device
        
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
        """獲取Transformer動作 - 修正設備分配問題"""
        if len(self.state_buffer) < self.transformer_config['sequence_length']:
            return np.zeros(6)
        
        try:
            # ✅ 確保所有數據都在正確設備上
            state_seq = torch.FloatTensor(np.array(list(self.state_buffer))).unsqueeze(0).to(self.device)
            action_seq = torch.FloatTensor(np.array(list(self.action_buffer))).unsqueeze(0).to(self.device)
            reward_seq = torch.FloatTensor(np.array(list(self.reward_buffer))).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, logprob, entropy, value = self.agent.get_action_and_value(
                    state_seq, action_seq, reward_seq
                )
            
            # ✅ 確保儲存的訓練數據在CPU上（節省GPU內存）
            self.episode_states.append((state_seq.cpu(), action_seq.cpu(), reward_seq.cpu()))
            self.episode_actions.append(action.cpu())
            self.episode_logprobs.append(logprob.cpu())
            self.episode_values.append(value.cpu())
            
            return action.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Transformer推理錯誤: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(6)
    
    def process_special_joints(self, motor_angle, leg_idx, joint_idx):
        """處理特殊關節 - 參考controller2"""
        # 膝關節處理：確保正值（站立姿態）
        if joint_idx == 2:
            if self.knee_clamp_positive and motor_angle <= 0:
                return 0.0
        
        # 踝關節特殊處理：特定腿部的負值限制
        elif joint_idx == 3:
            if not self.use_knee_signal_for_ankle:
                if (leg_idx == 2 or leg_idx == 5) and motor_angle >= 0:  # R1, L1中腿
                    return 0.0
        
        return motor_angle
    
    def adjust_signal_direction(self, motor_angle, leg_idx, joint_idx):
        """調整訊號方向 - 參考controller2
        
        反向規則：
        - 右側腿部(leg_idx 1-3): 踝關節反向
        - 左側腿部(leg_idx 4-6): 髖關節和膝關節反向  
        - 額外踝關節反向: R0(leg_idx=1), L0(leg_idx=6), R1(leg_idx=2), L1(leg_idx=5)
        """
        
        # 右側腿部踝關節反向（當不使用膝關節信號控制踝關節時）
        if not self.use_knee_signal_for_ankle:
            if leg_idx <= 3 and joint_idx == 3:
                motor_angle = -motor_angle
        
        # 左側腿部髖關節和膝關節反向
        if leg_idx >= 4 and (joint_idx == 1 or joint_idx == 2):
            motor_angle = -motor_angle
        
        # 額外的踝關節反向（當不使用膝關節信號控制踝關節時）
        if not self.use_knee_signal_for_ankle:
            if (leg_idx == 1 or leg_idx == 6) and joint_idx == 3:  # R0, L0前腿
                motor_angle = -motor_angle
            if (leg_idx == 2 or leg_idx == 5) and joint_idx == 3:  # R1, L1中腿
                motor_angle = -motor_angle

        return motor_angle
    
    def replace_ankle_with_knee_signal(self, motor_angle, leg_idx, joint_idx):
        """將踝關節訊號替換為膝關節訊號 - 參考controller2"""
        if joint_idx == 3 and self.use_knee_signal_for_ankle:
            # 使用同隻腳膝關節的處理過訊號
            if leg_idx in self.processed_signals and 2 in self.processed_signals[leg_idx]:
                knee_signal = self.processed_signals[leg_idx][2][self.current_step]
                return knee_signal * 1  # 可以調整係數
            else:
                # 如果沒有膝關節信號，使用基礎踝關節角度
                return motor_angle
        return motor_angle
    
    def apply_height_offset(self, motor_angle, leg_idx, joint_idx):
        """應用機身高度偏移 - 參考controller2"""
        # 只對膝關節，以及在不使用膝關節信號控制踝關節時的踝關節應用偏移
        should_apply_offset = (
            joint_idx == 2 or  # 膝關節
            (joint_idx == 3 and not self.use_knee_signal_for_ankle)  # 踝關節(條件性)
        )
    
        if should_apply_offset:
            # 右側腿部(1-3)用負偏移，左側腿部(4-6)用正偏移
            offset_direction = -1 if leg_idx <= 3 else 1
            return motor_angle + (self.body_height_offset * offset_direction)
        
        return motor_angle
    
    def apply_fixed_angles_with_corrections(self, corrections):
        """應用固定角度和Transformer修正 - 使用controller2的處理流程"""
        # 基礎固定角度
        base_angles = {
            'hip': 0.0,      # 髖關節固定為0
            'knee': 0.0,     # 膝關節基礎角度
            'ankle': 0.0     # 踝關節基礎角度
        }
        
        # 初始化處理過的訊號記錄（如果不存在）
        if not hasattr(self, 'processed_signals'):
            self.processed_signals = {}
            for leg_idx in range(1, self.NUM_LEGS + 1):
                self.processed_signals[leg_idx] = {}
                for joint_idx in range(1, 4):
                    self.processed_signals[leg_idx][joint_idx] = np.zeros(self.MAX_STEPS + 1)
        
        for leg_idx in range(1, self.NUM_LEGS + 1):
            if leg_idx not in self.motors:
                continue
            
            for joint_idx in range(1, 4):  # 1:髖, 2:膝, 3:踝
                if joint_idx not in self.motors[leg_idx]:
                    continue
                
                # 步驟1: 獲取基礎角度
                if joint_idx == 1:  # 髖關節
                    motor_angle = base_angles['hip']
                elif joint_idx == 2:  # 膝關節
                    motor_angle = base_angles['knee']
                else:  # 踝關節
                    motor_angle = base_angles['ankle']
                
                # 步驟2: 踝關節訊號替換（如果啟用）
                motor_angle = self.replace_ankle_with_knee_signal(motor_angle, leg_idx, joint_idx)
                
                # 步驟3: 特殊關節處理
                motor_angle = self.process_special_joints(motor_angle, leg_idx, joint_idx)
                
                # 步驟4: 訊號方向調整（馬達轉動方向相關）
                motor_angle = self.adjust_signal_direction(motor_angle, leg_idx, joint_idx)
                
                # 步驟5: 應用機身高度偏移
                motor_angle = self.apply_height_offset(motor_angle, leg_idx, joint_idx)
                
                # 步驟6: 添加Transformer修正量（只對膝關節）
                if joint_idx == 2:  # 只有膝關節接受Transformer修正
                    correction_idx = leg_idx - 1  # 6個膝關節對應6個修正量
                    if correction_idx < len(corrections):
                        motor_angle += corrections[correction_idx]
                        if self.current_step % 200 == 0:
                            print(f"腿{leg_idx}膝關節修正量: {corrections[correction_idx]:.4f}")
                
                # 步驟7: 儲存處理過的訊號
                self.processed_signals[leg_idx][joint_idx][self.current_step] = motor_angle
                
                # 步驟8: 發送到馬達
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
    
    def save_episode_data(self):
        """保存episode訓練數據到檔案（重置前）"""
        episode_data = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'episode_total_reward': getattr(self, 'episode_total_reward', 0),
            'termination_reason': self.termination_reason,
            'current_step': self.current_step,
            'best_reward': self.best_reward,
            # 深拷貝訓練數據以避免引用問題
            'states': [s for s in self.episode_states] if self.episode_states else [],
            'actions': [a.clone() if hasattr(a, 'clone') else a for a in self.episode_actions],
            'rewards': self.episode_rewards.copy() if self.episode_rewards else [],
            'logprobs': [lp.clone() if hasattr(lp, 'clone') else lp for lp in self.episode_logprobs],
            'values': [v.clone() if hasattr(v, 'clone') else v for v in self.episode_values],
            'dones': self.episode_dones.copy() if self.episode_dones else [],
            # 保存緩存狀態
            'state_buffer': list(self.state_buffer),
            'action_buffer': list(self.action_buffer),
            'reward_buffer': list(self.reward_buffer)
        }
        
        # 保存到檔案
        episode_file = os.path.join(self.logs_dir, "pending_episode_data.pkl")
        try:
            with open(episode_file, 'wb') as f:
                pickle.dump(episode_data, f)
            print(f"✅ Episode數據已保存到檔案")
        except Exception as e:
            print(f"❌ 保存episode數據失敗: {e}")
    
    def load_and_process_episode_data(self):
        """載入並處理episode訓練數據（重置後立即調用）"""
        episode_file = os.path.join(self.logs_dir, "pending_episode_data.pkl")
        
        if not os.path.exists(episode_file):
            return False
        
        try:
            with open(episode_file, 'rb') as f:
                episode_data = pickle.load(f)
            
            print(f"✅ 載入待處理Episode數據")
            print(f"   Episode: {episode_data.get('episode_count', 0)}")
            print(f"   總獎勵: {episode_data.get('episode_total_reward', 0):.4f}")
            print(f"   終止原因: {episode_data.get('termination_reason', '未知')}")
            print(f"   步數: {episode_data.get('current_step', 0)}")
            
            # 恢復訓練統計（但不是當前episode狀態）
            self.total_steps = episode_data.get('total_steps', 0)
            self.best_reward = episode_data.get('best_reward', -float('inf'))
            
            # 執行PPO訓練（如果有足夠數據）
            states = episode_data.get('states', [])
            actions = episode_data.get('actions', [])
            rewards = episode_data.get('rewards', [])
            logprobs = episode_data.get('logprobs', [])
            values = episode_data.get('values', [])
            dones = episode_data.get('dones', [])
            
            if len(states) >= 10:  # 需要足夠的數據進行訓練
                print("執行延遲的PPO訓練...")
                try:
                    # 轉換數據格式
                    rewards_tensor = torch.tensor(rewards)
                    logprobs_tensor = torch.stack(logprobs) if logprobs else torch.tensor([])
                    values_tensor = torch.stack(values).squeeze() if values else torch.tensor([])
                    dones_tensor = torch.tensor(dones)
                    
                    # 執行PPO更新
                    loss_info = self.trainer.update(states, actions, rewards_tensor, 
                                                  logprobs_tensor, values_tensor, dones_tensor)
                    
                    print(f"PPO更新完成 - PG損失: {loss_info['pg_loss']:.4f}, "
                          f"V損失: {loss_info['v_loss']:.4f}, "
                          f"熵損失: {loss_info['entropy_loss']:.4f}")
                    
                    # 更新最佳獎勵
                    episode_reward = episode_data.get('episode_total_reward', 0)
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                        self.save_training_checkpoint()  # 保存最佳模型
                    
                except Exception as e:
                    print(f"❌ PPO訓練失敗: {e}")
            else:
                print(f"⚠️ 數據不足，跳過PPO訓練 (數據量: {len(states)})")
            
            # 清理episode數據檔案
            os.remove(episode_file)
            print("✅ Episode數據檔案已清理")
            
            return True
            
        except Exception as e:
            print(f"❌ 載入episode數據失敗: {e}")
            # 嘗試清理損壞的檔案
            try:
                os.remove(episode_file)
                print("🗑️ 已清理損壞的數據檔案")
            except:
                pass
            return False
    
    def reset_episode(self):
        """重置episode - 使用simulationReset()避免控制器中斷"""
        print(f"\n=== Episode {self.episode_count} 結束 ===")
        if hasattr(self, 'episode_total_reward'):
            print(f"總獎勵: {self.episode_total_reward:.4f}")
        if self.termination_reason:
            print(f"終止原因: {self.termination_reason}")
        print(f"步數: {self.current_step}")
        
        # ✅ 先執行PPO訓練（在重置前）
        if len(self.episode_states) >= 10:  # 確保有足夠數據
            print("執行PPO訓練...")
            self.train_ppo()
        else:
            print(f"⚠️ 數據不足，跳過PPO訓練 (數據量: {len(self.episode_states)})")
        
        # 保存最佳模型
        if hasattr(self, 'episode_total_reward') and self.episode_total_reward > self.best_reward:
            self.best_reward = self.episode_total_reward
            self.save_training_checkpoint()
            print(f"🏆 新的最佳獎勵: {self.best_reward:.4f}")
        
        # ✅ 使用simulationReset()重置環境
        print("重置模擬環境...")
        try:
            self.robot.simulationReset()
            print("✅ 模擬環境重置成功")
        except Exception as e:
            print(f"❌ 模擬環境重置失敗: {e}")
        
        # 更新episode計數
        self.episode_count += 1
        
        # 重置當前episode狀態
        self.current_step = 0
        self.episode_terminated = False
        self.termination_reason = None
        self.episode_total_reward = 0  # 重置當前episode獎勵
        
        # 清空當前episode數據
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_logprobs.clear()
        self.episode_values.clear()
        self.episode_dones.clear()
        
        # 重新初始化緩存
        self._initialize_buffers()
        
        print(f"開始 Episode {self.episode_count}")
        
        # 定期保存檢查點
        if self.episode_count % 10 == 0:
            self.save_training_checkpoint()
            print(f"📁 已保存檢查點 (Episode {self.episode_count})")
    
    def train_ppo(self):
        """執行PPO訓練 - 修正設備分配問題"""
        if len(self.episode_states) < 10:  # 需要足夠的數據
            print(f"⚠️ 訓練數據不足: {len(self.episode_states)}")
            return
        
        try:
            # ✅ 轉換數據格式並確保設備一致性
            states = self.episode_states
            actions = self.episode_actions
            rewards = torch.FloatTensor(self.episode_rewards).to(self.device)
            
            # 處理logprobs和values
            if len(self.episode_logprobs) > 0:
                logprobs = torch.stack([lp.to(self.device) for lp in self.episode_logprobs])
            else:
                logprobs = torch.zeros(len(states)).to(self.device)
            
            if len(self.episode_values) > 0:
                values = torch.stack([v.to(self.device) for v in self.episode_values]).squeeze()
            else:
                values = torch.zeros(len(states)).to(self.device)
            
            # ✅ 確保dones維度與其他張量匹配
            dones_list = self.episode_dones
            if len(dones_list) > len(states):
                dones_list = dones_list[:len(states)]  # 截斷到正確長度
            elif len(dones_list) < len(states):
                # 補齊到正確長度（用0填充）
                dones_list.extend([0.0] * (len(states) - len(dones_list)))
            
            dones = torch.FloatTensor(dones_list).to(self.device)
            
            # 確保維度一致
            if len(values.shape) == 0:
                values = values.unsqueeze(0)
            if len(logprobs.shape) == 0:
                logprobs = logprobs.unsqueeze(0)
            
            # ✅ 確保所有張量長度一致
            min_length = min(len(states), len(actions), rewards.shape[0], 
                           logprobs.shape[0], values.shape[0], dones.shape[0])
            
            if min_length < len(states):
                states = states[:min_length]
                actions = actions[:min_length]
                rewards = rewards[:min_length]
                logprobs = logprobs[:min_length]
                values = values[:min_length]
                dones = dones[:min_length]
            
            print(f"訓練數據維度檢查:")
            print(f"  states: {len(states)}")
            print(f"  actions: {len(actions)}")
            print(f"  rewards: {rewards.shape}, device: {rewards.device}")
            print(f"  logprobs: {logprobs.shape}, device: {logprobs.device}")
            print(f"  values: {values.shape}, device: {values.device}")
            print(f"  dones: {dones.shape}, device: {dones.device}")
            
            # 執行PPO更新
            loss_info = self.trainer.update(states, actions, rewards, logprobs, values, dones)
            
            print(f"PPO更新完成 - PG損失: {loss_info['pg_loss']:.4f}, "
                  f"V損失: {loss_info['v_loss']:.4f}, "
                  f"熵損失: {loss_info['entropy_loss']:.4f}")
            
        except Exception as e:
            print(f"PPO訓練錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """主運行循環"""
        print("開始PPO訓練...")
        
        # 檢查是否有現有檢查點
        latest_checkpoint = os.path.join(self.models_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_checkpoint):
            self.load_training_checkpoint(latest_checkpoint)
        
        # 初始化episode獎勵
        self.episode_total_reward = 0
        
        print(f"當前Episode: {self.episode_count}")
        
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
                # ✅ 在reset之前先完成當前步的數據收集
                
                # 獲取Transformer動作（即使終止也要完整收集這一步的數據）
                transformer_corrections = self.get_transformer_action()
                self.action_buffer.append(transformer_corrections)
                
                # 計算最終獎勵
                current_reward = self.calculate_reward(raw_imu_data)
                self.reward_buffer.append(np.array([current_reward]))
                self.episode_rewards.append(current_reward)
                self.episode_total_reward += current_reward
                
                # 標記為終止
                self.episode_dones.append(1.0)
                
                # 應用控制指令（最後一次）
                self.apply_fixed_angles_with_corrections(transformer_corrections)
                
                # 然後重置episode
                self.reset_episode()
                continue
            
            # 非終止情況的正常流程
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