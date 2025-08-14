#!/usr/bin/env python3
"""
六足機器人 PPO + Transformer Webots 控制器
專門用於訓練 - 簡化版本
"""

import sys
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, Union, List
import json
from dataclasses import dataclass, field
from collections import deque

# Webots imports
from controller import Supervisor

# TensorBoard import
from torch.utils.tensorboard import SummaryWriter

try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import LinearSchedule
    from stable_baselines3.common.env_checker import check_env
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    print(f'請安裝必要套件: pip install stable-baselines3[extra] gymnasium')
    sys.exit(1)

@dataclass
class HexapodConfig:
    """六足機器人訓練配置 - 統一參數管理"""
    
    # === 環境參數 ===
    max_episode_steps: int = 2048
    sequence_length: int = 100
    control_start_step: int = 100
    
    # === CPG 參數 ===
    knee_clamp_positive: bool = True
    use_knee_signal_for_ankle: bool = True
    body_height_offset: float = 0.5
    
    # === 獎勵權重 ===
    w_s: float = 1.0  # 穩定性獎勵權重
    w_c: float = 0.05  # 控制量獎勵權重
    
    # === Transformer 架構參數 ===
    transformer_features_dim: int = 6
    transformer_n_heads: int = 2
    transformer_n_layers: int = 3
    transformer_dropout: float = 0.1
    
    # === PPO 超參數 ===
    use_linear_learning_rate_decay: bool = False
    learning_rate_start: float = 3e-4
    learning_rate_end: float = 1e-5
    #fixed_learning_rate: float = 2.0633e-05
    fixed_learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 1024
    n_epochs: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # === 訓練參數 ===
    total_timesteps: int = 100000000
    save_frequency_ratio: float = 0.05  # 總步數的比例
    random_seed: int = 42
    
    # === 系統參數 ===
    tensorboard_log_dir: str = "./tensorboard_logs"
    model_save_dir: str = "./models"

    # === 獎勵參數 ===
    imbalance_threshold: float = 0.05  # 非平衡閾值 (roll/pitch abs > 此值)
    response_bonus: float = 0.1  # 每隻正確回應腳的獎勵
    response_penalty: float = -0.05  # 每隻未回應腳的懲罰

    # === action噪音參數 ===
    noise_probability: float = 0.05 # 加入噪音的機率，0.05=5%
    noise_std: float = 0.1 # 加入的噪音值的高斯分布的寬(標準差)
    # === 計算屬性方法 ===
    def get_save_frequency(self) -> int:
        """計算儲存頻率"""
        return int(self.total_timesteps * self.save_frequency_ratio)
    
    def get_net_arch(self) -> Dict[str, List[int]]:
        """獲取網路架構"""
        return dict(
            pi=[self.transformer_features_dim, self.transformer_features_dim], 
            vf=[self.transformer_features_dim, self.transformer_features_dim]
        )
    
    def get_transformer_kwargs(self) -> Dict[str, Any]:
        """獲取 Transformer 參數字典"""
        return {
            'features_dim': self.transformer_features_dim,
            'n_heads': self.transformer_n_heads,
            'n_layers': self.transformer_n_layers,
            'sequence_length': self.sequence_length,
            'dropout': self.transformer_dropout
        }
    def get_learning_rate(self):
        """獲取學習率設置"""
        if self.use_linear_learning_rate_decay:
            return LinearSchedule(
                start=self.learning_rate_start,
                end=self.learning_rate_end,
                end_fraction=1.0
            )
        else:
            return self.fixed_learning_rate



class PositionalEncoding(nn.Module):
    """位置編碼模組 - 支援 batch_first"""
    
    def __init__(self, d_model: int, max_len: int = 500, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if batch_first:
            # 格式: [1, max_len, d_model] 適用於 [batch_size, seq_len, d_model]
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        else:
            # 格式: [max_len, 1, d_model] 適用於 [seq_len, batch_size, d_model]
            pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor
            - 如果 batch_first=True: [batch_size, seq_len, embedding_dim]
            - 如果 batch_first=False: [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            # x: [batch_size, seq_len, embedding_dim]
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]
        else:
            # x: [seq_len, batch_size, embedding_dim]
            seq_len = x.size(0)
            return x + self.pe[:seq_len, :]


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    自定義 Transformer 特徵提取器
    處理序列數據並輸出策略網路所需的特徵
    """
    
    def __init__(self, observation_space, config: HexapodConfig):
        # 先調用父類初始化
        super().__init__(observation_space, config.transformer_features_dim)
        
        self.sequence_length = config.sequence_length
        self.state_dim = observation_space.shape[0] // config.sequence_length
        self._features_dim = config.transformer_features_dim
        print(f"observation_space.shape[0]={observation_space.shape[0]}\nsequence_length={self.sequence_length}\nstate_dim={self.state_dim}\nobservation_space.shape={observation_space.shape}")
        
        # 輸入投影層
        #self.input_projection = nn.Linear(self.state_dim, config.transformer_features_dim)
        
        # 位置編碼
        self.pos_encoding = PositionalEncoding(config.transformer_features_dim, config.sequence_length)
        
        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_features_dim,
            nhead=config.transformer_n_heads,
            dim_feedforward=config.transformer_features_dim * 2,
            dropout=config.transformer_dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_n_layers)
        
        # 輸出投影層
        #self.output_projection = nn.Linear(config.transformer_features_dim, config.transformer_features_dim)

        
        print(f"🤖 Transformer 特徵提取器初始化:")
        print(f"   📐 序列長度: {self.sequence_length}")
        print(f"   📊 狀態維度: {self.state_dim}")
        print(f"   🧠 特徵維度: {config.transformer_features_dim}")
        print(f"   👁️  注意力頭數: {config.transformer_n_heads}")
        print(f"   🏗️  Transformer層數: {config.transformer_n_layers}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, sequence_length * state_dim]
        Returns:
            features: [batch_size, features_dim]
        """
        # print(f"🔍 TransformerFeaturesExtractor 輸入形狀: {observations.shape}")
        batch_size = observations.shape[0]
        # print(f"📊 批次大小: {batch_size}")

        # 重塑輸入: [batch_size, seq_len * state_dim] -> [batch_size, seq_len, state_dim]
        x = observations.view(batch_size, self.sequence_length, self.state_dim)
        # print(f"🔄 重塑後形狀: {x.shape}")

        # 投影到特徵空間: [batch_size, seq_len, features_dim]
        #x = self.input_projection(x)
        # print(f"📍 輸入投影後形狀: {x.shape}")

        # 加入位置編碼
        x = self.pos_encoding(x)
        
        # 通過 Transformer 編碼器
        x = self.transformer_encoder(x)
        
        # 取最後一個時間步的輸出: [batch_size, features_dim]
        x = x[:, -1, :]
        # print(f"🎯 最終輸出形狀: {x.shape}")
        
        # 最終投影
        #features = self.output_projection(x)
        features = x
        # print(f"✅ 特徵形狀: {features.shape}")
        # print("-" * 50)  # 分隔線
        
        return features


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    自定義 Actor-Critic 策略，使用 Transformer 特徵提取器
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # 移除 config 參數（如果存在），因為我們通過 features_extractor_kwargs 傳遞
        kwargs.pop('config', None)  # 安全移除，避免重複傳遞
        
        # 確保使用我們的 TransformerFeaturesExtractor
        kwargs.setdefault('features_extractor_class', TransformerFeaturesExtractor)
        kwargs.setdefault('features_extractor_kwargs', {})
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )

class HexapodBalanceEnv(Supervisor, gym.Env):
    """
    六足機器人平衡強化學習環境
    直接在 Webots Controller 中運行
    """
    
    def __init__(self, config: HexapodConfig):
        super().__init__()
        
        # 環境參數
        self.max_episode_steps = config.max_episode_steps
        self.sequence_length = config.sequence_length
        self.current_step = 0
        
        # CPG參數
        self.knee_clamp_positive = config.knee_clamp_positive
        self.use_knee_signal_for_ankle = config.use_knee_signal_for_ankle
        self.body_height_offset = config.body_height_offset
        self.control_start_step = config.control_start_step
        
        # 獎勵權重
        self.w_s = config.w_s # 穩定性獎勵權重
        self.w_c = config.w_c # 控制量獎勵權重
        
        # 儲存配置引用（可選）
        self.config = config

        # 隨機平台旋轉相關
        self.random_platform_angle = 0.0  # 當前episode的隨機角度
        # 平台擺動角度
        self.platform_angle=0.0

        self.prev_roll = 0.0  # 上一步 roll
        self.prev_pitch = 0.0  # 上一步 pitch
        self.prev_states = np.zeros(6, dtype=np.float32)  # 上一步 6 分量狀態
        self.imbalance_threshold = config.imbalance_threshold
        self.response_bonus = config.response_bonus
        self.response_penalty = config.response_penalty

        #action噪音相關
        self.noise_probability=config.noise_probability
        self.noise_std=config.noise_std
        
        # Episode統計（用於記錄平均值）
        self.episode_stats = {
            'stability_rewards': [],
            'control_rewards': [],
            'penalties': [],
            'rolls': [],
            'pitches': [],
            'distances_from_origin': []
        }
        
        self.spec = type('SimpleSpec', (), {'id': 'HexapodBalance-v0','max_episode_steps': self.max_episode_steps})()
        
        # 狀態和動作空間定義
        self._setup_spaces()
        
        # Webots設置
        self.timestep = int(self.getBasicTimeStep())
        self.motors = {}
        self.gps_device = None
        self.imu_device = None
        self.platform_motor_joint = None
        
        # 狀態序列緩存（用於Transformer）
        self.state_sequence = deque(maxlen=self.sequence_length)
        
        # 初始化設備
        self._init_devices()
        
        print("✅ 六足機器人平衡環境已初始化（增強版本）")
        print(f"   🎯 穩定性獎勵權重: {self.w_s}")
        print(f"   🎮 控制量獎勵權重: {self.w_c}")

    def _setup_spaces(self):
        """設置狀態和動作空間"""
        # 動作空間：6個膝關節的修正量 [-1, 1]，使用[-1, 1]是官方推薦要對稱以0為中心
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # 狀態空間：序列化的狀態 [sequence_length * state_dim]
        single_state_dim = 6  # 六個腳的方向分量
        sequence_dim = self.sequence_length * single_state_dim  # 50 * 6 = 300
        
        self.observation_space = spaces.Box(
            low=-4.0, 
            high=4.0, 
            shape=(sequence_dim,), 
            dtype=np.float32
        )
        
        # print(f"📊 觀察空間維度: {sequence_dim} (序列長度: {self.sequence_length} × 狀態維度: {single_state_dim})")
        # print(f"🎮 動作空間維度: {self.action_space.shape[0]}")

    def _init_devices(self):
        """初始化Webots設備"""
        try:
            self._init_motors()
            self._init_gps()
            self._init_imu()
            self._init_platform_motor()
        except Exception as e:
            print(f"❌ 設備初始化失敗: {e}")
            raise

    def _init_motors(self):
        """初始化機器人馬達"""
        leg_mapping = {
            1: ('R0', '右前腿'),
            2: ('R1', '右中腿'), 
            3: ('R2', '右後腿'),
            4: ('L2', '左後腿'),
            5: ('L1', '左中腿'),
            6: ('L0', '左前腿')
        }
        
        joint_names = ['0', '1', '2']
        
        for leg_idx in range(1, 7):
            if leg_idx not in leg_mapping:
                continue
                
            leg_name, _ = leg_mapping[leg_idx]
            self.motors[leg_idx] = {}
            
            for j, joint_name in enumerate(joint_names):
                joint_idx = j + 1
                motor_name = f"{leg_name}{joint_name}"
                
                try:
                    motor = self.getDevice(motor_name)
                    if motor is None:
                        continue
                    
                    motor.setPosition(0.0)
                    motor.setVelocity(motor.getMaxVelocity())
                    
                    self.motors[leg_idx][joint_idx] = motor
                    
                except Exception as e:
                    print(f"❌ 初始化馬達 {motor_name} 時發生錯誤: {e}")

    def _init_gps(self):
        """初始化GPS感測器"""
        try:
            self.gps_device = self.getDevice("gps")
            if self.gps_device:
                self.gps_device.enable(self.timestep)
        except Exception as e:
            print(f"❌ GPS初始化失敗: {e}")

    def _init_imu(self):
        """初始化IMU感測器"""
        try:
            self.imu_device = self.getDevice("inertialunit1")
            if self.imu_device:
                self.imu_device.enable(self.timestep)
        except Exception as e:
            print(f"❌ IMU初始化失敗: {e}")

    def _init_platform_motor(self):
        """初始化平台馬達"""
        try:
            platform_node = self.getFromDef("experimental_platform")
            if platform_node is None:
                return
            
            children_field = platform_node.getField("children")
            children_count = children_field.getCount()
            
            for i in range(children_count):
                child = children_field.getMFNode(i)
                if child and child.getDef() == "platform_motor":
                    self.platform_motor_joint = child
                    break
                    
        except Exception as e:
            print(f"❌ 平台馬達初始化失敗: {e}")

    def _set_random_platform_rotation(self):
        """設定隨機平台旋轉角度"""
        try:
            # 生成0-2π之間的隨機角度
            self.random_platform_angle = np.random.uniform(0, 2 * np.pi)
            
            # 獲取 experimental_platform 節點
            platform_node = self.getFromDef("experimental_platform")
            if platform_node:
                # 設定 Solid 的 rotation 字段: [0, 0, 1, random_angle]
                rotation_field = platform_node.getField("rotation")
                if rotation_field:
                    rotation_field.setSFRotation([0, 0, 1, self.random_platform_angle])
                    print(f"🔄 隨機平台旋轉角度: {self.random_platform_angle:.3f} 弧度 ({math.degrees(self.random_platform_angle):.1f}°)")
                        
        except Exception as e:
            print(f"❌ 設定隨機平台旋轉失敗: {e}")

    def _get_imu_data(self):
        """讀取IMU數據"""
        try:
            if self.imu_device:
                roll_pitch_yaw = self.imu_device.getRollPitchYaw()
                return roll_pitch_yaw[0], roll_pitch_yaw[1]
            else:
                return 0.0, 0.0
        except Exception as e:
            return 0.0, 0.0

    def _get_gps_data(self):
        """讀取GPS數據"""
        try:
            if self.gps_device:
                position = self.gps_device.getValues()
                return position[0], position[1], position[2]
            else:
                return 0.0, 0.0, 0.0
        except Exception as e:
            return 0.0, 0.0, 0.0

    def _calculate_single_state(self):
        """計算單步狀態"""
        roll, pitch = self._get_imu_data()

        self.prev_roll = roll
        self.prev_pitch = pitch
        
        sqrt_half = math.sqrt(0.5)
        
        states = np.array([
            (pitch + roll) * sqrt_half,
            roll,
            (-pitch + roll) * sqrt_half,
            (-pitch - roll) * sqrt_half,
            -roll,
            (pitch - roll) * sqrt_half
        ], dtype=np.float32)
        
        return states

    def _get_sequence_observation(self):
        """獲取序列觀察"""
        # 獲取當前序列長度
        current_length = len(self.state_sequence)
        
        if current_length == 0:
            # 如果序列為空，全部填充零
            sequence = np.zeros((self.sequence_length, 6), dtype=np.float32)
        elif current_length < self.sequence_length:
            # 如果序列不夠長，前面用零填充
            padding_length = self.sequence_length - current_length
            padding = np.zeros((padding_length, 6), dtype=np.float32)
            
            # 將現有序列轉為數組
            existing_sequence = np.array(list(self.state_sequence), dtype=np.float32)
            
            # 拼接：[零填充] + [現有序列]
            sequence = np.vstack([padding, existing_sequence])
        else:
            # 序列長度足夠，直接轉換
            sequence = np.array(list(self.state_sequence), dtype=np.float32)
        
        # 展平為一維數組 [sequence_length * 6]
        flattened = sequence.flatten()
        
        # 除錯資訊
        """ if len(self.state_sequence) <= 2:  # 只在開始時印出
            print(f"🔍 序列形狀: {sequence.shape}")
            print(f"📊 展平後形狀: {flattened.shape}")
            print(f"📈 當前序列長度: {len(self.state_sequence)}") """
        
        return flattened

    def _calculate_reward(self, action):
        """計算獎勵函數"""
        roll, pitch = self._get_imu_data()
        x, y, z = self._get_gps_data()
        
        # 穩定性獎勵
        stability_term = (abs(pitch) + abs(roll)) / 2
        r_s = math.exp(-(stability_term ** 2) / (0.1 ** 2))

        # 2. 控制量獎勵，當狀態中大於零的分量對應的腳的控制量也大於零，給予獎勵
        r_c = 0.0
        is_imbalanced = (abs(self.prev_roll) > self.imbalance_threshold) or (abs(self.prev_pitch) > self.imbalance_threshold)
        if is_imbalanced:
            positive_feet = np.where(self.prev_states > 0)[0]  # 分量 >0 的腳索引 (0-5)
            for foot_idx in positive_feet:
                if action[foot_idx] > 0:
                    r_c += self.response_bonus  # 正獎勵
                else:
                    r_c += self.response_penalty  # 懲罰
            if len(positive_feet) > 0:
                r_c /= len(positive_feet)  # 平均化，避免過大
        # 3. 懲罰項

        p = 0
        # 跌倒懲罰
        if abs(pitch) >= 0.785 or abs(roll) >= 0.785:
            p += -1
        
        # 邊界懲罰
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            p += -1
        
        # 總獎勵：加權組合
        total_reward = self.w_s * r_s + self.w_c * r_c + p
        
        return total_reward, r_s, r_c, p

    def _is_done(self):
        """檢查episode是否結束"""
        roll, pitch = self._get_imu_data()
        x, y, z = self._get_gps_data()
        
        if abs(pitch) >= 0.785 or abs(roll) >= 0.785:
            return True, True, "跌倒"
        
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            return True, True, "出界"
        
        if self.current_step >= self.max_episode_steps:
            return False, True, "超時"
        
        return False, False, ""

    def _control_platform(self):
        """控制平台進行正弦波運動"""
        if not self.platform_motor_joint:
            return
        
        try:
            current_time = self.getTime()
            self.platform_angle = 0.2 * math.sin(1.0 * math.pi * current_time)
            
            joint_params_field = self.platform_motor_joint.getField("jointParameters")
            joint_params_node = joint_params_field.getSFNode()
            if joint_params_node:
                position_field = joint_params_node.getField("position")
                if position_field:
                    position_field.setSFFloat(self.platform_angle)
                    
        except Exception as e:
            print(f"平台控制錯誤: {e}")

    def _apply_actions(self, rl_corrections):
        """應用動作到機器人"""
        #print(f"rl_corrections:{rl_corrections}\n")
        processed_signals = {}
        for leg_idx in range(1, 7):
            processed_signals[leg_idx] = {}
        
        for leg_idx in range(1, 7):
            if leg_idx not in self.motors:
                continue
            
            for joint_idx in range(1, 4):
                if joint_idx not in self.motors[leg_idx]:
                    continue
                
                
                if joint_idx == 2:  # 膝關節
                    motor_angle = 0.0 + rl_corrections[leg_idx - 1]
                else:
                    motor_angle = 0.0
                motor_angle = self._replace_ankle_with_knee_signal(motor_angle, leg_idx, joint_idx, processed_signals)
                motor_angle = self._process_special_joints(motor_angle, leg_idx, joint_idx)
                motor_angle = self._adjust_signal_direction(motor_angle, leg_idx, joint_idx)
                motor_angle = self._apply_height_offset(motor_angle, leg_idx, joint_idx)

                processed_signals[leg_idx][joint_idx] = motor_angle
                
                try:
                    if self.current_step >= self.control_start_step:
                        limited_angle = max(-1.0, min(1.0, motor_angle))
                        self.motors[leg_idx][joint_idx].setPosition(limited_angle)
                    else:
                        init_angle = self._replace_ankle_with_knee_signal(0.0, leg_idx, joint_idx, {})
                        init_angle = self._apply_height_offset(init_angle, leg_idx, joint_idx)
                        self.motors[leg_idx][joint_idx].setPosition(init_angle)
                except Exception as e:
                    print(f"設定馬達角度錯誤 (腿{leg_idx}, 關節{joint_idx}): {e}")

    def _replace_ankle_with_knee_signal(self, motor_angle, leg_idx, joint_idx, processed_signals):
        """將踝關節訊號替換為同隻腳膝關節處理後的訊號"""
        if joint_idx == 3 and self.use_knee_signal_for_ankle:
            if leg_idx in processed_signals and 2 in processed_signals[leg_idx]:
                knee_signal = processed_signals[leg_idx][2]
                return knee_signal * 1
            else:
                knee_angle = 0.0
                knee_angle = self._process_special_joints(knee_angle, leg_idx, 2)
                knee_angle = self._adjust_signal_direction(knee_angle, leg_idx, 2)
                knee_angle = self._apply_height_offset(knee_angle, leg_idx, 2)
                return knee_angle * 1
        return motor_angle

    def _process_special_joints(self, motor_angle, leg_idx, joint_idx):
        """處理特殊關節"""
        if joint_idx == 2 and self.knee_clamp_positive and motor_angle <= 0:
            return 0.0
        
        if joint_idx == 3 and not self.use_knee_signal_for_ankle:
            if (leg_idx == 2 or leg_idx == 5) and motor_angle >= 0:
                return 0.0
        
        return motor_angle

    def _adjust_signal_direction(self, motor_angle, leg_idx, joint_idx):
        """調整訊號方向"""
        if not self.use_knee_signal_for_ankle and leg_idx <= 3 and joint_idx == 3:
            motor_angle = -motor_angle
        
        if leg_idx >= 4 and (joint_idx == 1 or joint_idx == 2):
            motor_angle = -motor_angle
        
        if not self.use_knee_signal_for_ankle and joint_idx == 3:
            if leg_idx in [1, 6, 2, 5]:
                motor_angle = -motor_angle
        
        return motor_angle

    def _apply_height_offset(self, motor_angle, leg_idx, joint_idx):
        """應用機身高度偏移"""
        should_apply_offset = (
            joint_idx == 2 or
            (joint_idx == 3 and not self.use_knee_signal_for_ankle)
        )
        
        if should_apply_offset:
            offset_direction = -1 if leg_idx <= 3 else 1
            return motor_angle + (self.body_height_offset * offset_direction)
        
        return motor_angle

    def _record_episode_stats(self, stability_reward, control_reward, penalty, roll, pitch, x, y):
        """記錄episode統計數據"""
        # 計算與原點的距離
        distance_from_origin = math.sqrt(x**2 + y**2)
        
        self.episode_stats['stability_rewards'].append(stability_reward)
        self.episode_stats['control_rewards'].append(control_reward)
        self.episode_stats['penalties'].append(penalty)
        self.episode_stats['rolls'].append(abs(roll))  # 記錄絕對值
        self.episode_stats['pitches'].append(abs(pitch))  # 記錄絕對值
        self.episode_stats['distances_from_origin'].append(distance_from_origin)

    def _get_episode_averages(self):
        """計算episode平均值"""
        if not self.episode_stats['stability_rewards']:
            return {}
        
        return {
            'avg_stability_reward': np.mean(self.episode_stats['stability_rewards']),
            'avg_control_reward': np.mean(self.episode_stats['control_rewards']),
            'avg_penalty': np.mean(self.episode_stats['penalties']),
            'avg_abs_roll': np.mean(self.episode_stats['rolls']),  # 絕對值的平均
            'avg_abs_pitch': np.mean(self.episode_stats['pitches']),  # 絕對值的平均
            'avg_distance_from_origin': np.mean(self.episode_stats['distances_from_origin']),
            'platform_angle': self.random_platform_angle
        }

    def _reset_episode_stats(self):
        """重置episode統計數據"""
        for key in self.episode_stats:
            self.episode_stats[key].clear()

    def reset(self, seed=None, options=None):
        """重置環境"""
        if seed is not None:
            np.random.seed(seed)
        
        print("🔄 重置環境...")
        
        # 重置模擬
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)
        
        # 重置計數器和狀態序列
        self.current_step = 0
        self.state_sequence.clear()

        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_states = np.zeros(6, dtype=np.float32)
        
        # 重置episode統計
        self._reset_episode_stats()
        
        # 重新初始化設備
        self._init_devices()
        # 設定隨機平台旋轉
        self._set_random_platform_rotation()
        
        # 執行幾步以穩定系統
        for _ in range(3):
            super().step(self.timestep)
        
        # 獲取初始序列觀察
        initial_obs = self._get_sequence_observation()
        
        info = {
            'step': self.current_step,
            'imu_data': self._get_imu_data(),
            'gps_data': self._get_gps_data(),
            'stability_reward': 0.0,
            'control_reward': 0.0,
            'penalty': 0.0,
            'reason': '',
            'roll': 0.0,
            'pitch': 0.0,
            'position_x': 0.0,
            'platform_angle': self.random_platform_angle
        }
        
        return initial_obs, info

    def step(self, action):
        """執行一步動作"""
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 控制平台運動
        self._control_platform()
        
        # 應用動作到機器人，添加隨機噪音以增強探索
        if np.random.random() < self.noise_probability:
            #添加高斯噪音
            action += np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action, -1, 1)
        self._apply_actions(action)
        
        # 執行物理步驟
        
        super().step(self.timestep)
        
        # 更新步數
        self.current_step += 1
        
        # 更新狀態序列
        current_state = self._calculate_single_state()
        self.state_sequence.append(current_state)
        self.prev_states=current_state
        
        # 獲取新的序列觀察
        new_obs = self._get_sequence_observation()
        # 除錯：印出觀察形狀
        """ if self.current_step % 100 == 0:  # 每100步印一次
            print(f"🔍 環境觀察形狀: {new_obs.shape}")
            print(f"📊 觀察數值範圍: [{new_obs.min():.3f}, {new_obs.max():.3f}]") """
        
        # 計算獎勵（包含控制量獎勵）
        reward, stability_reward, control_reward, penalty = self._calculate_reward(action)
        
        # 檢查是否結束
        terminated, truncated, reason = self._is_done()
        # 獲取當前狀態數據
        roll, pitch = self._get_imu_data()
        x, y, z = self._get_gps_data()

        # 記錄episode統計
        self._record_episode_stats(stability_reward, control_reward, penalty, roll, pitch, x, y)
        
        info = {
            'stability_reward': float(stability_reward),
            'control_reward': float(control_reward),
            'penalty': float(penalty),
            'step': self.current_step,
            'reason': reason,
            'imu_data': self._get_imu_data(),
            'gps_data': self._get_gps_data(),
            'roll': float(roll),
            'pitch': float(pitch),
            'position_x': float(x),
            'platform_angle': self.random_platform_angle
        }
        
        # 如果episode結束，添加平均值到info
        if terminated or truncated:
            episode_averages = self._get_episode_averages()
            info.update(episode_averages)

        # 每100步打印進度
        if self.current_step % 100 == 0:
            print(f"步數: {self.current_step}, 總獎勵: {reward:.3f}, "
                f"穩定: {stability_reward:.3f}, 控制: {control_reward:.3f}, "
                f"姿態: roll={roll:.3f}, pitch={pitch:.3f}")
        
        return new_obs, float(reward), terminated, truncated, info

    def close(self):
        """關閉環境"""
        print("👋 關閉環境...")


class WebotsPPOController:
    """
    在 Webots 中運行的 PPO 訓練控制器
    """
    
    def __init__(self, config: HexapodConfig):
        print("🚀 初始化 Webots PPO 訓練控制器")
        
        # 設置隨機種子
        set_random_seed(config.random_seed)
        
        # 創建環境
        self.env = HexapodBalanceEnv(config)

        print("🔍 檢查環境是否符合 Gym API 標準...")
        try:
            check_env(self.env, warn=True, skip_render_check=True)
            print("✅ 環境檢查通過！符合 Gym API 標準")
        except Exception as e:
            print(f"❌ 環境檢查失敗: {e}")
            print("🔧 請修正環境實作後再繼續訓練")
            raise
        
        # 訓練參數
        self.total_timesteps = config.total_timesteps
        self.save_freq = config.get_save_frequency()
        
        # 創建帶編號的訓練資料夾
        self.training_id = self._get_next_training_id()
        self.training_folder = f"training_{self.training_id:03d}"
        self.model_save_path = os.path.join(config.model_save_dir, self.training_folder)
        self.tensorboard_path = os.path.join(config.tensorboard_log_dir, self.training_folder)

        self.config = config
        
        # 確保儲存路徑存在
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.tensorboard_path, exist_ok=True)
        
        print(f"📁 訓練資料夾: {self.training_folder}")
        print(f"💾 模型儲存路徑: {self.model_save_path}")
        print(f"📊 TensorBoard路徑: {self.tensorboard_path}")
        
        # 創建模型
        self._create_model()
        
        # 訓練統計
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.current_episode_reward = 0

        # TensorBoard Writer（在 callback 中初始化）
        self.tb_writer = None
        
        # 保存訓練配置
        self._save_training_config()
        
        print("✅ 控制器初始化完成")

    def _get_next_training_id(self):
        """獲取下一個訓練編號"""
        base_dirs = ["./models", "./tensorboard_logs"]
        max_id = 0
        
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                for folder_name in os.listdir(base_dir):
                    if folder_name.startswith("training_"):
                        try:
                            # 提取編號 training_001 -> 1
                            folder_id = int(folder_name.split("_")[1])
                            max_id = max(max_id, folder_id)
                        except (ValueError, IndexError):
                            continue
        
        next_id = max_id + 1
        print(f"🔢 下一個訓練編號: {next_id}")
        return next_id

    def _save_training_config(self):
        """保存訓練配置到檔案"""
        import json
        from datetime import datetime
        
        config = {
            "training_id": self.training_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_timesteps": self.total_timesteps,
            "save_frequency": self.save_freq,
            "environment": {
                "max_episode_steps": self.env.max_episode_steps,
                "sequence_length": self.env.sequence_length,
                "knee_clamp_positive": self.env.knee_clamp_positive,
                "use_knee_signal_for_ankle": self.env.use_knee_signal_for_ankle,
                "body_height_offset": self.env.body_height_offset,
                "control_start_step": self.env.control_start_step,
                "stability_reward_weight": self.env.w_s,
                "control_reward_weight": self.env.w_c
            },
            "gpu_info": {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "pytorch_version": torch.__version__
            }
        }
        
        config_path = os.path.join(self.model_save_path, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"📝 訓練配置已保存: {config_path}")

    def _create_model(self):
        """創建 PPO 模型"""
        print("🤖 創建 PPO 模型...")
        
        # 檢查 GPU 可用性
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 使用設備: {device}")
        
        if device == 'cuda':
            print(f"   GPU 名稱: {torch.cuda.get_device_name(0)}")
            print(f"   GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        lr_schedule = self.config.get_learning_rate()
        
        # 創建策略參數
        policy_kwargs = {
            'features_extractor_class': TransformerFeaturesExtractor,
            'features_extractor_kwargs': {'config': self.config},  # 傳遞配置
            'net_arch': self.config.get_net_arch()
        }
        
        # 創建向量化環境
        self.vec_env = DummyVecEnv([lambda: Monitor(self.env)])
        
        # 創建 PPO 模型
        """ self.model = PPO(
            TransformerActorCriticPolicy,
            self.vec_env,
            learning_rate=lr_schedule,
            n_steps=4096,  
            batch_size=2048,  
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,  # 自動選擇最佳設備
            tensorboard_log=self.tensorboard_path  # 使用編號資料夾
        ) """
        self.model = PPO(
            TransformerActorCriticPolicy,
            self.vec_env,
            learning_rate=lr_schedule,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=self.tensorboard_path
        )
        print("=== 策略架構 ===")
        print(self.model.policy)
        
        print(f"🧠 模型已創建，參數數量: {sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)}")
        print(f"💾 模型設備: {next(self.model.policy.parameters()).device}")

    def run(self):
        """主訓練循環"""
        print("🏃 開始訓練...")
        print("-" * 60)
        
        try:
            # 開始訓練
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=self._training_callback,
                log_interval=1,
                reset_num_timesteps=False
            )
            
            print("✅ 訓練完成！")
            
            # 儲存最終模型
            final_model_path = os.path.join(self.model_save_path, "ppo_hexapod_final")
            self.model.save(final_model_path)
            print(f"💾 最終模型已儲存: {final_model_path}")
            
        except KeyboardInterrupt:
            print("\n⏹️  訓練被用戶中斷")
            
            # 儲存中斷時的模型
            interrupt_model_path = os.path.join(self.model_save_path, "ppo_hexapod_interrupted")
            self.model.save(interrupt_model_path)
            print(f"💾 中斷模型已儲存: {interrupt_model_path}")
        
        except Exception as e:
            print(f"❌ 訓練過程中發生錯誤: {e}")
            
            # 儲存錯誤時的模型
            error_model_path = os.path.join(self.model_save_path, "ppo_hexapod_error")
            self.model.save(error_model_path)
            print(f"💾 錯誤模型已儲存: {error_model_path}")
            raise
        
        finally:
            # 清理資源
            if hasattr(self, 'tb_writer') and self.tb_writer:
                self.tb_writer.close()
                print("📊 TensorBoard Writer 已關閉")
            self.env.close()
            print("🧹 資源清理完成")

    def _training_callback(self, locals_, globals_):
        """整合TensorBoard的訓練回調函數"""
        self.total_steps = locals_['self'].num_timesteps
        
        # 初始化 TensorBoard Writer
        if not hasattr(self, 'tb_writer') or self.tb_writer is None:
            self.tb_writer = SummaryWriter(os.path.join(self.tensorboard_path, "detailed_logs"))
            print(f"📊 TensorBoard Writer 已初始化: {self.tensorboard_path}")
        
        # 記錄訓練指標
        if len(locals_['infos']) > 0:
            for info in locals_['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    # 記錄到 TensorBoard
                    self.tb_writer.add_scalar('Episode/Reward', episode_reward, self.total_steps)
                    self.tb_writer.add_scalar('Episode/Length', episode_length, self.total_steps)
                    
                                        # 記錄episode平均值（修正後）
                    if 'avg_stability_reward' in info:
                        self.tb_writer.add_scalar('Environment/Avg_Stability_Reward', info['avg_stability_reward'], self.total_steps)
                    if 'avg_control_reward' in info:
                        self.tb_writer.add_scalar('Environment/Avg_Control_Reward', info['avg_control_reward'], self.total_steps)
                    if 'avg_penalty' in info:
                        self.tb_writer.add_scalar('Environment/Avg_Penalty', info['avg_penalty'], self.total_steps)
                    if 'avg_abs_roll' in info:
                        self.tb_writer.add_scalar('Environment/Avg_Abs_Roll', info['avg_abs_roll'], self.total_steps)
                    if 'avg_abs_pitch' in info:
                        self.tb_writer.add_scalar('Environment/Avg_Abs_Pitch', info['avg_abs_pitch'], self.total_steps)
                    if 'avg_distance_from_origin' in info:
                        self.tb_writer.add_scalar('Environment/Avg_Distance_From_Origin', info['avg_distance_from_origin'], self.total_steps)
                    if 'platform_angle' in info:
                        self.tb_writer.add_scalar('Environment/Platform_Angle', info['platform_angle'], self.total_steps)
                    
                    # 記錄獎勵分解（修正後）
                    if 'avg_stability_reward' in info and 'avg_control_reward' in info:
                        total_reward_components = (
                            self.env.w_s * info['avg_stability_reward'] + 
                            self.env.w_c * info['avg_control_reward'] + 
                            info.get('avg_penalty', 0)
                        )
                        self.tb_writer.add_scalar('Reward/Total_Components', total_reward_components, self.total_steps)
                        self.tb_writer.add_scalar('Reward/Stability_Weighted', self.env.w_s * info['avg_stability_reward'], self.total_steps)
                        self.tb_writer.add_scalar('Reward/Control_Weighted', self.env.w_c * info['avg_control_reward'], self.total_steps)
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_count += 1
                    
                    print(f"📈 Episode {self.episode_count} 完成: 總獎勵={episode_reward:.3f}, 長度={episode_length}")
                    if 'avg_stability_reward' in info:
                        print(f"   平均穩定獎勵: {info['avg_stability_reward']:.3f}")
                    if 'avg_control_reward' in info:
                        print(f"   平均控制獎勵: {info['avg_control_reward']:.3f}")
                    if 'avg_abs_roll' in info and 'avg_abs_pitch' in info:
                        print(f"   平均絕對姿態: roll={info['avg_abs_roll']:.3f}, pitch={info['avg_abs_pitch']:.3f}")
                    if 'avg_distance_from_origin' in info:
                        print(f"   平均距離原點: {info['avg_distance_from_origin']:.3f}")
                    if 'platform_angle' in info:
                        print(f"   平台角度: {math.degrees(info['platform_angle']):.1f}°")
                    
                    # 計算平均獎勵
                    if len(self.episode_rewards) >= 10:
                        avg_reward = np.mean(self.episode_rewards[-10:])
                        self.tb_writer.add_scalar('Episode/Average_Reward_10', avg_reward, self.total_steps)
                        print(f"📊 最近10個episode平均獎勵: {avg_reward:.3f}")
        
        # 記錄學習指標
        if hasattr(locals_['self'], 'logger') and locals_['self'].logger.name_to_value:
            for key, value in locals_['self'].logger.name_to_value.items():
                if any(keyword in key.lower() for keyword in ['loss', 'entropy', 'kl', 'value', 'policy']):
                    self.tb_writer.add_scalar(f'Training/{key}', value, self.total_steps)
        
        # 定期儲存模型
        if self.total_steps % self.save_freq == 0 and self.total_steps > 0:
            model_path = os.path.join(self.model_save_path, f"ppo_hexapod_{self.total_steps}")
            self.model.save(model_path)
            print(f"💾 模型已儲存: {model_path} (步數: {self.total_steps})")
            
            # 顯示 GPU 記憶體用量（如果使用 GPU）
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"📊 GPU 記憶體: {memory_allocated:.2f} GB / {memory_reserved:.2f} GB")
                
                # 記錄 GPU 記憶體使用到 TensorBoard
                self.tb_writer.add_scalar('System/GPU_Memory_Allocated_GB', memory_allocated, self.total_steps)
                self.tb_writer.add_scalar('System/GPU_Memory_Reserved_GB', memory_reserved, self.total_steps)
        
        # 強制刷新 TensorBoard 寫入
        if self.tb_writer:
            self.tb_writer.flush()
        
        return True


def check_gpu_requirements():
    """檢查 GPU 需求和環境"""
    print("🔍 檢查 GPU 環境...")
    
    # 檢查 PyTorch CUDA 支援
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 數量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"   記憶體: {props.total_memory / 1024**3:.1f} GB")
            print(f"   計算能力: {props.major}.{props.minor}")
        
        # 測試 GPU 記憶體
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✅ GPU 記憶體測試通過")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU 記憶體測試失敗: {e}")
            return False
            
        return True
    else:
        print("⚠️  GPU 不可用，將使用 CPU")
        print("建議安裝支援 CUDA 的 PyTorch 版本:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False


def optimize_gpu_settings():
    """優化 GPU 設定"""
    if torch.cuda.is_available():
        print("⚙️  優化 GPU 設定...")
        
        # 啟用 cuDNN 自動調優
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN benchmark 已啟用")
        
        # 設定記憶體分配策略
        torch.cuda.empty_cache()
        print("✅ GPU 記憶體已清空")
        
        # 顯示初始記憶體使用
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 GPU 記憶體使用: {memory_allocated:.2f} GB / {memory_reserved:.2f} GB")


def main():
    """
    Webots Controller 主入口點 - 專門用於訓練
    """
    print("=" * 60)
    print("🕷️  六足機器人 PPO + Transformer 訓練程式")
    print("=" * 60)

    config = HexapodConfig()

    # === 可選：調整特定參數 ===
    # config.learning_rate_start = 1e-4  # 修改學習率
    # config.batch_size = 512            # 修改批次大小
    # config.transformer_n_heads = 8     # 修改注意力頭數
    # config.w_s = 1.5
    
    try:
        # 檢查 GPU 環境
        gpu_available = check_gpu_requirements()
        
        # 優化 GPU 設定
        if gpu_available:
            optimize_gpu_settings()
        
        # 創建並運行控制器
        controller = WebotsPPOController(config)
        
        # 顯示 TensorBoard 使用說明
        print(f"\n📊 TensorBoard 使用說明:")
        print("1. 訓練開始後，開啟新的命令列視窗")
        print(f"2. 執行: tensorboard --logdir=./tensorboard_logs/{controller.training_folder}")
        print("3. 在瀏覽器開啟: http://localhost:6006")
        print("4. 即可即時監控訓練進度！")
        print(f"📁 本次訓練資料夾: {controller.training_folder}")
        print("   • Environment/Avg_* - Episode平均值")
        print("   • Environment/Avg_Distance_From_Origin - 與原點距離")
        print("   • Environment/Avg_Abs_Roll/Pitch - 絕對值姿態角")
        print("   • Reward/* - 獎勵成分分解")
        print("   • Environment/Platform_Angle - 隨機平台角度")
        print("-" * 60)
        
        controller.run()
        
    except Exception as e:
        print(f"❌ 控制器運行失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 程式結束")


# 程式入口點
if __name__ == "__main__":
    print("🎯 運行模式: 訓練專用")
    main()