#!/usr/bin/env python3
"""
六足機器人 PPO + Transformer Webots 測試控制器
專門用於測試已訓練模型的性能
"""

import sys
import os
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from datetime import datetime

# Webots imports
from controller import Supervisor

try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    print(f'請安裝必要套件: pip install stable-baselines3[extra] gymnasium')
    sys.exit(1)


class PositionalEncoding(nn.Module):
    """位置編碼模組"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """自定義 Transformer 特徵提取器"""
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        sequence_length: int = 50,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)
        
        self.sequence_length = sequence_length
        self.state_dim = observation_space.shape[0] // sequence_length
        self._features_dim = features_dim
        
        self.input_projection = nn.Linear(self.state_dim, features_dim)
        self.pos_encoding = PositionalEncoding(features_dim, sequence_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=n_heads,
            dim_feedforward=features_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(features_dim, features_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.sequence_length, self.state_dim)
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x[-1]
        features = self.output_projection(x)
        return features


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """自定義 Actor-Critic 策略，使用 Transformer 特徵提取器"""
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=None,
        *args,
        **kwargs
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs
        )


class HexapodTestEnv(Supervisor, gym.Env):
    """六足機器人測試環境（簡化版，移除訓練相關功能）"""
    
    def __init__(self, max_episode_steps=2000, sequence_length=50):
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.sequence_length = sequence_length
        self.current_step = 0
        
        # CPG參數
        self.knee_clamp_positive = True
        self.use_knee_signal_for_ankle = True
        self.body_height_offset = 0.5
        self.control_start_step = 100
        
        self.spec = type('SimpleSpec', (), {'id': 'HexapodTest-v0','max_episode_steps':max_episode_steps})()
        
        self._setup_spaces()
        
        # Webots設置
        self.timestep = int(self.getBasicTimeStep())
        self.motors = {}
        self.gps_device = None
        self.imu_device = None
        self.platform_motor_joint = None
        
        # 狀態序列緩存
        self.state_sequence = deque(maxlen=sequence_length)
        
        # 初始化設備
        self._init_devices()

    def _setup_spaces(self):
        """設置狀態和動作空間"""
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        single_state_dim = 6
        sequence_dim = self.sequence_length * single_state_dim
        
        self.observation_space = spaces.Box(
            low=-4.0, high=4.0, shape=(sequence_dim,), dtype=np.float32
        )

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
            1: ('R0', '右前腿'), 2: ('R1', '右中腿'), 3: ('R2', '右後腿'),
            4: ('L2', '左後腿'), 5: ('L1', '左中腿'), 6: ('L0', '左前腿')
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
        current_length = len(self.state_sequence)
        
        if current_length == 0:
            sequence = np.zeros((self.sequence_length, 6), dtype=np.float32)
        elif current_length < self.sequence_length:
            padding_length = self.sequence_length - current_length
            padding = np.zeros((padding_length, 6), dtype=np.float32)
            existing_sequence = np.array(list(self.state_sequence), dtype=np.float32)
            sequence = np.vstack([padding, existing_sequence])
        else:
            sequence = np.array(list(self.state_sequence), dtype=np.float32)
        
        return sequence.flatten()

    def _calculate_reward(self):
        """計算獎勵函數"""
        roll, pitch = self._get_imu_data()
        x, y, z = self._get_gps_data()
        
        stability_term = (abs(pitch) + abs(roll)) / 2
        r_s = math.exp(-(stability_term ** 2) / (0.1 ** 2))
        
        p = 0
        if abs(pitch) >= 0.785 or abs(roll) >= 0.785:
            p += -1
        
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            p += -1
        
        total_reward = r_s + p
        return total_reward, r_s, p

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
            target_angle = 0.2 * math.sin(1.0 * math.pi * current_time)
            
            joint_params_field = self.platform_motor_joint.getField("jointParameters")
            joint_params_node = joint_params_field.getSFNode()
            if joint_params_node:
                position_field = joint_params_node.getField("position")
                if position_field:
                    position_field.setSFFloat(target_angle)
                    
        except Exception as e:
            pass

    def _apply_actions(self, rl_corrections):
        """應用動作到機器人"""
        processed_signals = {}
        for leg_idx in range(1, 7):
            processed_signals[leg_idx] = {}
        
        for leg_idx in range(1, 7):
            if leg_idx not in self.motors:
                continue
            
            for joint_idx in range(1, 4):
                if joint_idx not in self.motors[leg_idx]:
                    continue
                
                motor_angle = 0.0
                motor_angle = self._replace_ankle_with_knee_signal(motor_angle, leg_idx, joint_idx, processed_signals)
                motor_angle = self._process_special_joints(motor_angle, leg_idx, joint_idx)
                motor_angle = self._adjust_signal_direction(motor_angle, leg_idx, joint_idx)
                motor_angle = self._apply_height_offset(motor_angle, leg_idx, joint_idx)
                
                if joint_idx == 2:
                    final_motor_angle = motor_angle + rl_corrections[leg_idx - 1]
                else:
                    final_motor_angle = motor_angle
                
                processed_signals[leg_idx][joint_idx] = final_motor_angle
                
                try:
                    if self.current_step >= self.control_start_step:
                        limited_angle = max(-1.0, min(1.0, final_motor_angle))
                        self.motors[leg_idx][joint_idx].setPosition(limited_angle)
                    else:
                        init_angle = self._replace_ankle_with_knee_signal(0.0, leg_idx, joint_idx, {})
                        init_angle = self._apply_height_offset(init_angle, leg_idx, joint_idx)
                        self.motors[leg_idx][joint_idx].setPosition(init_angle)
                except Exception as e:
                    pass

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

    def reset(self, seed=None, options=None):
        """重置環境"""
        if seed is not None:
            np.random.seed(seed)
        
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)
        
        self.current_step = 0
        self.state_sequence.clear()
        
        self._init_devices()
        
        for _ in range(3):
            super().step(self.timestep)
        
        initial_obs = self._get_sequence_observation()
        
        info = {
            'step': self.current_step,
            'imu_data': self._get_imu_data(),
            'gps_data': self._get_gps_data(),
            'stability_reward': 0.0,
            'penalty': 0.0,
            'reason': '',
            'roll': 0.0,
            'pitch': 0.0,
            'position_x': 0.0
        }
        
        return initial_obs, info

    def step(self, action):
        """執行一步動作"""
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self._control_platform()
        self._apply_actions(action)
        super().step(self.timestep)
        
        self.current_step += 1
        
        current_state = self._calculate_single_state()
        self.state_sequence.append(current_state)
        
        new_obs = self._get_sequence_observation()
        reward, stability_reward, penalty = self._calculate_reward()
        terminated, truncated, reason = self._is_done()
        
        roll, pitch = self._get_imu_data()
        x, y, z = self._get_gps_data()
        
        info = {
            'stability_reward': float(stability_reward),
            'penalty': float(penalty),
            'step': self.current_step,
            'reason': reason,
            'imu_data': self._get_imu_data(),
            'gps_data': self._get_gps_data(),
            'roll': float(roll),
            'pitch': float(pitch),
            'position_x': float(x)
        }
        
        return new_obs, float(reward), terminated, truncated, info

    def close(self):
        """關閉環境"""
        pass


class TestResult:
    """測試結果記錄類別"""
    
    def __init__(self):
        self.episodes = []
        self.start_time = None
        self.end_time = None
        self.model_info = {}
        self.config = {}
    
    def add_episode(self, episode_data: Dict[str, Any]):
        """添加episode結果"""
        self.episodes.append(episode_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """計算統計數據"""
        if not self.episodes:
            return {}
        
        rewards = [ep['total_reward'] for ep in self.episodes]
        durations = [ep['duration'] for ep in self.episodes]
        final_rolls = [ep['final_roll'] for ep in self.episodes]
        final_pitches = [ep['final_pitch'] for ep in self.episodes]
        
        success_count = sum(1 for ep in self.episodes if ep['reason'] == '超時')
        
        stats = {
            'total_episodes': len(self.episodes),
            'success_rate': success_count / len(self.episodes) * 100,
            'average_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'average_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'min_duration': np.min(durations),
            'average_final_roll': np.mean(np.abs(final_rolls)),
            'average_final_pitch': np.mean(np.abs(final_pitches)),
            'max_abs_roll': np.max(np.abs(final_rolls)),
            'max_abs_pitch': np.max(np.abs(final_pitches))
        }
        
        return stats
    
    def save_to_file(self, filepath: str):
        """儲存結果到檔案"""
        result_data = {
            'test_info': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
                'model_info': self.model_info,
                'config': self.config
            },
            'statistics': self.get_statistics(),
            'episodes': self.episodes
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)


class WebotsPPOTester:
    """Webots PPO 測試器"""
    
    def __init__(self):
        print("🧪 初始化 Webots PPO 測試器")
        
        # 測試配置
        self.config = {
            'num_episodes': 10,           # 測試episode數量
            'max_episode_steps': 2000,    # 每個episode最大步數
            'sequence_length': 50,        # 序列長度
            'deterministic': True,        # 是否使用確定性動作
            'verbose': True,              # 詳細輸出
            'save_results': True          # 是否儲存結果
        }
        
        # 模型相關
        self.model = None
        self.model_path = None
        self.model_info = {}
        
        # 環境
        self.env = None
        
        # 結果記錄
        self.test_result = TestResult()

    def list_available_models(self) -> List[Tuple[str, List[str]]]:
        """列出可用的訓練模型"""
        available_models = []
        models_dir = "./models"
        
        if not os.path.exists(models_dir):
            print("❌ 找不到模型資料夾: ./models")
            return available_models
        
        # 尋找訓練資料夾
        for folder_name in sorted(os.listdir(models_dir)):
            if folder_name.startswith("training_"):
                training_path = os.path.join(models_dir, folder_name)
                if os.path.isdir(training_path):
                    # 尋找該訓練資料夾中的模型檔案
                    model_files = []
                    for file_name in os.listdir(training_path):
                        if file_name.startswith("ppo_hexapod_") and not file_name.endswith(".json"):
                            model_files.append(file_name)
                    
                    if model_files:
                        available_models.append((folder_name, sorted(model_files)))
        
        return available_models

    def select_model(self, training_id: Optional[str] = None, model_name: Optional[str] = None, custom_path: Optional[str] = None) -> bool:
        """選擇要測試的模型"""
        
        # 如果指定了自定義路徑，直接使用
        if custom_path:
            if os.path.exists(custom_path + ".zip"):
                self.model_path = custom_path
                self.model_info = {
                    'training_folder': 'custom',
                    'model_file': os.path.basename(custom_path),
                    'model_path': custom_path
                }
                print(f"✅ 已選擇自定義模型: {custom_path}")
                return True
            else:
                print(f"❌ 自定義模型檔案不存在: {custom_path}.zip")
                return False
        
        available_models = self.list_available_models()
        
        if not available_models:
            print("❌ 沒有找到可用的模型檔案")
            return False
        
        print("\n📋 可用的訓練模型:")
        for i, (training_folder, model_files) in enumerate(available_models):
            print(f"  {i+1}. {training_folder}")
            for model_file in model_files:
                print(f"     └── {model_file}")
        
        # 如果指定了training_id，尋找對應的模型
        if training_id:
            target_folder = f"training_{training_id:03d}" if isinstance(training_id, int) else training_id
            for training_folder, model_files in available_models:
                if training_folder == target_folder:
                    if model_name and model_name in model_files:
                        selected_model = model_name
                    else:
                        # 預設選擇final模型，如果沒有則選擇最後一個
                        final_models = [f for f in model_files if "final" in f]
                        if final_models:
                            selected_model = final_models[0]
                        else:
                            selected_model = model_files[-1]
                    
                    self.model_path = os.path.join("./models", training_folder, selected_model)
                    self.model_info = {
                        'training_folder': training_folder,
                        'model_file': selected_model,
                        'model_path': self.model_path
                    }
                    print(f"✅ 已選擇模型: {training_folder}/{selected_model}")
                    return True
            
            print(f"❌ 找不到指定的訓練: {target_folder}")
            return False
        
        # 如果沒有指定，自動選擇最新的模型
        if available_models:
            # 選擇最新的訓練資料夾（按資料夾名稱排序，最後一個）
            latest_training_folder, model_files = available_models[-1]
            
            # 優先選擇final模型，如果沒有則選擇最後一個模型
            final_models = [f for f in model_files if "final" in f]
            if final_models:
                selected_model = final_models[0]
            else:
                selected_model = model_files[-1]
            
            self.model_path = os.path.join("./models", latest_training_folder, selected_model)
            self.model_info = {
                'training_folder': latest_training_folder,
                'model_file': selected_model,
                'model_path': self.model_path
            }
            print(f"✅ 自動選擇最新模型: {latest_training_folder}/{selected_model}")
            return True
        
        print("❌ 沒有可用的模型")
        return False

    def load_model(self) -> bool:
        """載入選定的模型"""
        if not self.model_path:
            print("❌ 尚未選擇模型")
            return False
        
        if not os.path.exists(self.model_path + ".zip"):
            print(f"❌ 模型檔案不存在: {self.model_path}.zip")
            return False
        
        try:
            print(f"📦 載入模型: {self.model_path}")
            
            # 檢查設備
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🔧 使用設備: {device}")
            
            # 載入模型
            self.model = PPO.load(self.model_path, device=device)
            
            # 讀取訓練配置（如果存在）
            config_path = os.path.join(os.path.dirname(self.model_path), "training_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    training_config = json.load(f)
                self.model_info['training_config'] = training_config
                print(f"📝 已載入訓練配置")
            
            print("✅ 模型載入成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False

    def create_test_environment(self) -> bool:
        """創建測試環境"""
        try:
            print("🌍 創建測試環境...")
            
            # 從模型配置中獲取環境參數（如果可用）
            env_config = {}
            if 'training_config' in self.model_info and 'environment' in self.model_info['training_config']:
                env_config = self.model_info['training_config']['environment']
                print(f"📝 使用訓練時的環境配置")
            
            # 創建環境
            self.env = HexapodTestEnv(
                max_episode_steps=self.config['max_episode_steps'],
                sequence_length=self.config['sequence_length']
            )
            
            # 如果有訓練配置，同步環境參數
            if env_config:
                self.env.knee_clamp_positive = env_config.get('knee_clamp_positive', True)
                self.env.use_knee_signal_for_ankle = env_config.get('use_knee_signal_for_ankle', True)
                self.env.body_height_offset = env_config.get('body_height_offset', 0.5)
                self.env.control_start_step = env_config.get('control_start_step', 100)
                print(f"⚙️  環境參數已同步訓練配置")
            
            print("✅ 測試環境創建成功")
            return True
            
        except Exception as e:
            print(f"❌ 測試環境創建失敗: {e}")
            return False

    def run_single_episode(self, episode_num: int) -> Dict[str, Any]:
        """執行單個測試episode"""
        print(f"\n🚀 開始Episode {episode_num}")
        
        # 重置環境
        obs, info = self.env.reset()
        episode_reward = 0.0
        step_count = 0
        episode_data = []
        
        done = False
        while not done:
            # 預測動作
            action, _states = self.model.predict(obs, deterministic=self.config['deterministic'])
            print(f"步數 {step_count}: 動作 = {action}")
            print(f"觀察形狀: {obs.shape}, 動作形狀: {action.shape}")
            print(f"   🔍 觀察前10個值: {obs[:10]}")
            
            # 執行動作
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # 記錄步驟數據（可選）
            step_data = {
                'step': step_count,
                'reward': reward,
                'roll': info['roll'],
                'pitch': info['pitch'],
                'position_x': info['position_x'],
                'action': action.tolist() if hasattr(action, 'tolist') else list(action)
            }
            episode_data.append(step_data)
            
            # 詳細輸出（每50步）
            if self.config['verbose'] and step_count % 50 == 0:
                print(f"   步數: {step_count}, 獎勵: {episode_reward:.3f}, "
                      f"姿態: roll={info['roll']:.3f}, pitch={info['pitch']:.3f}")
        
        # 記錄最終狀態
        final_roll, final_pitch = info['roll'], info['pitch']
        final_x = info['position_x']
        end_reason = info['reason']
        
        episode_result = {
            'episode_num': episode_num,
            'total_reward': episode_reward,
            'duration': step_count,
            'end_reason': end_reason,
            'initial_roll': initial_roll,
            'initial_pitch': initial_pitch,
            'final_roll': final_roll,
            'final_pitch': final_pitch,
            'initial_position': [initial_x, initial_y, initial_z],
            'final_position_x': final_x,
            'max_abs_roll': max(abs(step['roll']) for step in episode_data),
            'max_abs_pitch': max(abs(step['pitch']) for step in episode_data),
            'reason': end_reason,
            'step_data': episode_data if self.config.get('save_step_data', False) else []
        }
        
        print(f"✅ Episode {episode_num} 完成:")
        print(f"   📊 總獎勵: {episode_reward:.3f}")
        print(f"   ⏱️  持續時間: {step_count} 步")
        print(f"   🎯 結束原因: {end_reason}")
        print(f"   📐 最終姿態: roll={final_roll:.3f}, pitch={final_pitch:.3f}")
        
        return episode_result

    def run_test(self) -> bool:
        """執行完整測試"""
        if not self.model:
            print("❌ 尚未載入模型")
            return False
        
        if not self.env:
            print("❌ 尚未創建環境")
            return False
        
        print(f"\n🧪 開始測試")
        print(f"📋 測試配置:")
        print(f"   Episode數量: {self.config['num_episodes']}")
        print(f"   每Episode最大步數: {self.config['max_episode_steps']}")
        print(f"   序列長度: {self.config['sequence_length']}")
        print(f"   確定性動作: {self.config['deterministic']}")
        print("-" * 50)
        
        # 記錄開始時間
        self.test_result.start_time = datetime.now()
        self.test_result.model_info = self.model_info.copy()
        self.test_result.config = self.config.copy()
        
        # 執行多個episode
        for episode_num in range(1, self.config['num_episodes'] + 1):
            try:
                episode_result = self.run_single_episode(episode_num)
                self.test_result.add_episode(episode_result)
                
            except KeyboardInterrupt:
                print("\n⏹️  測試被用戶中斷")
                break
            except Exception as e:
                print(f"❌ Episode {episode_num} 執行失敗: {e}")
                continue
        
        # 記錄結束時間
        self.test_result.end_time = datetime.now()
        
        # 顯示統計結果
        self._display_statistics()
        
        # 儲存結果
        if self.config['save_results']:
            self._save_results()
        
        return True

    def _display_statistics(self):
        """顯示測試統計結果"""
        stats = self.test_result.get_statistics()
        
        if not stats:
            print("❌ 沒有可用的統計數據")
            return
        
        print("\n" + "=" * 60)
        print("📊 測試統計結果")
        print("=" * 60)
        
        print(f"🔢 總Episode數: {stats['total_episodes']}")
        print(f"🎯 成功率: {stats['success_rate']:.1f}%")
        print(f"📈 平均獎勵: {stats['average_reward']:.3f} ± {stats['std_reward']:.3f}")
        print(f"📈 獎勵範圍: {stats['min_reward']:.3f} ~ {stats['max_reward']:.3f}")
        print(f"⏱️  平均持續時間: {stats['average_duration']:.1f} 步")
        print(f"⏱️  持續時間範圍: {stats['min_duration']} ~ {stats['max_duration']} 步")
        print(f"📐 平均最終姿態偏差:")
        print(f"   Roll: {stats['average_final_roll']:.3f} (最大: {stats['max_abs_roll']:.3f})")
        print(f"   Pitch: {stats['average_final_pitch']:.3f} (最大: {stats['max_abs_pitch']:.3f})")
        
        # 結束原因統計
        reasons = {}
        for episode in self.test_result.episodes:
            reason = episode['end_reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        
        print(f"\n📋 結束原因統計:")
        for reason, count in reasons.items():
            percentage = count / len(self.test_result.episodes) * 100
            print(f"   {reason}: {count} 次 ({percentage:.1f}%)")
        
        print("=" * 60)

    def _save_results(self):
        """儲存測試結果到檔案"""
        try:
            # 創建結果資料夾
            results_dir = "./test_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_folder = self.model_info.get('training_folder', 'unknown')
            model_file = self.model_info.get('model_file', 'unknown')
            
            filename = f"test_{training_folder}_{model_file}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # 儲存結果
            self.test_result.save_to_file(filepath)
            
            print(f"💾 測試結果已儲存: {filepath}")
            
        except Exception as e:
            print(f"❌ 儲存結果失敗: {e}")

    def configure_test(self, **kwargs):
        """配置測試參數"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"⚙️  {key} 設定為: {value}")
            else:
                print(f"⚠️  未知配置項: {key}")

    def close(self):
        """關閉測試器"""
        if self.env:
            self.env.close()
            print("🧹 環境已關閉")


def main():
    """主程式入口點"""
    print("=" * 60)
    print("🧪 六足機器人 PPO 模型測試程式")
    print("=" * 60)
    
    # ============================================
    # 測試配置區（直接在此處修改參數）
    # ============================================
    
    # 模型選擇配置
    MODEL_CONFIG = {
        'training_id': None,           # 指定訓練編號，如: 1, "001", "training_001"，None為自動選擇最新
        'model_name': None,            # 指定模型檔案名，如: "ppo_hexapod_final", None為自動選擇
        'custom_path': "C:/Users/User/Desktop/J/transformer_DRL_new_webots/controllers/rl_training_controller3/models/ppo_hexapod_500000",           # 自定義模型路徑，如: "./my_models/best_model"，會覆蓋上面的設定
    }
    
    # 自定義模型路徑使用範例：
    # MODEL_CONFIG = {
    #     'custom_path': "./my_models/renamed_model",              # 基本路徑
    #     # 或者
    #     'custom_path': "./models/training_001/my_best_model",    # 重命名的模型
    #     # 或者  
    #     'custom_path': "/absolute/path/to/your/model",           # 絕對路徑
    # }
    
    # 測試參數配置
    TEST_CONFIG = {
        'num_episodes': 1,            # 測試episode數量
        'max_episode_steps': 2000,     # 每個episode最大步數
        'sequence_length': 50,         # 序列長度（需與訓練時一致）
        'deterministic': True,         # 是否使用確定性動作（True=無隨機性）
        'verbose': True,               # 是否顯示詳細輸出
        'save_results': True,          # 是否儲存測試結果到檔案
        'save_step_data': True,       # 是否儲存每步詳細數據（會增大檔案）
    }
    
    print("📝 測試配置:")
    if MODEL_CONFIG['custom_path']:
        print(f"   自定義模型路徑: {MODEL_CONFIG['custom_path']}")
    else:
        print(f"   模型訓練編號: {MODEL_CONFIG['training_id'] or '自動選擇最新'}")
        print(f"   模型檔案名: {MODEL_CONFIG['model_name'] or '自動選擇'}")
    print(f"   測試Episodes: {TEST_CONFIG['num_episodes']}")
    print(f"   Episode最大步數: {TEST_CONFIG['max_episode_steps']}")
    print(f"   確定性動作: {TEST_CONFIG['deterministic']}")
    print("-" * 50)
    
    # ============================================
    # 測試執行流程
    # ============================================
    
    # 創建測試器
    tester = WebotsPPOTester()
    
    try:
        # 應用測試配置
        tester.configure_test(**TEST_CONFIG)
        
        # 1. 選擇模型
        print("\n🔍 步驟1: 選擇測試模型")
        if not tester.select_model(
            training_id=MODEL_CONFIG['training_id'], 
            model_name=MODEL_CONFIG['model_name'],
            custom_path=MODEL_CONFIG['custom_path']
        ):
            print("❌ 模型選擇失敗")
            return
        
        # 2. 載入模型
        print("\n📦 步驟2: 載入模型")
        if not tester.load_model():
            print("❌ 模型載入失敗")
            return
        
        # 3. 創建測試環境
        print("\n🌍 步驟3: 創建測試環境")
        if not tester.create_test_environment():
            print("❌ 測試環境創建失敗")
            return
        
        # 4. 執行測試
        print("\n🚀 步驟4: 執行測試")
        if not tester.run_test():
            print("❌ 測試執行失敗")
            return
        
        print("\n✅ 所有測試完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️  程式被用戶中斷")
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理資源
        tester.close()
    
    print("👋 程式結束")


# 程式入口點
if __name__ == "__main__":
    print("🎯 運行模式: 測試專用")
    main()