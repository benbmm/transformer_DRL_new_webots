#!/usr/bin/env python3
"""
六足機器人 PPO + Transformer 模型測試程式
簡單的功能測試，無需對比和報告
"""

import sys
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, Union
from collections import deque

# Webots imports
from controller import Supervisor

try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    print(f'請安裝必要套件: pip install stable-baselines3[extra] gymnasium')
    sys.exit(1)


# ===== 複製訓練程式中的必要類別 =====

class PositionalEncoding(nn.Module):
    """位置編碼模組 - 支援 batch_first"""
    
    def __init__(self, d_model: int, max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        if self.batch_first:
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]
        else:
            seq_len = x.size(0)
            return x + self.pe[:seq_len, :]


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
            dim_feedforward=features_dim * 2,
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
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
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
    """
    六足機器人測試環境 - 簡化版
    只保留測試所需的基本功能
    """
    
    def __init__(self, max_episode_steps=1000, sequence_length=50):
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
        
        self.timestep = int(self.getBasicTimeStep())
        self.motors = {}
        self.gps_device = None
        self.imu_device = None
        self.platform_motor_joint = None
        
        self.state_sequence = deque(maxlen=sequence_length)
        
        self._init_devices()
        
        print("✅ 測試環境已初始化")

    def _setup_spaces(self):
        """設置狀態和動作空間"""
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        single_state_dim = 6
        sequence_dim = self.sequence_length * single_state_dim
        
        self.observation_space = spaces.Box(
            low=-4.0, 
            high=4.0, 
            shape=(sequence_dim,), 
            dtype=np.float32
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
        
        flattened = sequence.flatten()
        return flattened

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
            print(f"平台控制錯誤: {e}")

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
                
                if joint_idx == 2:  # 膝關節
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
        print("👋 關閉測試環境...")


class HexapodTester:
    """
    六足機器人模型測試器
    """
    
    def __init__(self, test_steps=500):
        print("🧪 初始化六足機器人模型測試器")
        
        # 測試參數設定
        self.test_steps = test_steps
        self.test_episodes = 1  # 測試回合數
        
        # 創建測試環境
        self.env = HexapodTestEnv(max_episode_steps=test_steps, sequence_length=50)
        
        # 模型相關
        self.model = None
        self.model_loaded = False
        
        print(f"📊 測試設定：{self.test_steps} 步數，{self.test_episodes} 回合")

    def load_model(self, model_path):
        """載入訓練好的模型"""
        print(f"📥 載入模型: {model_path}")
        
        try:
            # 檢查模型檔案是否存在
            if not os.path.exists(f"{model_path}.zip"):
                print(f"❌ 模型檔案不存在: {model_path}.zip")
                return False
            
            # 載入模型
            self.model = PPO.load(model_path)
            self.model_loaded = True
            
            print("✅ 模型載入成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False

    def find_latest_model(self, models_dir="./models"):
        """尋找最新的訓練模型"""
        print(f"🔍 搜尋最新模型於: {models_dir}")
        
        if not os.path.exists(models_dir):
            print(f"❌ 模型資料夾不存在: {models_dir}")
            return None
        
        # 尋找所有訓練資料夾
        training_folders = []
        for item in os.listdir(models_dir):
            if item.startswith("training_") and os.path.isdir(os.path.join(models_dir, item)):
                training_folders.append(item)
        
        if not training_folders:
            print("❌ 找不到任何訓練資料夾")
            return None
        
        # 排序並取最新的
        training_folders.sort()
        latest_folder = training_folders[-1]
        folder_path = os.path.join(models_dir, latest_folder)
        
        print(f"📁 最新訓練資料夾: {latest_folder}")
        
        # 尋找資料夾中的模型檔案
        model_files = []
        for file in os.listdir(folder_path):
            if file.endswith(".zip"):
                model_files.append(file)
        
        if not model_files:
            print("❌ 訓練資料夾中找不到模型檔案")
            return None
        
        # 尋找最終模型或最新的檢查點
        if "ppo_hexapod_final.zip" in model_files:
            latest_model = "ppo_hexapod_final"
        else:
            # 按檔名排序，取最新的
            model_files.sort()
            latest_model = model_files[-1].replace(".zip", "")
        
        model_path = os.path.join(folder_path, latest_model)
        print(f"🎯 找到模型: {model_path}")
        
        return model_path

    def test_model(self):
        """執行模型測試"""
        if not self.model_loaded:
            print("❌ 請先載入模型")
            return
        
        print("🚀 開始模型測試...")
        print("-" * 50)
        
        total_rewards = []
        total_steps_completed = []
        
        for episode in range(self.test_episodes):
            print(f"📊 回合 {episode + 1}/{self.test_episodes}")
            
            # 重置環境
            obs, info = self.env.reset()
            episode_reward = 0
            steps_completed = 0
            
            for step in range(self.test_steps):
                # 使用模型預測動作
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 執行動作
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                steps_completed += 1
                
                # 每100步顯示進度
                if (step + 1) % 100 == 0:
                    roll = info['roll']
                    pitch = info['pitch']
                    print(f"  步數: {step + 1:4d}, 獎勵: {reward:6.3f}, "
                          f"姿態: roll={roll:6.3f}, pitch={pitch:6.3f}")
                
                # 如果episode結束
                if terminated or truncated:
                    reason = info.get('reason', '未知')
                    print(f"  ⚠️  Episode提前結束: {reason}")
                    break
            
            total_rewards.append(episode_reward)
            total_steps_completed.append(steps_completed)
            
            print(f"  ✅ 回合完成 - 總獎勵: {episode_reward:.3f}, 完成步數: {steps_completed}")
            print()
        
        # 顯示測試總結
        self._show_test_summary(total_rewards, total_steps_completed)

    def _show_test_summary(self, rewards, steps):
        """顯示測試總結"""
        print("=" * 50)
        print("📋 測試總結")
        print("=" * 50)
        
        avg_reward = np.mean(rewards)
        avg_steps = np.mean(steps)
        
        print(f"🎯 測試回合數: {len(rewards)}")
        print(f"📊 平均獎勵: {avg_reward:.3f}")
        print(f"⏱️  平均完成步數: {avg_steps:.1f}")
        print(f"📈 獎勵範圍: {min(rewards):.3f} ~ {max(rewards):.3f}")
        print(f"🏃 步數範圍: {min(steps)} ~ {max(steps)}")
        
        # 簡單的表現評估
        if avg_reward > 0.5:
            print("✅ 模型表現良好")
        elif avg_reward > 0:
            print("⚠️  模型表現普通")
        else:
            print("❌ 模型需要改進")
        
        print("=" * 50)

    def run_test(self, model_path=None):
        """執行完整測試流程"""
        try:
            # 載入模型
            if model_path:
                success = self.load_model(model_path)
            else:
                # 自動尋找最新模型
                latest_model_path = self.find_latest_model()
                if latest_model_path:
                    success = self.load_model(latest_model_path)
                else:
                    print("❌ 找不到可用的模型")
                    return
            
            if not success:
                print("❌ 模型載入失敗")
                return
            
            # 執行測試
            self.test_model()
            
        except KeyboardInterrupt:
            print("\n⏹️  測試被用戶中斷")
        except Exception as e:
            print(f"❌ 測試過程中發生錯誤: {e}")
        finally:
            # 清理資源
            self.env.close()
            print("🧹 資源清理完成")


def main():
    """
    測試程式主入口點
    """
    print("=" * 60)
    print("🧪 六足機器人 PPO + Transformer 模型測試程式")
    print("=" * 60)
    
    # 測試設定 - 可自由調整
    TEST_STEPS = 2000  # 📊 測試步數（可修改）
    
    try:
        # 創建測試器
        tester = HexapodTester(test_steps=TEST_STEPS)
        
        # 執行測試（自動尋找最新模型）
        tester.run_test()
        
        # 如果要測試特定模型，可以指定路徑：
        # tester.run_test(model_path="./models/training_001/ppo_hexapod_final")
        
    except Exception as e:
        print(f"❌ 測試程式運行失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("👋 測試程式結束")


# 程式入口點
if __name__ == "__main__":
    print("🧪 運行模式: 模型測試")
    main()