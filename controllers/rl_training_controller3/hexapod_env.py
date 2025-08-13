import sys
import time
import math
import numpy as np
from collections import deque
from controller import Supervisor

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    sys.exit('請安裝gymnasium: pip install gymnasium')


class HexapodBalanceEnv(Supervisor, gym.Env):
    """
    六足機器人平衡強化學習環境
    整合CPG控制器與Transformer+PPO訓練
    """
    
    def __init__(self, max_episode_steps=2000, sequence_length=50):
        super().__init__()
        
        # 環境參數
        self.max_episode_steps = max_episode_steps
        self.sequence_length = sequence_length
        self.current_step = 0
        
        # CPG參數（來自原控制器）
        self.knee_clamp_positive = True
        self.use_knee_signal_for_ankle = True
        self.body_height_offset = 0.5
        self.control_start_step = 100
        
        self.spec = type('SimpleSpec', (), {'id': 'HexapodBalance-v0','max_episode_steps':max_episode_steps})()
        
        # 狀態和動作空間定義
        self._setup_spaces()
        
        # Webots設置
        self.timestep = int(self.getBasicTimeStep())
        self.motors = {}
        self.gps_device = None
        self.imu_device = None
        self.platform_motor_joint = None
        
        # 狀態序列緩存（用於Transformer）
        self.state_sequence = deque(maxlen=sequence_length)
        self.action_sequence = deque(maxlen=sequence_length)
        self.reward_sequence = deque(maxlen=sequence_length)
        
        # 初始化設備
        self._init_devices()
        
        print("✅ 六足機器人平衡環境已初始化")
        print(f"📊 狀態空間: {self.observation_space}")
        print(f"🎮 動作空間: {self.action_space}")
        print(f"⏱️  控制頻率: {1000/self.timestep} Hz")

    def _setup_spaces(self):
        """設置狀態和動作空間"""
        # 動作空間：6個膝關節的修正量 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # 狀態空間：6個腳的方向分量
        # 由於我們使用單環境，實際的observation就是單步的狀態
        state_dim = 6  # 六個腳的方向分量
        
        # 單步狀態範圍 [-4, 4]（考慮到理論最大值3.332，加安全餘量）
        self.observation_space = spaces.Box(
            low=-4.0, 
            high=4.0, 
            shape=(state_dim,), 
            dtype=np.float32
        )

    def _init_devices(self):
        """初始化Webots設備"""
        try:
            # 初始化馬達（使用原控制器的對應關係）
            self._init_motors()
            
            # 初始化GPS
            self._init_gps()
            
            # 初始化IMU
            self._init_imu()
            
            # 初始化平台馬達
            self._init_platform_motor()
            
        except Exception as e:
            print(f"❌ 設備初始化失敗: {e}")
            raise

    def _init_motors(self):
        """初始化機器人馬達"""
        # 修改後的對應表（來自原控制器）
        leg_mapping = {
            1: ('R0', '右前腿'),
            2: ('R1', '右中腿'), 
            3: ('R2', '右後腿'),
            4: ('L2', '左後腿'),
            5: ('L1', '左中腿'),
            6: ('L0', '左前腿')
        }
        
        joint_names = ['0', '1', '2']  # 髖關節、膝關節、踝關節
        
        for leg_idx in range(1, 7):  # 6條腿
            if leg_idx not in leg_mapping:
                continue
                
            leg_name, _ = leg_mapping[leg_idx]
            self.motors[leg_idx] = {}
            
            for j, joint_name in enumerate(joint_names):
                joint_idx = j + 1  # CPG系統中關節索引從1開始
                motor_name = f"{leg_name}{joint_name}"
                
                try:
                    motor = self.getDevice(motor_name)
                    if motor is None:
                        print(f"⚠️  找不到馬達 {motor_name}")
                        continue
                    
                    # 設定馬達為位置控制模式
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
                print("✅ GPS感測器已啟用")
        except Exception as e:
            print(f"❌ GPS初始化失敗: {e}")

    def _init_imu(self):
        """初始化IMU感測器"""
        try:
            self.imu_device = self.getDevice("inertialunit1")
            if self.imu_device:
                self.imu_device.enable(self.timestep)
                print("✅ IMU感測器已啟用")
        except Exception as e:
            print(f"❌ IMU初始化失敗: {e}")

    def _init_platform_motor(self):
        """初始化平台馬達"""
        try:
            # 獲取experimental_platform節點
            platform_node = self.getFromDef("experimental_platform")
            if platform_node is None:
                print("⚠️  找不到 'experimental_platform' 節點")
                return
            
            # 獲取platform_motor的HingeJoint節點
            children_field = platform_node.getField("children")
            children_count = children_field.getCount()
            
            for i in range(children_count):
                child = children_field.getMFNode(i)
                if child and child.getDef() == "platform_motor":
                    self.platform_motor_joint = child
                    break
            
            if self.platform_motor_joint:
                print("✅ 平台馬達已連接")
            else:
                print("⚠️  找不到平台馬達")
                
        except Exception as e:
            print(f"❌ 平台馬達初始化失敗: {e}")

    def _get_imu_data(self):
        """讀取IMU數據"""
        try:
            if self.imu_device:
                roll_pitch_yaw = self.imu_device.getRollPitchYaw()
                return roll_pitch_yaw[0], roll_pitch_yaw[1]  # roll, pitch
                
            else:
                return 0.0, 0.0
        except Exception as e:
            print(f"讀取IMU數據錯誤: {e}")
            return 0.0, 0.0

    def _get_gps_data(self):
        """讀取GPS數據"""
        try:
            if self.gps_device:
                position = self.gps_device.getValues()
                return position[0], position[1], position[2]  # x, y, z
            else:
                return 0.0, 0.0, 0.0
        except Exception as e:
            print(f"讀取GPS數據錯誤: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_state(self):
        """計算6個腳的方向分量狀態"""
        roll, pitch = self._get_imu_data()
        
        # 計算六個腳的方向分量（來自小目標文件）
        sqrt_half = math.sqrt(0.5)
        
        states = np.array([
            (pitch + roll) * sqrt_half,    # e¹ - 前右腳
            roll,                          # e² - 右中腳  
            (-pitch + roll) * sqrt_half,   # e³ - 後右腳
            (-pitch - roll) * sqrt_half,   # e⁴ - 後左腳
            -roll,                         # e⁵ - 左中腳
            (pitch - roll) * sqrt_half     # e⁶ - 前左腳
        ], dtype=np.float32)
        
        return states

    def _calculate_reward(self):
        """計算獎勵函數"""
        roll, pitch = self._get_imu_data()
        x, y, z = self._get_gps_data()
        
        # 穩定性獎勵
        stability_term = (abs(pitch) + abs(roll)) / 2
        r_s = math.exp(-(stability_term ** 2) / (0.1 ** 2))
        
        p = 0
        # 跌倒懲罰
        if abs(pitch) >= 0.785 or abs(roll) >= 0.785:
            p += -1
        
        # 邊界懲罰
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            p += -1
        
        total_reward = r_s + p
        
        return total_reward, r_s, p

    def _is_done(self):
        """檢查episode是否結束"""
        roll, pitch = self._get_imu_data()
        x, y, z = self._get_gps_data()
        
        # 跌倒檢測
        if abs(pitch) >= 0.785 or abs(roll) >= 0.785:
            return True, True, "跌倒"  # terminated=True, truncated=True
        
        # 邊界檢測
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            return True, True, "出界"  # terminated=True, truncated=True
        
        # 時間限制
        if self.current_step >= self.max_episode_steps:
            return False, True, "超時"  # terminated=False, truncated=True
        
        return False, False, ""  # terminated=False, truncated=False

    def _control_platform(self):
        """控制平台進行正弦波運動"""
        if not self.platform_motor_joint:
            return
        
        try:
            current_time = self.getTime()
            # 平台運動: angle = 0.2 * sin(1.0 * π * t)
            target_angle = 0.2 * math.sin(1.0 * math.pi * current_time)
            
            # 設置平台角度
            joint_params_field = self.platform_motor_joint.getField("jointParameters")
            joint_params_node = joint_params_field.getSFNode()
            if joint_params_node:
                position_field = joint_params_node.getField("position")
                if position_field:
                    position_field.setSFFloat(target_angle)
                    
        except Exception as e:
            print(f"平台控制錯誤: {e}")

    def _apply_actions(self, rl_corrections):
        """應用動作到機器人（整合CPG基礎控制）
        
        訊號流程：
        1. 基礎訊號(0) → 完整處理流程 → 處理後訊號
        2. 處理後訊號 + RL修正量 → 馬達（僅膝關節）
        """
        # 儲存處理過的訊號
        processed_signals = {}
        for leg_idx in range(1, 7):
            processed_signals[leg_idx] = {}
        
        for leg_idx in range(1, 7):
            if leg_idx not in self.motors:
                continue
            
            for joint_idx in range(1, 4):  # 3個關節
                if joint_idx not in self.motors[leg_idx]:
                    continue
                
                # 步驟1: 基礎訊號（所有關節都從0開始）
                motor_angle = 0.0
                
                # 步驟2: 踝關節訊號替換
                motor_angle = self._replace_ankle_with_knee_signal(motor_angle, leg_idx, joint_idx, processed_signals)
                
                # 步驟3: 特殊關節處理
                motor_angle = self._process_special_joints(motor_angle, leg_idx, joint_idx)
                
                # 步驟4: 訊號方向調整
                motor_angle = self._adjust_signal_direction(motor_angle, leg_idx, joint_idx)
                
                # 步驟5: 高度偏移
                motor_angle = self._apply_height_offset(motor_angle, leg_idx, joint_idx)
                
                # 步驟6: 加入RL修正量（僅膝關節）
                if joint_idx == 2:  # 膝關節
                    final_motor_angle = motor_angle + rl_corrections[leg_idx - 1]
                else:
                    final_motor_angle = motor_angle
                
                # 儲存處理過的訊號（用於踝關節引用）
                processed_signals[leg_idx][joint_idx] = final_motor_angle  # 注意：這裡儲存的是加RL修正後的訊號
                
                # 發送到馬達
                try:
                    if self.current_step >= self.control_start_step:
                        limited_angle = max(-1.0, min(1.0, final_motor_angle))
                        self.motors[leg_idx][joint_idx].setPosition(limited_angle)
                    else:
                        # 初始階段的處理（不加RL修正）
                        init_angle = self._replace_ankle_with_knee_signal(0.0, leg_idx, joint_idx, {})
                        init_angle = self._apply_height_offset(init_angle, leg_idx, joint_idx)
                        self.motors[leg_idx][joint_idx].setPosition(init_angle)
                except Exception as e:
                    print(f"設定馬達角度錯誤 (腿{leg_idx}, 關節{joint_idx}): {e}")

    def _replace_ankle_with_knee_signal(self, motor_angle, leg_idx, joint_idx, processed_signals):
        """將踝關節訊號替換為同隻腳膝關節處理後的訊號"""
        if joint_idx == 3 and self.use_knee_signal_for_ankle:
            # 使用同隻腳膝關節的處理過訊號（包含RL修正）
            if leg_idx in processed_signals and 2 in processed_signals[leg_idx]:
                knee_signal = processed_signals[leg_idx][2]
                return knee_signal * 1
            else:
                # 如果還沒處理膝關節，先計算膝關節的處理後訊號
                knee_angle = 0.0
                # 這裡不能直接遞歸，需要手動計算膝關節的處理流程
                knee_angle = self._process_special_joints(knee_angle, leg_idx, 2)
                knee_angle = self._adjust_signal_direction(knee_angle, leg_idx, 2)
                knee_angle = self._apply_height_offset(knee_angle, leg_idx, 2)
                return knee_angle * 1
        return motor_angle

    def _process_special_joints(self, motor_angle, leg_idx, joint_idx):
        """處理特殊關節"""
        # 膝關節正值限制
        if joint_idx == 2 and self.knee_clamp_positive and motor_angle <= 0:
            return 0.0
        
        # 特定踝關節處理
        if joint_idx == 3 and not self.use_knee_signal_for_ankle:
            if (leg_idx == 2 or leg_idx == 5) and motor_angle >= 0:
                return 0.0
        
        return motor_angle

    def _adjust_signal_direction(self, motor_angle, leg_idx, joint_idx):
        """調整訊號方向"""
        # 右側腿部踝關節反向
        if not self.use_knee_signal_for_ankle and leg_idx <= 3 and joint_idx == 3:
            motor_angle = -motor_angle
        
        # 左側腿部髖關節和膝關節反向
        if leg_idx >= 4 and (joint_idx == 1 or joint_idx == 2):
            motor_angle = -motor_angle
        
        # 額外的反向處理
        if not self.use_knee_signal_for_ankle and joint_idx == 3:
            if leg_idx in [1, 6, 2, 5]:
                motor_angle = -motor_angle
        
        return motor_angle

    def _apply_height_offset(self, motor_angle, leg_idx, joint_idx):
        """應用機身高度偏移"""
        should_apply_offset = (
            joint_idx == 2 or  # 膝關節
            (joint_idx == 3 and not self.use_knee_signal_for_ankle)  # 踝關節(條件性)
        )
        
        if should_apply_offset:
            # 右側腿部用負偏移，左側腿部用正偏移
            offset_direction = -1 if leg_idx <= 3 else 1
            return motor_angle + (self.body_height_offset * offset_direction)
        
        return motor_angle

    def _update_sequences(self, state, action, reward):
        """更新狀態-動作-獎勵序列"""
        self.state_sequence.append(state.copy())
        if action is not None:
            self.action_sequence.append(action.copy())
        else:
            self.action_sequence.append(np.zeros(6))
        self.reward_sequence.append(reward)

    def get_sequence_data(self):
        """獲取序列數據（供Transformer使用）"""
        # 確保序列長度一致
        states = np.array(list(self.state_sequence))
        actions = np.array(list(self.action_sequence))
        rewards = np.array(list(self.reward_sequence))
        
        # 如果序列不夠長，用零填充
        if len(states) < self.sequence_length:
            pad_length = self.sequence_length - len(states)
            state_pad = np.zeros((pad_length, states.shape[1])) if len(states) > 0 else np.zeros((pad_length, 6))
            action_pad = np.zeros((pad_length, 6))
            reward_pad = np.zeros(pad_length)
            
            states = np.vstack([state_pad, states]) if len(states) > 0 else state_pad
            actions = np.vstack([action_pad, actions]) if len(actions) > 0 else action_pad
            rewards = np.concatenate([reward_pad, rewards]) if len(rewards) > 0 else reward_pad
        
        return {
            'states': states.astype(np.float32),
            'actions': actions.astype(np.float32),
            'rewards': rewards.astype(np.float32)
        }

    def reset(self, seed=None, options=None):
        """重置環境 - 符合新版 Gym 標準"""
        if seed is not None:
            np.random.seed(seed)
        
        print("🔄 重置環境...")
        
        # 重置模擬
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)
        
        # 重置計數器
        self.current_step = 0
        
        # 重新初始化設備
        self._init_devices()
        
        # 執行幾步以穩定系統
        for _ in range(3):
            super().step(self.timestep)
        
        # 獲取初始狀態
        initial_state = self._calculate_state()
        
        # 建立初始 info
        info = {
            'step': self.current_step,
            'imu_data': self._get_imu_data(),
            'gps_data': self._get_gps_data(),
            'stability_reward': 0.0,
            'penalty': 0.0,
            'reason': ''
        }
        
        print(f"✅ 環境重置完成，初始狀態: {initial_state}")
        
        # 確保返回 (observation, info) 格式
        return initial_state, info

    def step(self, action):
        """執行一步動作 - 符合新版 Gym 標準"""
        # 確保動作格式正確
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 控制平台運動
        self._control_platform()
        
        # 應用動作到機器人
        self._apply_actions(action)
        
        # 執行物理步驟
        super().step(self.timestep)
        
        # 更新步數
        self.current_step += 1
        
        # 獲取新狀態
        new_state = self._calculate_state()
        
        # 計算獎勵
        reward, stability_reward, penalty = self._calculate_reward()
        
        # 檢查是否結束 - 注意這裡返回 3 個值
        terminated, truncated, reason = self._is_done()
        
        # 建立info字典
        info = {
            'stability_reward': float(stability_reward),
            'penalty': float(penalty),
            'step': self.current_step,
            'reason': reason,
            'imu_data': self._get_imu_data(),
            'gps_data': self._get_gps_data()
        }
        
        # 每100步打印進度
        if self.current_step % 100 == 0:
            roll, pitch = self._get_imu_data()
            x, y, z = self._get_gps_data()
            print(f"步數: {self.current_step}, 獎勵: {reward:.3f}, "
                f"姿態: roll={roll:.3f}, pitch={pitch:.3f}, "
                f"位置: x={x:.3f}, y={y:.3f}")
        
        # 確保返回值都是正確的類型和數量 (5個值)
        return (
            new_state,           # observation
            float(reward),       # reward  
            terminated,          # terminated
            truncated,           # truncated
            info                 # info
        )

    def close(self):
        """關閉環境"""
        print("👋 關閉環境...")
        # Webots會自動處理清理工作