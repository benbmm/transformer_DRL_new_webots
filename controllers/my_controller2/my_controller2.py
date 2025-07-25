import sys
import time
import math
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from collections import deque

from controller import Robot, Motor

class Oscillator:
    """振盪器類別，儲存單個振盪器的所有狀態變數"""
    def __init__(self, max_steps):
        self.dUe = np.zeros(max_steps + 1)
        self.dUf = np.zeros(max_steps + 1)
        self.dVe = np.zeros(max_steps + 1)
        self.dVf = np.zeros(max_steps + 1)
        self.Ue = np.zeros(max_steps + 1)
        self.Uf = np.zeros(max_steps + 1)
        self.Ve = np.zeros(max_steps + 1)
        self.Vf = np.zeros(max_steps + 1)
        self.Ye = np.zeros(max_steps + 1)
        self.Yf = np.zeros(max_steps + 1)
        self.Y = np.zeros(max_steps + 1)

class CPG:
    """CPG類別，包含多個振盪器"""
    def __init__(self, max_steps, num_oscillators):
        # 為每個振盪器建立獨立的實例（索引從1開始）
        self.osc = [Oscillator(max_steps) for _ in range(num_oscillators + 1)]

class HexapodTransformer(nn.Module):
    """六足機器人地形適應Transformer"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用GPT2架構作為backbone
        transformer_config = GPT2Config(
            vocab_size=1,  # 不使用詞彙表
            n_positions=config['sequence_length'],
            n_embd=config['hidden_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head']
        )
        
        self.transformer = GPT2Model(transformer_config)
        
        # 輸入投影層 (state + action + reward)
        input_dim = config['state_dim'] + config['action_dim'] + config['reward_dim']
        self.input_projection = nn.Linear(input_dim, config['hidden_size'])
        
        # 輸出投影層 (12維修正量)
        self.output_projection = nn.Linear(config['hidden_size'], config['action_dim'])
        
        # 修正量限制
        self.max_correction = config.get('max_correction', 0.6)
        
    def forward(self, state_sequence, action_sequence, reward_sequence):
        """
        Args:
            state_sequence: [batch, seq_len, state_dim] - IMU數據
            action_sequence: [batch, seq_len, action_dim] - 之前的動作
            reward_sequence: [batch, seq_len, reward_dim] - 之前的獎勵
        Returns:
            corrections: [batch, action_dim] - 修正量
        """
        batch_size, seq_len = state_sequence.shape[:2]
        
        # 組合輸入
        combined_input = torch.cat([state_sequence, action_sequence, reward_sequence], dim=-1)
        
        # 輸入投影
        embeddings = self.input_projection(combined_input)
        
        # Transformer處理
        transformer_outputs = self.transformer(inputs_embeds=embeddings)
        last_hidden = transformer_outputs.last_hidden_state[:, -1, :]  # 取最後一個時間步
        
        # 輸出投影並限制範圍
        corrections = self.output_projection(last_hidden)
        corrections = torch.tanh(corrections) * self.max_correction
        
        return corrections

class HexapodController:
    """六足機器人控制器"""
    
    def __init__(self):
        # CPG參數
        self.MAX_STEPS = 2000
        self.NUM_LEGS = 6
        self.NUM_OSCILLATORS = 3
        self.STEP_SIZE = 0.2
        self.WFE = -1.5
        self.T1 = 0.5
        self.T2 = 7.5
        self.U0 = 1.3
        self.B = 3.0
        self.WIJ = -1.0
        
        # 控制訊號參數
        self.knee_clamp_positive = True
        self.use_knee_signal_for_ankle = True
        self.body_height_offset = 0.5
        
        # 其他參數
        self.rounding_digits = -1
        self.control_start_step = 100
        
        # 初始化Webots機器人
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # 初始化CPG系統
        self.legs = [CPG(self.MAX_STEPS, self.NUM_OSCILLATORS) for _ in range(self.NUM_LEGS + 1)]
        self.current_step = 1
        
        # 儲存處理過的訊號
        self.processed_signals = {}
        for leg_idx in range(1, self.NUM_LEGS + 1):
            self.processed_signals[leg_idx] = {}
            for joint_idx in range(1, self.NUM_OSCILLATORS + 1):
                self.processed_signals[leg_idx][joint_idx] = np.zeros(self.MAX_STEPS + 1)
        
        # 初始化馬達
        self.motors = {}
        self.init_motors()
        
        # 初始化IMU感測器
        self.init_imu()
        
        # 初始化GPS感測器
        self.init_gps()
        
        # 初始化CPG系統
        self.initialize_cpg_system()

        # Transformer配置
        self.transformer_config = {
            'sequence_length': 50,
            'state_dim': 6,
            'action_dim': 12,
            'reward_dim': 1,
            'hidden_size': 128,
            'n_layer': 3,
            'n_head': 2,
            'max_correction': 0.6
        }
        
        # 初始化Transformer模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer_model = HexapodTransformer(self.transformer_config).to(self.device)
        
        # 序列緩存
        self.state_buffer = deque(maxlen=self.transformer_config['sequence_length'])
        self.action_buffer = deque(maxlen=self.transformer_config['sequence_length'])
        self.reward_buffer = deque(maxlen=self.transformer_config['sequence_length'])
        
        # 初始化緩存
        self._initialize_buffers()
        
        # Transformer相關參數
        self.use_transformer = True
        self.transformer_start_step = 200

        # 新增：獎勵和終止條件相關變數
        self.initial_yaw = None  # 記錄初始方向
        self.start_position = None  # 記錄起始位置
        self.position_history_200 = deque(maxlen=200)  # 200步位置歷史
        self.episode_terminated = False  # 是否已終止
        self.termination_reason = None  # 終止原因

        print("Transformer模組已初始化")

        # 建立儲存資料夾
        self.create_output_directories()

        print("六足機器人CPG控制器已初始化")
        print(f"控制頻率: {1000/self.timestep} Hz")
        print(f"使用設備: {self.device}")

    def init_motors(self):
        """初始化所有馬達"""
        leg_mapping = {
            1: ('R0', '右前腿'),
            2: ('R1', '右中腿'), 
            3: ('R2', '右後腿'),
            4: ('L2', '左後腿'),
            5: ('L1', '左中腿'),
            6: ('L0', '左前腿')
        }
        
        joint_names = ['0', '1', '2']
        joint_descriptions = ['髖關節', '膝關節', '踝關節']
        
        print("=== 馬達初始化對應關係 ===")
        
        for leg_idx in range(1, self.NUM_LEGS + 1):
            if leg_idx not in leg_mapping:
                continue
                
            leg_name, leg_desc = leg_mapping[leg_idx]
            self.motors[leg_idx] = {}
            
            print(f"CPG leg_idx={leg_idx} -> {leg_name} ({leg_desc})")
            
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
                    print(f"  ✓  joint_idx={joint_idx} ({joint_desc}) -> 馬達 {motor_name}")
                    
                except Exception as e:
                    print(f"  ❌ 初始化馬達 {motor_name} 時發生錯誤: {e}")
        
        print("=========================")
    
    def init_imu(self):
        """初始化IMU感測器"""
        try:
            self.imu_device = self.robot.getDevice("inertialunit1")
            if self.imu_device is None:
                print("❌ 找不到IMU感測器 'inertialunit1'")
                self.imu_device = None
                return
            
            self.imu_device.enable(self.timestep)
            print("✅ IMU感測器已啟用")
            
        except Exception as e:
            print(f"❌ 初始化IMU感測器時發生錯誤: {e}")
            self.imu_device = None
    
    def init_gps(self):
        """初始化GPS感測器"""
        try:
            self.gps_device = self.robot.getDevice("gps")
            if self.gps_device is None:
                print("❌ 找不到GPS感測器 'gps'")
                for gps_name in ["GPS", "position_sensor", "supervisor"]:
                    self.gps_device = self.robot.getDevice(gps_name)
                    if self.gps_device is not None:
                        print(f"✅ 找到GPS感測器: {gps_name}")
                        break
                
                if self.gps_device is None:
                    print("⚠️ 未找到GPS感測器，將使用Supervisor節點獲取位置")
                    self.gps_device = None
                    return
            
            self.gps_device.enable(self.timestep)
            print("✅ GPS感測器已啟用")
            
        except Exception as e:
            print(f"❌ 初始化GPS感測器時發生錯誤: {e}")
            self.gps_device = None
    
    def _initialize_buffers(self):
        """初始化序列緩存"""
        seq_len = self.transformer_config['sequence_length']
        
        for _ in range(seq_len):
            self.state_buffer.append(np.zeros(6))  # 6維腳部方向分量
            self.action_buffer.append(np.zeros(12))
            self.reward_buffer.append(np.zeros(1))
        
        print(f"已初始化序列緩存，長度: {seq_len}")
    
    def create_output_directories(self):
        """建立輸出檔案的資料夾"""
        self.original_output_dir = "original_cpg_outputs"
        self.processed_output_dir = "processed_signals"
        
        try:
            if not os.path.exists(self.original_output_dir):
                os.makedirs(self.original_output_dir)
                print(f"✅ 建立資料夾: {self.original_output_dir}")
            
            if not os.path.exists(self.processed_output_dir):
                os.makedirs(self.processed_output_dir)
                print(f"✅ 建立資料夾: {self.processed_output_dir}")
                
        except Exception as e:
            print(f"❌ 建立資料夾時發生錯誤: {e}")
    
    def initialize_cpg_system(self):
        """初始化CPG系統"""
        uf_values = [
            0.01, 0.02, 0.03, 0.05, 0.06,
            0.07, 0.09, 0.10, 0.11, 
            0.13, 0.14, 0.15, 0.17, 0.18,
            0.19, 0.21, 0.22, 0.23
        ]
        
        vf_values = [
            0.025, 0.035, 0.045, 0.065, 0.075,
            0.085, 0.105, 0.115, 0.125,
            0.145, 0.155, 0.165, 0.185, 0.195,
            0.205, 0.225, 0.235, 0.245
        ]   
        
        index = 0
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for osc_idx in range(1, self.NUM_OSCILLATORS + 1):
                self.legs[leg_idx].osc[osc_idx].Ue[1] = 0.0
                self.legs[leg_idx].osc[osc_idx].Ve[1] = 0.0
                self.legs[leg_idx].osc[osc_idx].Uf[1] = uf_values[index]
                self.legs[leg_idx].osc[osc_idx].Vf[1] = vf_values[index]
                index += 1
    
    def get_imu_data(self):
        """讀取IMU數據並轉換為6維腳部方向分量"""
        try:
            if not hasattr(self, 'imu_device') or self.imu_device is None:
                return np.zeros(6)
            
            roll_pitch_yaw = self.imu_device.getRollPitchYaw()
            roll, pitch, yaw = roll_pitch_yaw
            
            # 轉換為6維腳部方向分量
            sqrt_half = np.sqrt(0.5)
            
            e1 = (pitch + roll) * sqrt_half    # 前右腳
            e2 = roll                          # 右中腳
            e3 = (-pitch + roll) * sqrt_half   # 後右腳
            e4 = (-pitch - roll) * sqrt_half   # 後左腳
            e5 = -roll                         # 左中腳
            e6 = (pitch - roll) * sqrt_half    # 前左腳
            
            return np.array([e1, e2, e3, e4, e5, e6])
        except Exception as e:
            if self.current_step % 100 == 0:
                print(f"讀取IMU數據錯誤: {e}")
            return np.zeros(6)
        
    def get_raw_imu_data(self):
        """讀取原始IMU歐拉角數據（用於獎勵計算和終止條件）"""
        try:
            if not hasattr(self, 'imu_device') or self.imu_device is None:
                return np.zeros(3)
            
            roll_pitch_yaw = self.imu_device.getRollPitchYaw()
            return np.array(roll_pitch_yaw)
        except Exception as e:
            if self.current_step % 100 == 0:
                print(f"讀取原始IMU數據錯誤: {e}")
            return np.zeros(3)
    
    def get_position_data(self):
        """讀取位置數據"""
        try:
            if self.gps_device is not None:
                position = self.gps_device.getValues()
                return np.array(position[:3])
            else:
                try:
                    supervisor = self.robot.getSelf()
                    if supervisor is not None:
                        position = supervisor.getPosition()
                        return np.array(position)
                except:
                    pass
                
            return np.zeros(3)
        except Exception as e:
            if self.current_step % 100 == 0:
                print(f"讀取位置數據錯誤: {e}")
            return np.zeros(3)
    
    def update_position_tracking(self):
        """更新速度計算"""
        current_position = self.get_position_data()
        
        # 記錄起始位置和初始yaw
        if self.start_position is None:
            self.start_position = current_position.copy()
        
        if self.initial_yaw is None:
            raw_imu = self.get_raw_imu_data()
            self.initial_yaw = raw_imu[2]  # yaw
        
        # 更新200步位置歷史
        self.position_history_200.append(current_position.copy())
    
    def get_average_speed(self):
        """獲取平均速度（基於序列長度的位置變化）"""
        if len(self.position_history_200) < 2:
            return 0.0
        
        # 使用n個step的位置變化計算速度
        seq_len = self.transformer_config['sequence_length']
        if len(self.position_history_200) >= seq_len:
            start_pos = list(self.position_history_200)[-seq_len]
            end_pos = list(self.position_history_200)[-1]
            
            displacement = end_pos - start_pos
            distance = np.linalg.norm(displacement[:2])  # 只考慮x,y平面距離
            
            time_interval = seq_len * self.timestep / 1000.0  # n*0.02秒
            speed = distance / time_interval if time_interval > 0 else 0.0
            return speed
        else:
            # 如果位置歷史不足，返回0速度
            return 0.0
    
    def calculate_direction_reward_and_penalty(self, raw_imu_data):
        """計算方向獎勵和懲罰（指數版本）"""
        roll, pitch, yaw = raw_imu_data
        
        # 計算與初始方向的偏差
        if self.initial_yaw is not None:
            theta = abs(yaw - self.initial_yaw)
            # 處理角度跨越π的情況
            if theta > np.pi:
                theta = 2 * np.pi - theta
        else:
            theta = abs(yaw)
        
        # 方向獎勵：指數函數
        r_theta = np.exp(-(theta**2) / (0.3**2))
        
        # 偏向懲罰：角度偏差 >= 0.785 rad 時
        p_theta = -1.0 if theta >= 0.785 else 0.0
        
        return r_theta, p_theta, theta
    
    def calculate_speed_reward_and_penalty(self):
        """計算速度獎勵和懲罰（指數版本）"""
        current_speed = self.get_average_speed()
        
        # 速度獎勵：v_max = 1
        r_v = min(max(current_speed, 0) / 1.0, 1.0)
        
        # 緩慢懲罰：200步前進距離 < 0.05m
        p_v = 0.0
        if len(self.position_history_200) >= 200:
            start_pos = list(self.position_history_200)[0]
            current_pos = list(self.position_history_200)[-1]
            total_distance = np.linalg.norm((current_pos - start_pos)[:2])
            
            if total_distance < 0.05:
                p_v = -1.0
        
        return r_v, p_v, current_speed
    
    def calculate_stability_reward_and_penalty(self, raw_imu_data):
        """計算穩定性獎勵和懲罰（指數版本）"""
        roll, pitch, yaw = raw_imu_data
        
        # 穩定性獎勵：指數函數
        stability_error = (abs(pitch) + abs(roll)) / 2.0
        r_s = np.exp(-(stability_error**2) / (0.1**2))
        
        # 跌倒懲罰：pitch or roll >= 0.785 rad
        p_s = -1.0 if (abs(pitch) >= 0.785 or abs(roll) >= 0.785) else 0.0
        
        return r_s, p_s, stability_error
    
    def calculate_control_reward(self, corrections):
        """計算控制量獎勵"""
        if corrections is None or len(corrections) == 0:
            return 1.0  # 沒有修正時給予最高獎勵
        
        # 計算修正量的平均絕對值
        avg_correction = np.mean(np.abs(corrections))
        
        # 控制量獎勵：指數函數
        r_c = np.exp(-(avg_correction**2) / (0.9**2))
        
        return r_c
    
    def calculate_reward(self, raw_imu_data, corrections=None):
        """計算綜合獎勵（完全按照target規格）"""
        # 計算各項獎勵和懲罰
        r_theta, p_theta, theta_error = self.calculate_direction_reward_and_penalty(raw_imu_data)
        r_v, p_v, current_speed = self.calculate_speed_reward_and_penalty()
        r_s, p_s, stability_error = self.calculate_stability_reward_and_penalty(raw_imu_data)
        r_c = self.calculate_control_reward(corrections)
        
        # 權重設定（按照target要求）
        w_s = 4   # 穩定性獎勵
        w_theta = 3   # 方向維持獎勵
        w_v = 2   # 速度獎勵
        w_c = 1   # 控制量獎勵
        
        # 總獎勵函數：R = w_θ*r_θ + p_θ + w_v*r_v + p_v + w_s*r_s + p_s + w_c*r_c
        total_reward = (w_theta * r_theta + p_theta + 
                       w_v * r_v + p_v + 
                       w_s * r_s + p_s + 
                       w_c * r_c)
        
        # 詳細日誌輸出
        if self.current_step % 100 == 0:
            print(f"指數獎勵詳細資訊 (步數 {self.current_step}):")
            print(f"  方向: r_θ={r_theta:.3f} (θ={theta_error:.4f}), p_θ={p_theta:.0f}")
            print(f"  速度: r_v={r_v:.3f} (v={current_speed:.3f}), p_v={p_v:.0f}")
            print(f"  穩定: r_s={r_s:.3f} (error={stability_error:.4f}), p_s={p_s:.0f}")
            print(f"  控制: r_c={r_c:.3f}")
            print(f"  總獎勵: {total_reward:.3f}")
            print(f"  原始角度 - Roll: {abs(raw_imu_data[0]):.4f}, Pitch: {abs(raw_imu_data[1]):.4f}, Yaw: {abs(raw_imu_data[2]):.4f}")
        
        return total_reward
    
    def check_termination_conditions(self, raw_imu_data):
        """檢查終止條件"""
        if self.episode_terminated:
            return True
        
        roll, pitch, yaw = raw_imu_data
        current_position = self.get_position_data()
        
        # 1. pitch or roll > 0.785 rad
        if abs(pitch) > 0.785 or abs(roll) > 0.785:
            self.episode_terminated = True
            self.termination_reason = f"跌倒終止 - Roll: {abs(roll):.4f}, Pitch: {abs(pitch):.4f}"
            return True
        
        # 2. 方向偏差 > 0.785 rad
        if self.initial_yaw is not None:
            theta = abs(yaw - self.initial_yaw)
            if theta > np.pi:
                theta = 2 * np.pi - theta
            if theta > 0.785:
                self.episode_terminated = True
                self.termination_reason = f"方向偏離終止 - 偏差: {theta:.4f} rad"
                return True
        
        # 3. 200步前進距離 < 0.05m
        if len(self.position_history_200) >= 200:
            start_pos = list(self.position_history_200)[0]
            total_distance = np.linalg.norm((current_position - start_pos)[:2])
            if total_distance < 0.05:
                self.episode_terminated = True
                self.termination_reason = f"移動緩慢終止 - 200步距離: {total_distance:.4f}m"
                return True
        
        # 4. 前進3.5m
        if self.start_position is not None:
            total_forward_distance = np.linalg.norm((current_position - self.start_position)[:2])
            if total_forward_distance >= 3.5:
                self.episode_terminated = True
                self.termination_reason = f"成功完成 - 前進距離: {total_forward_distance:.4f}m"
                return True
        
        # 5. 到達最大step=2000
        if self.current_step >= self.MAX_STEPS:
            self.episode_terminated = True
            self.termination_reason = f"達到最大步數: {self.MAX_STEPS}"
            return True
        
        return False
    
    def get_transformer_correction(self):
        """使用Transformer產生修正量"""
        if not self.use_transformer or self.current_step < self.transformer_start_step:
            return np.zeros(12)
        
        try:
            state_seq = torch.FloatTensor(list(self.state_buffer)).unsqueeze(0).to(self.device)
            action_seq = torch.FloatTensor(list(self.action_buffer)).unsqueeze(0).to(self.device)
            reward_seq = torch.FloatTensor(list(self.reward_buffer)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                corrections = self.transformer_model(state_seq, action_seq, reward_seq)
                corrections = corrections.cpu().numpy().flatten()
            
            return corrections
        
        except Exception as e:
            print(f"Transformer推理錯誤: {e}")
            return np.zeros(12)
    
    def clamp_positive(self, value):
        """限制為正值"""
        return max(0.0, value)
    
    def get_neighbor_index(self, current, offset):
        """取得相鄰腿部索引"""
        result = (current + offset) % self.NUM_LEGS
        return self.NUM_LEGS if result == 0 else result
    
    def update_oscillator(self, leg_idx, osc_idx, step):
        """更新單個振盪器的狀態"""
        leg = self.legs[leg_idx]
        osc = leg.osc[osc_idx]
        
        k = (osc_idx % 3) + 1
        kk = ((osc_idx + 1) % 3) + 1
        neighbor_oscs = [k, kk]
        
        neighbor_legs = [
            self.get_neighbor_index(leg_idx, 1),
            self.get_neighbor_index(leg_idx, 5)
        ]
        
        coupling_e = 0.0
        coupling_f = 0.0
        
        for neighbor_osc in neighbor_oscs:
            coupling_e += self.clamp_positive(leg.osc[neighbor_osc].Ye[step])
            coupling_f += self.clamp_positive(leg.osc[neighbor_osc].Yf[step])
        
        for neighbor_leg in neighbor_legs:
            coupling_e += self.clamp_positive(self.legs[neighbor_leg].osc[osc_idx].Ye[step])
            coupling_f += self.clamp_positive(self.legs[neighbor_leg].osc[osc_idx].Yf[step])
        
        osc.dUe[step] = (-osc.Ue[step] + self.WFE * osc.Yf[step] 
                     - self.B * osc.Ve[step] + self.U0 + self.WIJ * coupling_e) / self.T1
    
        osc.Ue[step + 1] = osc.Ue[step] + self.STEP_SIZE * osc.dUe[step]
        osc.Ye[step + 1] = self.clamp_positive(osc.Ue[step + 1])
        
        osc.dVe[step] = (-osc.Ve[step] + osc.Ye[step + 1]) / self.T2
        osc.Ve[step + 1] = osc.Ve[step] + self.STEP_SIZE * osc.dVe[step]
        
        osc.dUf[step] = (-osc.Uf[step] + self.WFE * osc.Ye[step] 
                     - self.B * osc.Vf[step] + self.U0 + self.WIJ * coupling_f) / self.T1
    
        osc.Uf[step + 1] = osc.Uf[step] + self.STEP_SIZE * osc.dUf[step]
        osc.Yf[step + 1] = self.clamp_positive(osc.Uf[step + 1])
        
        osc.dVf[step] = (-osc.Vf[step] + osc.Yf[step + 1]) / self.T2
        osc.Vf[step + 1] = osc.Vf[step] + self.STEP_SIZE * osc.dVf[step]
        
        if self.rounding_digits == -1:
            osc.Y[step] = osc.Yf[step] - osc.Ye[step]
        else:
            osc.Y[step] = round(osc.Yf[step] - osc.Ye[step], self.rounding_digits)
    
    def calculate_cpg_output(self, step):
        """計算所有CPG的輸出"""
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for osc_idx in range(1, self.NUM_OSCILLATORS + 1):
                self.update_oscillator(leg_idx, osc_idx, step)
    
    def process_special_joints(self, motor_angle, leg_idx, joint_idx):
        """處理特殊關節"""
        if joint_idx == 2:
            if self.knee_clamp_positive and motor_angle <= 0:
                return 0.0
        
        elif joint_idx == 3:
            if not self.use_knee_signal_for_ankle:
                if (leg_idx == 2 or leg_idx == 5) and motor_angle >= 0:
                    return 0.0
        
        return motor_angle
    
    def adjust_signal_direction(self, motor_angle, leg_idx, joint_idx):
        """調整訊號方向"""
        if not self.use_knee_signal_for_ankle:
            if leg_idx <= 3 and joint_idx == 3:
                motor_angle = -motor_angle
        
        if leg_idx >= 4 and (joint_idx == 1 or joint_idx == 2):
            motor_angle = -motor_angle
        
        if not self.use_knee_signal_for_ankle:
            if (leg_idx == 1 or leg_idx == 6) and joint_idx == 3:
                motor_angle = -motor_angle
            if (leg_idx == 2 or leg_idx == 5) and joint_idx == 3:
                motor_angle = -motor_angle

        return motor_angle
    
    def replace_ankle_with_knee_signal(self, motor_angle, leg_idx, joint_idx):
        """將踝關節訊號替換為膝關節訊號"""
        if joint_idx == 3 and self.use_knee_signal_for_ankle:
            knee_signal = self.processed_signals[leg_idx][2][self.current_step]
            return knee_signal * 1
        return motor_angle
    
    def apply_height_offset(self, motor_angle, leg_idx, joint_idx):
        """應用機身高度偏移"""
        should_apply_offset = (
            joint_idx == 2 or
            (joint_idx == 3 and not self.use_knee_signal_for_ankle)
        )
    
        if should_apply_offset:
            offset_direction = -1 if leg_idx <= 3 else 1
            return motor_angle + (self.body_height_offset * offset_direction)
        
        return motor_angle
    
    def apply_motor_commands(self):
        """將CPG+Transformer輸出應用到馬達"""
        step = self.current_step
        
        # 獲取IMU數據（6維腳部方向分量）
        imu_data = self.get_imu_data()
        self.state_buffer.append(imu_data)
        
        # 獲取原始IMU數據（3維歐拉角，用於獎勵計算）
        raw_imu_data = self.get_raw_imu_data()
        
        self.update_position_tracking()
        
        # 檢查終止條件
        if self.check_termination_conditions(raw_imu_data):
            return  # 如果已終止，不再執行馬達命令
        
        transformer_corrections = self.get_transformer_correction()
        
        current_actions = np.zeros(12)
        action_idx = 0
        
        for leg_idx in range(1, self.NUM_LEGS + 1):
            if leg_idx not in self.motors:
                continue
            
            for joint_idx in range(1, self.NUM_OSCILLATORS + 1):
                if joint_idx not in self.motors[leg_idx]:
                    continue
                
                motor_angle = self.legs[leg_idx].osc[joint_idx].Y[step]
                
                motor_angle = self.replace_ankle_with_knee_signal(motor_angle, leg_idx, joint_idx)
                motor_angle = self.process_special_joints(motor_angle, leg_idx, joint_idx)
                motor_angle = self.adjust_signal_direction(motor_angle, leg_idx, joint_idx)
                motor_angle = self.apply_height_offset(motor_angle, leg_idx, joint_idx)
                
                # 添加Transformer修正量（只對膝關節和踝關節）
                if joint_idx in [2, 3]:  # 膝關節和踝關節
                    correction_idx = (leg_idx - 1) * 2 + (joint_idx - 2)
                    if correction_idx < len(transformer_corrections):
                        motor_angle += transformer_corrections[correction_idx]
                        if self.current_step % 100 == 0:
                            print(f"步數{self.current_step}: 腿{leg_idx}關節{joint_idx}修正量: {transformer_corrections[correction_idx]:.4f}")
                
                # 儲存動作到緩存
                if joint_idx in [2, 3]:
                    current_actions[action_idx] = motor_angle
                    action_idx += 1
                
                # 儲存處理過的訊號
                self.processed_signals[leg_idx][joint_idx][step] = motor_angle
                
                # 發送到馬達
                try:
                    if step >= self.control_start_step:
                        self.motors[leg_idx][joint_idx].setPosition(motor_angle)
                except Exception as e:
                    print(f"設定馬達角度時發生錯誤 (腿{leg_idx}, 關節{joint_idx}): {e}")
        
        # 更新動作緩存
        self.action_buffer.append(current_actions)
        
        # 計算實際獎勵（使用原始IMU數據和修正量）
        current_reward = self.calculate_reward(raw_imu_data, transformer_corrections)
        self.reward_buffer.append(np.array([current_reward]))
    
    def save_cpg_outputs(self):
        """儲存原始CPG輸出到檔案"""
        print("\n💾 正在儲存原始CPG輸出...")
        
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for osc_idx in range(1, self.NUM_OSCILLATORS + 1):
                filename = f"YYout{leg_idx}{osc_idx}.txt"
                filepath = os.path.join(self.original_output_dir, filename)
                try:
                    with open(filepath, "w") as f:
                        for step in range(1, self.current_step):
                            output_value = self.legs[leg_idx].osc[osc_idx].Y[step]
                            f.write(f"{output_value}\n")
                    print(f"✅ 已儲存 {filepath}")
                except Exception as e:
                    print(f"❌ 儲存 {filepath} 時發生錯誤: {e}")
        
        print("✅ 原始CPG輸出檔案已儲存完成")
    
    def save_processed_signals(self):
        """儲存處理過的訊號到檔案"""
        print("\n💾 正在儲存處理過的訊號...")
        
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for joint_idx in range(1, self.NUM_OSCILLATORS + 1):
                filename = f"Processed{leg_idx}{joint_idx}.txt"
                filepath = os.path.join(self.processed_output_dir, filename)
                try:
                    with open(filepath, "w") as f:
                        for step in range(1, self.current_step):
                            processed_value = self.processed_signals[leg_idx][joint_idx][step]
                            f.write(f"{processed_value}\n")
                    print(f"✅ 已儲存 {filepath}")
                except Exception as e:
                    print(f"❌ 儲存 {filepath} 時發生錯誤: {e}")
        
        print("✅ 處理過的訊號檔案已儲存完成")

    def run(self):
        """主要控制迴圈"""
        print("\n🚀 開始CPG+Transformer控制...")
        
        try:
            while self.robot.step(self.timestep) != -1:
                if self.current_step < self.MAX_STEPS and not self.episode_terminated:
                    self.calculate_cpg_output(self.current_step)
                    self.apply_motor_commands()
                    self.current_step += 1
                    
                    if self.current_step % 100 == 0:
                        print(f"當前步數: {self.current_step}/{self.MAX_STEPS}")
                        
                        if len(self.state_buffer) > 0:
                            current_imu = list(self.state_buffer)[-1]
                            current_reward = list(self.reward_buffer)[-1][0]
                            raw_imu = self.get_raw_imu_data()
                            position = self.get_position_data()
                            
                            if self.start_position is not None:
                                forward_distance = np.linalg.norm((position - self.start_position)[:2])
                                print(f"前進距離: {forward_distance:.3f}m")
                            
                            print(f"IMU腳部方向分量 - e1~e6: {current_imu}")
                            print(f"原始歐拉角 - Roll: {raw_imu[0]:.3f}, Pitch: {raw_imu[1]:.3f}, Yaw: {raw_imu[2]:.3f}")
                            print(f"當前獎勵: {current_reward:.3f}, 速度: {self.get_average_speed():.3f} m/s")
                            
                            # 檢查是否接近終止條件
                            if abs(raw_imu[0]) > 0.7 or abs(raw_imu[1]) > 0.7:
                                print(f"⚠️ 警告：接近跌倒閾值 (0.785)")
                            
                            if self.initial_yaw is not None:
                                theta = abs(raw_imu[2] - self.initial_yaw)
                                if theta > np.pi:
                                    theta = 2 * np.pi - theta
                                if theta > 0.7:
                                    print(f"⚠️ 警告：接近方向偏離閾值 (0.785)")
                
                else:
                    # 達到最大步數或episode已終止
                    if self.termination_reason:
                        print(f"\n📋 Episode終止原因: {self.termination_reason}")
                    else:
                        print(f"\n✅ 已達到最大步數 {self.MAX_STEPS}")
                    
                    self.save_cpg_outputs()
                    self.save_processed_signals()
                    print("模擬完成!")
                    break
                
        except KeyboardInterrupt:
            print("\n👋 使用者中斷，儲存目前資料...")
            self.save_cpg_outputs()
            self.save_processed_signals()
            print("程式結束")

def main():
    """主函數"""
    try:
        controller = HexapodController()
        controller.run()
    except Exception as e:
        print(f"❌ 控制器執行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()