"""
六足機器人CPG控制器 - Webots Python版本
基於Matsuoka震盪器的中央模式生成器
控制18個關節（每隻腳3個關節）的六足機器人
"""

import numpy as np
import math
from controller import Robot, Motor, InertialUnit, GPS, Accelerometer, Gyro
import time

class MatsuokaOscillator:
    """Matsuoka震盪器類別"""
    
    def __init__(self):
        # 震盪器狀態變數
        self.Ue = 0.0  # 伸展神經元內部狀態
        self.Uf = 0.0  # 彎曲神經元內部狀態
        self.Ve = 0.0  # 伸展神經元適應變數
        self.Vf = 0.0  # 彎曲神經元適應變數
        self.Ye = 0.0  # 伸展神經元輸出
        self.Yf = 0.0  # 彎曲神經元輸出
        self.Y = 0.0   # 最終輸出 (Yf - Ye)
        
        # Matsuoka震盪器參數
        self.t1 = 0.5    # 膜時間常數
        self.t2 = 7.5    # 適應時間常數
        self.U0 = 1.2    # 外部輸入
        self.b = 3.0     # 適應強度
        self.Wfe = -1.5  # 伸展-彎曲權重連接
        
    def update(self, dt, Wij_input=0.0, external_input=0.0):
        """更新震盪器狀態"""
        # 伸展神經元
        dUe = (-self.Ue + (self.Wfe * self.Yf) - (self.b * self.Ve) + 
               self.U0 + Wij_input + external_input) / self.t1
        self.Ue += dt * dUe
        self.Ye = max(0.0, self.Ue)
        
        dVe = (-self.Ve + self.Ye) / self.t2
        self.Ve += dt * dVe
        
        # 彎曲神經元
        dUf = (-self.Uf + (self.Wfe * self.Ye) - (self.b * self.Vf) + 
               self.U0 + Wij_input + external_input) / self.t1
        self.Uf += dt * dUf
        self.Yf = max(0.0, self.Uf)
        
        dVf = (-self.Vf + self.Yf) / self.t2
        self.Vf += dt * dVf
        
        # 最終輸出
        self.Y = self.Yf - self.Ye
        
        return self.Y

class LegCPG:
    """單隻腿的CPG控制器"""
    
    def __init__(self, leg_id):
        self.leg_id = leg_id
        # 每隻腿有3個關節（髖關節、膝關節、踝關節）
        self.oscillators = [MatsuokaOscillator() for _ in range(3)]
        
        # 關節輸出
        self.joint_outputs = [0.0, 0.0, 0.0]
        
        # CPG參數
        self.Wij = -1.0  # 關節間權重連接
        
    def update(self, dt, inter_leg_coupling=None, pitch=0.0, roll=0.0):
        """更新腿部CPG"""
        # 計算姿態反饋（針對第4個虛擬震盪器）
        feed = self._calculate_attitude_feedback(pitch, roll)
        
        # 更新每個關節的震盪器
        for i in range(3):
            # 計算關節間耦合
            other_joints = [j for j in range(3) if j != i]
            intra_coupling = sum(max(0.0, self.oscillators[j].Ye) + 
                               max(0.0, self.oscillators[j].Yf) 
                               for j in other_joints) * self.Wij
            
            # 腿間耦合
            inter_coupling = 0.0
            if inter_leg_coupling:
                inter_coupling = inter_leg_coupling * self.Wij
            
            # 外部輸入（僅對第一個關節有特殊處理）
            external_input = 0.0
            if i == 0:  # 髖關節可能需要特殊的姿態控制
                external_input = feed * 0.1
                
            # 更新震盪器
            total_coupling = intra_coupling + inter_coupling
            output = self.oscillators[i].update(dt, total_coupling, external_input)
            
            # 關節限制和後處理
            self.joint_outputs[i] = self._process_joint_output(i, output)
            
        return self.joint_outputs
    
    def _calculate_attitude_feedback(self, pitch, roll):
        """計算姿態反饋"""
        if self.leg_id == 0:  # L0 (前左腿)
            return (pitch + roll) * 0.707
        elif self.leg_id == 1:  # L1 (中左腿)
            return roll
        elif self.leg_id == 2:  # L2 (後左腿)
            return (-pitch + roll) * 0.707
        elif self.leg_id == 3:  # R0 (前右腿)
            return (-pitch - roll) * 0.707
        elif self.leg_id == 4:  # R1 (中右腿)
            return -roll
        elif self.leg_id == 5:  # R2 (後右腿)
            return (pitch - roll) * 0.707
        return 0.0
    
    def _process_joint_output(self, joint_idx, output):
        """處理關節輸出"""
        # 膝關節限制為正值
        if joint_idx == 1:  # 膝關節
            output = max(0.0, output)
            
        # 輸出範圍限制
        output = max(-0.7, min(0.7, output))
        
        return output

class HexapodCPGController:
    """六足機器人CPG控制器"""
    
    def __init__(self):
        # 初始化機器人
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # 初始化設備
        self._init_devices()
        
        # 初始化CPG
        self.legs = [LegCPG(i) for i in range(6)]
        
        # 控制參數
        self.dt = 0.02  # 20ms控制週期
        self.step_count = 0
        
        # 運動參數
        self.gait_amplitude = [1.0, 1.0]  # [左側振幅, 右側振幅]
        
        print("六足機器人CPG控制器初始化完成")
        
    def _init_devices(self):
        """初始化機器人設備"""
        # 關節名稱定義
        self.servo_names = [
            "R00", "R01", "R02",  # 右前腿
            "R10", "R11", "R12",  # 右中腿  
            "R20", "R21", "R22",  # 右後腿
            "L00", "L01", "L02",  # 左前腿
            "L10", "L11", "L12",  # 左中腿
            "L20", "L21", "L22"   # 左後腿
        ]
        
        # 初始化馬達
        self.motors = []
        for name in self.servo_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(0.0)
                self.motors.append(motor)
            else:
                print(f"警告: 無法找到馬達 {name}")
                self.motors.append(None)
        
        # 初始化感測器
        try:
            self.imu = self.robot.getDevice("inertialunit1")
            if self.imu:
                self.imu.enable(self.timestep)
        except:
            print("警告: IMU初始化失敗")
            self.imu = None
            
        try:
            self.gps = self.robot.getDevice("GPS")
            if self.gps:
                self.gps.enable(self.timestep)
        except:
            print("警告: GPS初始化失敗")
            self.gps = None
    
    def get_attitude(self):
        """獲取機器人姿態"""
        if self.imu:
            rpy = self.imu.getRollPitchYaw()
            pitch = rpy[1]
            roll = rpy[0]
            
            # 死區處理
            if abs(pitch) < 0.03:
                pitch = 0.0
            if abs(roll) < 0.03:
                roll = 0.0
                
            return pitch, roll
        return 0.0, 0.0
    
    def get_position(self):
        """獲取機器人位置"""
        if self.gps:
            pos = self.gps.getValues()
            return pos[0], pos[2]  # x, z座標
        return 0.0, 0.0
    
    def update_cpg(self):
        """更新CPG並生成關節角度"""
        # 獲取機器人姿態
        pitch, roll = self.get_attitude()
        
        # 更新每隻腿的CPG
        leg_outputs = []
        for i, leg in enumerate(self.legs):
            # 計算腿間耦合（簡化版本）
            neighbor_legs = [(i + 2) % 6, (i + 4) % 6]  # 相鄰腿
            inter_coupling = sum(max(0.0, self.legs[j].oscillators[0].Ye) + 
                               max(0.0, self.legs[j].oscillators[0].Yf) 
                               for j in neighbor_legs) * 0.1
            
            # 更新腿部CPG
            outputs = leg.update(self.dt, inter_coupling, pitch, roll)
            leg_outputs.append(outputs)
        
        return leg_outputs
    
    def apply_joint_commands(self, leg_outputs):
        """將CPG輸出應用到機器人關節"""
        # 初始化階段不動作
        if self.step_count <= 10:
            return
            
        # 對每隻腿應用輸出
        for leg_idx, outputs in enumerate(leg_outputs):
            base_motor_idx = leg_idx * 3
            
            # 根據腿的位置調整輸出
            if leg_idx < 3:  # 右側腿 (R00-R22)
                amplitude = self.gait_amplitude[1]
                # 右側腿的特殊處理
                if leg_idx == 0:  # R0
                    hip_angle = outputs[0] * amplitude
                    knee_angle = outputs[1] * 1.5 - 0.52359877  # 偏移30度
                    ankle_angle = outputs[2] * 1.0 - 0.52359877
                elif leg_idx == 1:  # R1  
                    hip_angle = outputs[0] * amplitude
                    knee_angle = outputs[1] * 1.5 + 0.52359877
                    ankle_angle = outputs[2] * 1.0 + 0.52359877
                else:  # R2
                    hip_angle = outputs[0] * amplitude  
                    knee_angle = outputs[1] * 1.5 + 0.52359877
                    ankle_angle = outputs[2] * 1.0 + 0.52359877
            else:  # 左側腿 (L00-L22)
                amplitude = self.gait_amplitude[0]
                leg_offset = leg_idx - 3
                if leg_offset == 0:  # L0
                    hip_angle = outputs[0] * amplitude
                    knee_angle = outputs[1] * 1.5 - 0.52359877
                    ankle_angle = -outputs[2] * 1.0 - 0.52359877  # 左側踝關節反向
                elif leg_offset == 1:  # L1
                    hip_angle = outputs[0] * amplitude
                    knee_angle = outputs[1] * 1.5 - 0.52359877  
                    ankle_angle = -outputs[2] * 1.0 - 0.52359877
                else:  # L2
                    hip_angle = outputs[0] * amplitude
                    knee_angle = outputs[1] * 1.5 - 0.52359877
                    ankle_angle = -outputs[2] * 1.0 - 0.52359877
            
            # 設定關節角度
            if self.motors[base_motor_idx]:
                self.motors[base_motor_idx].setPosition(hip_angle)
            if self.motors[base_motor_idx + 1]:
                self.motors[base_motor_idx + 1].setPosition(knee_angle)
            if self.motors[base_motor_idx + 2]:
                self.motors[base_motor_idx + 2].setPosition(ankle_angle)
    
    def set_gait_parameters(self, left_amplitude, right_amplitude):
        """設定步態參數"""
        self.gait_amplitude[0] = left_amplitude
        self.gait_amplitude[1] = right_amplitude
    
    def run(self):
        """主控制迴圈"""
        print("開始CPG控制...")
        
        # 初始化所有關節為0位置
        for motor in self.motors:
            if motor:
                motor.setPosition(0.0)
        
        # 等待初始化
        for _ in range(25):  # 0.5秒
            self.robot.step(self.timestep)
        
        last_time = time.time()
        
        while self.robot.step(self.timestep) != -1:
            current_time = time.time()
            
            # 控制頻率為50Hz
            if current_time - last_time >= self.dt:
                # 更新CPG
                leg_outputs = self.update_cpg()
                
                # 應用關節指令
                #self.apply_joint_commands(leg_outputs)
                
                self.step_count += 1
                last_time = current_time
                
                # 每100步輸出一次狀態
                if self.step_count % 100 == 0:
                    pos_x, pos_z = self.get_position()
                    pitch, roll = self.get_attitude()
                    print(f"步數: {self.step_count}, 位置: ({pos_x:.3f}, {pos_z:.3f}), "
                          f"姿態: pitch={pitch:.3f}, roll={roll:.3f}")

# 主程式
if __name__ == "__main__":
    # 創建控制器實例
    controller = HexapodCPGController()
    
    # 設定步態參數 (可調整以改變行走模式)
    controller.set_gait_parameters(left_amplitude=1.0, right_amplitude=1.0)
    
    # 開始運行
    controller.run()