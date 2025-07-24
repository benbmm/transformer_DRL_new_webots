import sys
import time
import math
import os
import numpy as np
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

class HexapodController:
    """六足機器人控制器"""
    
    def __init__(self):
        # CPG參數
        self.MAX_STEPS = 5000
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
        self.knee_clamp_positive = True      # 是否限制膝關節為正值
        self.use_knee_signal_for_ankle = True  # True=踝關節使用膝關節反向訊號，False=使用原本CPG輸出
        self.body_height_offset = 0.5   # 機身高度偏移量（正值=更高）
        
        # 其他參數
        self.rounding_digits  = -1  # 控制CPG計算結果四捨五路到小數點後第幾位，=-1表示不四捨五入
        self.control_start_step = 100        # 馬達開始控制的步數
        # 初始化Webots機器人
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # 初始化CPG系統
        self.legs = [CPG(self.MAX_STEPS, self.NUM_OSCILLATORS) for _ in range(self.NUM_LEGS + 1)] # 索引0不使用，1-6為實際腿部
        self.current_step = 1
        
        # 新增：儲存處理過的訊號
        self.processed_signals = {}
        for leg_idx in range(1, self.NUM_LEGS + 1):
            self.processed_signals[leg_idx] = {}
            for joint_idx in range(1, self.NUM_OSCILLATORS + 1):
                self.processed_signals[leg_idx][joint_idx] = np.zeros(self.MAX_STEPS + 1)
        
        # 初始化馬達
        self.motors = {}
        self.init_motors()
        self.init_gps()
        
        # 初始化CPG系統
        self.initialize_cpg_system()

        # 建立儲存資料夾
        self.create_output_directories()

        print("六足機器人CPG控制器已初始化")
        print(f"控制頻率: {1000/self.timestep} Hz")

    def init_motors(self):
        """初始化所有馬達（修改後的對應關係）"""
        # 修改後的對應表
        # leg_idx -> (leg_name, description)
        leg_mapping = {
            1: ('R0', '右前腿'),
            2: ('R1', '右中腿'), 
            3: ('R2', '右後腿'),
            4: ('L2', '左後腿'),  # 注意：L2 對應 leg_idx=4
            5: ('L1', '左中腿'),  # 注意：L1 對應 leg_idx=5
            6: ('L0', '左前腿')   # 注意：L0 對應 leg_idx=6
        }
        
        joint_names = ['0', '1', '2']  # 髖關節、膝關節、踝關節
        joint_descriptions = ['髖關節', '膝關節', '踝關節']
        
        print("=== 馬達初始化對應關係 ===")
        
        for leg_idx in range(1, self.NUM_LEGS + 1):
            if leg_idx not in leg_mapping:
                continue
                
            leg_name, leg_desc = leg_mapping[leg_idx]
            self.motors[leg_idx] = {}
            
            print(f"CPG leg_idx={leg_idx} -> {leg_name} ({leg_desc})")
            
            for j, joint_name in enumerate(joint_names):
                joint_idx = j + 1  # CPG系統中關節索引從1開始
                motor_name = f"{leg_name}{joint_name}"
                joint_desc = joint_descriptions[j]
                
                try:
                    motor = self.robot.getDevice(motor_name)
                    if motor is None:
                        print(f"  ⚠️  找不到馬達 {motor_name}")
                        continue
                    
                    # 設定馬達為位置控制模式
                    motor.setPosition(0.0)
                    motor.setVelocity(motor.getMaxVelocity())
                    
                    self.motors[leg_idx][joint_idx] = motor
                    print(f"  ✓  joint_idx={joint_idx} ({joint_desc}) -> 馬達 {motor_name}")
                    
                except Exception as e:
                    print(f"  ❌ 初始化馬達 {motor_name} 時發生錯誤: {e}")
        
        print("=========================")
    
    def init_gps(self):
        """初始化GPS感測器"""
        try:
            self.gps_device = self.robot.getDevice("gps")
            if self.gps_device is None:
                print("❌ 找不到GPS感測器")
                return
            
            self.gps_device.enable(self.timestep)
            print("✅ GPS感測器已啟用")
            
        except Exception as e:
            print(f"❌ 初始化GPS感測器時發生錯誤: {e}")
            self.gps_device = None
    
    def create_output_directories(self):
        """建立輸出檔案的資料夾"""
        self.original_output_dir = "original_cpg_outputs"
        self.processed_output_dir = "processed_signals"
        
        try:
            # 建立原始CPG輸出資料夾
            if not os.path.exists(self.original_output_dir):
                os.makedirs(self.original_output_dir)
                print(f"✅ 建立資料夾: {self.original_output_dir}")
            
            # 建立處理過訊號資料夾
            if not os.path.exists(self.processed_output_dir):
                os.makedirs(self.processed_output_dir)
                print(f"✅ 建立資料夾: {self.processed_output_dir}")
                
        except Exception as e:
            print(f"❌ 建立資料夾時發生錯誤: {e}")
    
    def initialize_cpg_system(self):
        """初始化CPG系統"""
        # 預定義的初始值
        uf_values = [
            0.01, 0.02, 0.03,  0.05, 0.06,
            0.07,  0.09, 0.10, 0.11, 
            0.13, 0.14, 0.15,  0.17, 0.18,
            0.19,  0.21, 0.22, 0.23
        ]
        
        vf_values = [
            0.025, 0.035, 0.045,  0.065, 0.075,
            0.085,  0.105, 0.115, 0.125,
            0.145, 0.155, 0.165, 0.185, 0.195,
            0.205,  0.225, 0.235, 0.245
        ]   
        
        index = 0
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for osc_idx in range(1, self.NUM_OSCILLATORS + 1):
                self.legs[leg_idx].osc[osc_idx].Ue[1] = 0.0
                self.legs[leg_idx].osc[osc_idx].Ve[1] = 0.0
                self.legs[leg_idx].osc[osc_idx].Uf[1] = uf_values[index]
                self.legs[leg_idx].osc[osc_idx].Vf[1] = vf_values[index]
                index += 1
    
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
        
        # 計算相鄰振盪器索引
        k = (osc_idx % 3) + 1
        kk = ((osc_idx + 1) % 3) + 1
        neighbor_oscs = [k, kk]
        
        # 計算相鄰腿部索引
        neighbor_legs = [
            self.get_neighbor_index(leg_idx, 1),
            self.get_neighbor_index(leg_idx, 5)
        ]
        
        # 計算耦合項
        coupling_e = 0.0
        coupling_f = 0.0
        
        for neighbor_osc in neighbor_oscs:
            coupling_e += self.clamp_positive(leg.osc[neighbor_osc].Ye[step])
            coupling_f += self.clamp_positive(leg.osc[neighbor_osc].Yf[step])
        
        for neighbor_leg in neighbor_legs:
            coupling_e += self.clamp_positive(self.legs[neighbor_leg].osc[osc_idx].Ye[step])
            coupling_f += self.clamp_positive(self.legs[neighbor_leg].osc[osc_idx].Yf[step])
        
        # 計算伸展神經元
        osc.dUe[step] = (-osc.Ue[step] + self.WFE * osc.Yf[step] 
                     - self.B * osc.Ve[step] + self.U0 + self.WIJ * coupling_e) / self.T1
    
        osc.Ue[step + 1] = osc.Ue[step] + self.STEP_SIZE * osc.dUe[step]
        osc.Ye[step + 1] = self.clamp_positive(osc.Ue[step + 1])
        
        osc.dVe[step] = (-osc.Ve[step] + osc.Ye[step + 1]) / self.T2
        osc.Ve[step + 1] = osc.Ve[step] + self.STEP_SIZE * osc.dVe[step]
        
        # 計算彎曲神經元
        osc.dUf[step] = (-osc.Uf[step] + self.WFE * osc.Ye[step] 
                     - self.B * osc.Vf[step] + self.U0 + self.WIJ * coupling_f) / self.T1
    
        osc.Uf[step + 1] = osc.Uf[step] + self.STEP_SIZE * osc.dUf[step]
        osc.Yf[step + 1] = self.clamp_positive(osc.Uf[step + 1])
        
        osc.dVf[step] = (-osc.Vf[step] + osc.Yf[step + 1]) / self.T2
        osc.Vf[step + 1] = osc.Vf[step] + self.STEP_SIZE * osc.dVf[step]
        
        # 計算最終輸出
        if self.rounding_digits  == -1:
            osc.Y[step] = osc.Yf[step] - osc.Ye[step]  # 不四捨五入
        else:
            osc.Y[step] = round(osc.Yf[step] - osc.Ye[step], self.rounding_digits )
    
    def calculate_cpg_output(self, step):
        """計算所有CPG的輸出"""
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for osc_idx in range(1, self.NUM_OSCILLATORS + 1):
                self.update_oscillator(leg_idx, osc_idx, step)
    
    def process_special_joints(self, motor_angle, leg_idx, joint_idx):
        """處理特殊關節（合併膝關節和踝關節處理）
        
        特殊處理：
        - 所有膝關節 (joint_idx=2): 可選正值限制
        - R12 (leg_idx=2, joint_idx=3): 限制小於0
        - L12 (leg_idx=5, joint_idx=3): 限制小於0

        !!目前使用踝關節跟隨膝關節!!
        """
        # 膝關節處理
        if joint_idx == 2:
            if self.knee_clamp_positive and motor_angle <= 0:
                return 0.0
        
        # 特定踝關節處理
        elif joint_idx == 3:
            # 只有在使用原本CPG輸出時才處理踝關節特殊限制
            if not self.use_knee_signal_for_ankle:
                if (leg_idx == 2 or leg_idx == 5) and motor_angle >= 0:  # R12 或 L12
                    return 0.0
            # 使用膝關節反向訊號時，跳過踝關節的特殊處理
        
        return motor_angle
    
    def adjust_signal_direction(self, motor_angle, leg_idx, joint_idx):
        """調整訊號方向
        
        反向規則：
        - 右側腿部(leg_idx 1-3): 踝關節反向
        - 左側腿部(leg_idx 4-6): 髖關節和膝關節反向  
        - 額外踝關節反向: R02(leg_idx=1), L02(leg_idx=6), R12(leg_idx=2), L12(leg_idx=5)
        
        """
        
        # 右側腿部踝關節反向
        if not self.use_knee_signal_for_ankle:
            if leg_idx <= 3 and joint_idx == 3:
                motor_angle = -motor_angle
        # 左側腿部髖關節和膝關節反向
        if leg_idx >= 4 and (joint_idx == 1 or joint_idx == 2):
            motor_angle = -motor_angle
        # 額外的R02, L02反向
        if not self.use_knee_signal_for_ankle:
            if (leg_idx == 1 or leg_idx == 6) and joint_idx == 3:
                motor_angle = -motor_angle
            if (leg_idx == 2 or leg_idx == 5) and joint_idx == 3:
                motor_angle = -motor_angle

        return motor_angle
    
    def replace_ankle_with_knee_signal(self, motor_angle, leg_idx, joint_idx):
        """將踝關節訊號替換為同隻腳膝關節訊號的-1倍"""
        if joint_idx == 3 and self.use_knee_signal_for_ankle:
            # 使用同隻腳膝關節的訊號乘以-1
            knee_signal =self.processed_signals[leg_idx][2][self.current_step]
            return knee_signal * 1
        return motor_angle
    
    def apply_height_offset(self, motor_angle, leg_idx, joint_idx):
        """應用機身高度偏移"""
        # 只處理膝關節，以及在不使用膝關節信號控制踝關節時的踝關節
        should_apply_offset = (
            joint_idx == 2 or  # 膝關節
            (joint_idx == 3 and not self.use_knee_signal_for_ankle)  # 踝關節(條件性)
        )
    
        if should_apply_offset:
            # 右側腿部(1-3)用負偏移，左側腿部(4-6)用正偏移
            offset_direction = -1 if leg_idx <= 3 else 1
            return motor_angle + (self.body_height_offset * offset_direction)
        
        return motor_angle
    
    def apply_motor_commands(self):
        """將CPG輸出應用到馬達"""
        step = self.current_step
        for leg_idx in range(1, self.NUM_LEGS + 1):
            if leg_idx not in self.motors:
                continue
            
            for joint_idx in range(1, self.NUM_OSCILLATORS + 1):
                if joint_idx not in self.motors[leg_idx]:
                    continue
                
                # 處理流程: CPG輸出 → 踝關節訊號替換 → 特殊關節處理 → 訊號方向調整 → 馬達
                motor_angle = self.legs[leg_idx].osc[joint_idx].Y[step]
                #踝關節訊號替換(由use_knee_signal_for_ankle參數決定)
                motor_angle = self.replace_ankle_with_knee_signal(motor_angle, leg_idx, joint_idx)
                #特殊關節處理(改變姿勢相關)
                motor_angle = self.process_special_joints(motor_angle, leg_idx, joint_idx)
                #訊號方向調整(馬達轉動方向相關，前後左右方位相關)
                motor_angle = self.adjust_signal_direction(motor_angle, leg_idx, joint_idx)
                #高度偏移
                motor_angle = self.apply_height_offset(motor_angle, leg_idx, joint_idx)
                
                # 儲存處理過的訊號
                self.processed_signals[leg_idx][joint_idx][step] = motor_angle
                
                # 發送到馬達
                try:
                    if (step >= self.control_start_step):
                        self.motors[leg_idx][joint_idx].setPosition(motor_angle)
                    else:
                        motor_angle = self.replace_ankle_with_knee_signal(0, leg_idx, joint_idx)
                        motor_angle = self.apply_height_offset(motor_angle, leg_idx, joint_idx)
                        self.motors[leg_idx][joint_idx].setPosition(motor_angle)
                except Exception as e:
                    print(f"設定馬達角度時發生錯誤 (腿{leg_idx}, 關節{joint_idx}): {e}")
    
    def save_cpg_outputs(self):
        """儲存原始CPG輸出到檔案（未經處理的數據）"""
        print("\n💾 正在儲存原始CPG輸出...")
        
        # 儲存每個振盪器的原始輸出
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for osc_idx in range(1, self.NUM_OSCILLATORS + 1):
                filename = f"YYout{leg_idx}{osc_idx}.txt"
                filepath = os.path.join(self.original_output_dir, filename)
                try:
                    with open(filepath, "w") as f:
                        for step in range(1, self.current_step):
                            # 儲存原始的Y值，不做任何處理
                            output_value = self.legs[leg_idx].osc[osc_idx].Y[step]
                            f.write(f"{output_value}\n")
                    print(f"✅ 已儲存 {filepath}")
                except Exception as e:
                    print(f"❌ 儲存 {filepath} 時發生錯誤: {e}")
        
        print("✅ 原始CPG輸出檔案已儲存完成")
    
    def save_processed_signals(self):
        """儲存處理過的訊號到檔案"""
        print("\n💾 正在儲存處理過的訊號...")
        
        # 儲存每個關節的處理過訊號
        for leg_idx in range(1, self.NUM_LEGS + 1):
            for joint_idx in range(1, self.NUM_OSCILLATORS + 1):
                filename = f"Processed{leg_idx}{joint_idx}.txt"
                filepath = os.path.join(self.processed_output_dir, filename)
                try:
                    with open(filepath, "w") as f:
                        for step in range(1, self.current_step):
                            # 儲存處理過的訊號
                            processed_value = self.processed_signals[leg_idx][joint_idx][step]
                            f.write(f"{processed_value}\n")
                    print(f"✅ 已儲存 {filepath}")
                except Exception as e:
                    print(f"❌ 儲存 {filepath} 時發生錯誤: {e}")
        
        print("✅ 處理過的訊號檔案已儲存完成")
    


    def run(self):
        """主要控制迴圈"""
        print("\n🚀 開始CPG控制...")
        
        try:
            while self.robot.step(self.timestep) != -1:
                # 計算當前步的CPG輸出
                if self.current_step < self.MAX_STEPS:
                    self.calculate_cpg_output(self.current_step)
                    self.apply_motor_commands()
                    self.current_step += 1
                    
                    # 每100步顯示一次進度
                    if self.current_step % 100 == 0:
                        if hasattr(self, 'gps_device') and self.gps_device is not None:
                            position = self.gps_device.getValues()
                            height = position[2]
                            print(f"當前步數: {self.current_step}/{self.MAX_STEPS}, 機器人高度: {height:.4f} m")
                        else:
                            print(f"當前步數: {self.current_step}/{self.MAX_STEPS}")
                else:
                    # 達到最大步數，儲存所有資料並停止模擬
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