"""
控制experimental_platform的platform_motor，實現正弦波運動並加入隨機雜訊
"""

from controller import Supervisor
import math
import random
import time

# 初始化Supervisor
supervisor = Supervisor()

# 獲取基本時間步長
timestep = int(supervisor.getBasicTimeStep())

# 初始化隨機種子
random.seed(int(time.time()))

# 獲取experimental_platform節點
platform_node = supervisor.getFromDef("experimental_platform")
if platform_node is None:
    print("Error: Could not find 'experimental_platform' node")
    exit(1)

# 獲取platform_motor的HingeJoint節點
platform_motor_joint = None
# 遍歷platform_node的children來找到platform_motor
children_field = platform_node.getField("children")
children_count = children_field.getCount()

for i in range(children_count):
    child = children_field.getMFNode(i)
    if child is not None:
        child_def = child.getDef()
        if child_def == "platform_motor":
            platform_motor_joint = child
            break

if platform_motor_joint is None:
    print("Error: Could not find 'platform_motor' joint")
    exit(1)

# 獲取joint parameters
joint_params_field = platform_motor_joint.getField("jointParameters")
joint_params_node = joint_params_field.getSFNode()

if joint_params_node is None:
    print("Error: Could not access joint parameters")
    exit(1)

# 獲取position字段
position_field = joint_params_node.getField("position")

if position_field is None:
    print("Error: Could not access position field")
    exit(1)

print("Platform motor controller started")
print("Control function: angle = 0.2 * sin(2π * t) + noise(±0.05)")

# 計數器用於控制打印頻率
print_counter = 0
print_interval = int(1000 / timestep)  # 每秒打印一次

# 主控制循環
while supervisor.step(timestep) != -1:
    # 獲取當前模擬時間(秒)
    current_time = supervisor.getTime()
    
    # 計算正弦波角度: 0.2 * sin(2π * t)
    sine_angle = 0.2 * math.sin(1.0 * math.pi * current_time)
    
    # 生成隨機雜訊: ±0.05 弧度
    #noise = (random.random() - 0.5) * 2.0 * 0.001
    noise =0
    # 計算最終角度
    target_angle = sine_angle + noise
    
    # 設置馬達角度
    position_field.setSFFloat(target_angle)
    
    # 每秒打印一次當前狀態
    if print_counter % print_interval == 0:
        print(f"Time: {current_time:.2f} s, Sine: {sine_angle:.4f}, Noise: {noise:.4f}, Target: {target_angle:.4f} rad")
    
    print_counter += 1

# 清理
supervisor.cleanup()