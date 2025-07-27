"""
六足機器人RL訓練主控制器
整合Webots環境與PPO+Transformer訓練系統

這是Webots控制器的主入口檔案，負責：
1. 初始化所有系統組件
2. 選擇運行模式（訓練/測試/續訓）
3. 協調環境、策略網路和訓練器
"""

import sys
import os
import time

# 確保可以導入同目錄下的其他模組
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def print_banner():
    """顯示系統標題"""
    print("=" * 60)
    print("🤖 六足機器人強化學習訓練系統")
    print("🧠 Transformer + PPO + Webots")
    print("=" * 60)

def get_run_mode():
    """
    確定運行模式
    在實際使用中，可以通過環境變量或檔案配置來控制
    """
    # 檢查是否有模式配置檔案
    mode_file = os.path.join(current_dir, "run_mode.txt")
    
    if os.path.exists(mode_file):
        with open(mode_file, 'r') as f:
            mode = f.read().strip().lower()
        print(f"📁 從檔案讀取模式: {mode}")
    else:
        # 預設為訓練模式
        mode = "train"
        print(f"🔧 使用預設模式: {mode}")
    
    return mode

def train_new_model():
    """訓練新模型"""
    print("\n🚀 開始訓練新模型...")
    
    try:
        # 導入訓練配置
        from ppo_training_system import PPOConfig, PPOTrainer, main as train_main
        from hexapod_balance_env import HexapodBalanceEnv
        from transformer_policy import TransformerPolicyNetwork
        
        print("✅ 所有模組載入成功")
        
        # 使用預設配置或自定義配置
        config = get_training_config()
        print(f"⚙️  訓練配置: {config.total_timesteps:,} 步, LR={config.learning_rate}")
        
        # 啟動訓練主程序
        train_main()
        
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        print("請確保所有Python檔案都在正確的目錄中")
        return False
    except Exception as e:
        print(f"❌ 訓練過程出錯: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_trained_model():
    """測試已訓練的模型"""
    print("\n🧪 測試已訓練模型...")
    
    try:
        # 導入必要模組
        from hexapod_balance_env import HexapodBalanceEnv
        from transformer_policy import TransformerPolicyNetwork, TransformerPolicyWrapper
        import torch
        
        # 尋找最佳模型
        model_path = find_best_model()
        if not model_path:
            print("❌ 找不到已訓練的模型，請先進行訓練")
            return False
        
        # ✅ 智能設備選擇
        print(f"📁 載入模型: {model_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🎮 使用設備: {device}")
        # 創建環境
        env = HexapodBalanceEnv(max_episode_steps=2000, sequence_length=50)
        print("✅ 環境創建完成")
        
        # 創建策略網路
        policy = TransformerPolicyNetwork(
            state_dim=6,
            action_dim=6,
            sequence_length=50,
            hidden_size=128,
            n_layer=3,
            n_head=2
        )
        print("✅ 策略網路創建完成")
        
        # 載入訓練好的模型
        checkpoint = torch.load(model_path, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print("✅ 模型權重載入完成")
        
        # 創建策略包裝器
        wrapper = TransformerPolicyWrapper(policy)
        
        # 運行測試episodes
        run_test_episodes(env, wrapper, num_episodes=5)
        
        env.close()
        
    except Exception as e:
        print(f"❌ 測試過程出錯: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def resume_training():
    """續訓現有模型"""
    print("\n🔄 續訓現有模型...")
    
    try:
        from ppo_training_system import PPOConfig, PPOTrainer
        from hexapod_balance_env import HexapodBalanceEnv
        from transformer_policy import TransformerPolicyNetwork
        
        # 尋找最新的檢查點
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("❌ 找不到檢查點，請先進行訓練或檢查檔案路徑")
            return False
        
        print(f"📁 載入檢查點: {checkpoint_path}")
        
        # 創建系統組件
        config = get_training_config()
        env = HexapodBalanceEnv(
            max_episode_steps=config.max_episode_steps,
            sequence_length=config.sequence_length
        )
        policy = TransformerPolicyNetwork(
            state_dim=6,
            action_dim=6,
            sequence_length=config.sequence_length,
            hidden_size=config.hidden_size,
            n_layer=config.n_layer,
            n_head=config.n_head
        )
        
        # 創建訓練器並載入檢查點
        trainer = PPOTrainer(env, policy, config)
        trainer.load_model(checkpoint_path)
        
        print("✅ 檢查點載入完成，繼續訓練...")
        
        # 繼續訓練
        trainer.train()
        
    except Exception as e:
        print(f"❌ 續訓過程出錯: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_test_episodes(env, wrapper, num_episodes=5):
    """運行測試episodes"""
    print(f"\n🎯 運行 {num_episodes} 個測試episodes...")
    
    all_rewards = []
    all_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # 重置環境和策略
        state = env.reset()
        wrapper.reset_sequence_cache()
        
        total_reward = 0
        step_count = 0
        
        # 運行episode
        while True:
            # 使用確定性動作（測試時）
            action = wrapper.predict_action(state, deterministic=True)
            
            # 執行動作
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # 每100步顯示一次進度
            if step_count % 100 == 0:
                imu_data = info.get('imu_data', (0, 0))
                roll, pitch = imu_data
                print(f"  步數: {step_count}, 獎勵: {total_reward:.3f}, "
                      f"姿態: roll={roll:.3f}, pitch={pitch:.3f}")
            
            # 檢查終止條件
            if done or step_count >= 2000:
                reason = info.get('reason', 'max_steps')
                print(f"  Episode結束: {reason}")
                break
            
            state = next_state
        
        all_rewards.append(total_reward)
        all_lengths.append(step_count)
        
        print(f"  最終獎勵: {total_reward:.3f}")
        print(f"  Episode長度: {step_count}")
    
    # 顯示統計結果
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_length = sum(all_lengths) / len(all_lengths)

    avg_reward_per_step = avg_reward / avg_length if avg_length > 0 else 0
    
    print(f"\n📊 測試結果統計:")
    print(f"  平均獎勵: {avg_reward:.3f}")
    print(f"  平均每步獎勵: {avg_reward_per_step:.3f}")
    print(f"  平均長度: {avg_length:.1f}")
    print(f"  最佳獎勵: {max(all_rewards):.3f}")
    print(f"  最差獎勵: {min(all_rewards):.3f}")
    
    if avg_reward_per_step >= 0.8:
        print("🏆 性能評級: 優秀 (≥0.8)")
    elif avg_reward_per_step >= 0.5:
        print("🥈 性能評級: 良好 (≥0.5)")
    elif avg_reward_per_step >= 0.2:
        print("🥉 性能評級: 一般 (≥0.2)")
    else:
        print("❌ 性能評級: 差 (<0.2)")

def get_training_config():
    """獲取訓練配置"""
    from ppo_training_system import PPOConfig
    
    # 檢查是否有自定義配置檔案
    config_file = os.path.join(current_dir, "training_config.txt")
    
    if os.path.exists(config_file):
        print(f"📁 使用自定義配置: {config_file}")
        # 這裡可以實現從檔案讀取配置的邏輯
        # 暫時使用預設配置
    
    # 根據使用需求選擇配置
    quick_test = check_quick_test_mode()
    
    if quick_test:
        print("⚡ 使用快速測試配置")
        config = PPOConfig(
            # 快速測試配置
            total_timesteps=50000,        # 50K步
            episodes_per_update=2,        # 較小批次
            update_epochs=2,              # 較少epochs
            max_episode_steps=1000,       # 較短episodes
            eval_frequency=10,            # 頻繁評估
            save_frequency=20,            # 頻繁保存
            log_frequency=5,              # 頻繁日誌
            
            # 網路配置
            hidden_size=128,
            n_layer=3,
            n_head=2,
            
            # 學習配置
            learning_rate=3e-4,
            target_reward=500,            # 較低目標用於快速測試
            
            # 日誌配置
            use_wandb=False,
            run_name=f"hexapod_quick_test_{int(time.time())}"
        )
    else:
        print("🚀 使用正式訓練配置")
        config = PPOConfig(
            # 正式訓練配置
            total_timesteps=1000000,      # 1M步
            episodes_per_update=4,        # 標準批次
            update_epochs=4,              # 標準epochs
            max_episode_steps=2000,       # 完整episodes
            eval_frequency=20,            # 定期評估
            save_frequency=50,            # 定期保存
            log_frequency=10,             # 定期日誌
            
            # 網路配置  
            hidden_size=128,
            n_layer=3,
            n_head=2,
            
            # 學習配置
            learning_rate=3e-4,
            target_reward=1600,            # 高目標獎勵
            
            # 日誌配置
            use_wandb=False,              # 可設為True啟用Weights & Biases
            run_name=f"hexapod_training_{int(time.time())}"
        )
    
    return config

def check_quick_test_mode():
    """檢查是否為快速測試模式"""
    # 檢查快速測試標誌檔案
    quick_test_file = os.path.join(current_dir, "quick_test.flag")
    return os.path.exists(quick_test_file)

def find_best_model():
    """尋找最佳訓練模型"""
    import glob
    
    # ✅ 更全面的搜尋路徑
    possible_paths = [
        os.path.join(current_dir, "model_best.pt"),
        os.path.join(current_dir, "runs", "*", "model_best.pt"),
        os.path.join(current_dir, "runs", "*", "model_final.pt"),  # 也尋找final模型
        os.path.join(current_dir, "model_*.pt"),  # 當前目錄中的所有模型
    ]
    
    print("🔍 搜尋已訓練模型...")
    
    for path_pattern in possible_paths:
        try:
            print(f"  檢查路徑: {path_pattern}")
            
            if '*' in path_pattern:
                # ✅ 安全的 glob 操作
                matches = glob.glob(path_pattern)
                if matches:
                    print(f"  找到 {len(matches)} 個匹配檔案")
                    # ✅ 安全的檔案時間檢查
                    best_match = None
                    best_time = 0
                    
                    for match in matches:
                        try:
                            mtime = os.path.getmtime(match)
                            if mtime > best_time:
                                best_time = mtime
                                best_match = match
                        except (OSError, PermissionError) as e:
                            print(f"    ⚠️ 無法讀取檔案時間 {match}: {e}")
                            continue
                    
                    if best_match:
                        print(f"  ✅ 選擇最新檔案: {best_match}")
                        return best_match
            else:
                # ✅ 安全的檔案存在檢查
                if os.path.exists(path_pattern) and os.path.isfile(path_pattern):
                    print(f"  ✅ 找到檔案: {path_pattern}")
                    return path_pattern
                    
        except Exception as e:
            print(f"  ⚠️ 檢查路徑 {path_pattern} 時出錯: {e}")
            continue
    
    print("  ❌ 未找到任何模型檔案")
    return None

def find_latest_checkpoint():
    """尋找最新的檢查點"""
    import glob
    runs_dir = os.path.join(current_dir, "runs")
    if not os.path.exists(runs_dir):
        # 也嘗試在當前目錄直接尋找
        current_pattern = os.path.join(current_dir, "model_*.pt")
        checkpoints = glob.glob(current_pattern)
        if checkpoints:
            return max(checkpoints, key=os.path.getmtime)
        return None
    
    # 尋找所有模型檔案
    pattern = os.path.join(runs_dir, "*", "model_*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # 返回最新的檔案
    return max(checkpoints, key=os.path.getmtime)

def create_mode_files():
    """創建模式控制檔案（輔助工具）"""
    print("\n🔧 創建模式控制檔案...")
    
    # 創建模式選擇檔案
    modes = {
        "train": "訓練新模型",
        "test": "測試已訓練模型", 
        "resume": "續訓現有模型"
    }
    
    print("選擇模式:")
    for key, desc in modes.items():
        print(f"  {key}: {desc}")
    
    # 預設使用訓練模式
    selected_mode = "train"
    
    mode_file = os.path.join(current_dir, "run_mode.txt")
    with open(mode_file, 'w') as f:
        f.write(selected_mode)
    
    print(f"✅ 模式已設置為: {selected_mode}")
    
    # 創建快速測試標誌（可選）
    create_quick_test = True  # 可以改為False使用正式訓練
    if create_quick_test:
        quick_test_file = os.path.join(current_dir, "quick_test.flag")
        with open(quick_test_file, 'w') as f:
            f.write("快速測試模式")
        print("⚡ 已啟用快速測試模式")

def cleanup_on_exit():
    """退出時的清理工作"""
    print("\n🧹 清理資源...")
    
    # 這裡可以添加清理邏輯
    # 例如保存最後的狀態、關閉日誌等
    
    print("✅ 清理完成")

def main():
    """主函數：Webots控制器入口點"""
    print_banner()
    
    try:
        # 創建必要的控制檔案
        create_mode_files()
        
        # 確定運行模式
        mode = get_run_mode()
        
        print(f"\n🎮 運行模式: {mode}")
        
        # 根據模式執行相應操作
        success = False
        
        if mode == "train":
            success = train_new_model()
        elif mode == "test":
            success = test_trained_model()
        elif mode == "resume":
            success = resume_training()
        else:
            print(f"❌ 未知模式: {mode}")
            print("支援的模式: train, test, resume")
        
        if success:
            print(f"\n🎉 {mode} 模式執行完成!")
        else:
            print(f"\n❌ {mode} 模式執行失敗!")
        
    except KeyboardInterrupt:
        print("\n⏹️  程式被使用者中斷")
    except Exception as e:
        print(f"\n💥 未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_on_exit()

if __name__ == "__main__":
    main()