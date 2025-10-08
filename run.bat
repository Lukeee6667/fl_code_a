@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM =============================================================================
REM 正确的AlignIns+FedUP方法运行脚本 (Windows版本)
REM 使用方法: run.bat 或者双击运行
REM =============================================================================

echo ==========================================
echo 正确的AlignIns+FedUP方法联邦学习实验
echo ==========================================

REM =============================================================================
REM 实验配置参数 - 可以根据需要修改这些参数
REM =============================================================================

REM 基础实验参数
set DATASET=cifar10
set MODEL=resnet18
set NUM_AGENTS=100
set NUM_CORRUPT=20
set ATTACK_TYPE=badnet

REM 联邦学习参数
set ROUNDS=200
set LOCAL_EPOCHS=5
set LEARNING_RATE=0.01
set BATCH_SIZE=64

REM AlignIns检测参数
set STRICT_THRESHOLD=0.7
set STANDARD_THRESHOLD=0.85
set SUSPICIOUS_WEIGHT=0.3

REM FedUP剪枝参数
set PRUNING_RATIO=0.1
set ADAPTIVE_PRUNING=true

REM 系统参数
set SEED=42
set DEVICE=auto
set SAVE_RESULTS=true
set RESULTS_DIR=./results
set LOG_LEVEL=INFO
set LOG_INTERVAL=10

REM =============================================================================
REM 预设实验配置 - 取消注释来使用预设配置
REM =============================================================================

REM 快速测试配置 (小规模, 快速验证)
REM set DATASET=mnist
REM set MODEL=lenet
REM set NUM_AGENTS=20
REM set NUM_CORRUPT=4
REM set ROUNDS=50
REM set LOCAL_EPOCHS=3

REM 标准实验配置 (中等规模)
REM set DATASET=cifar10
REM set MODEL=resnet18
REM set NUM_AGENTS=50
REM set NUM_CORRUPT=10
REM set ROUNDS=100
REM set LOCAL_EPOCHS=5

REM =============================================================================
REM 显示当前配置
REM =============================================================================

echo 当前实验配置:
echo ------------------------------------------
echo 数据集: %DATASET%
echo 模型: %MODEL%
echo 客户端总数: %NUM_AGENTS%
echo 恶意客户端数量: %NUM_CORRUPT%
echo 攻击类型: %ATTACK_TYPE%
echo 联邦学习轮数: %ROUNDS%
echo 本地训练轮数: %LOCAL_EPOCHS%
echo 学习率: %LEARNING_RATE%
echo 批次大小: %BATCH_SIZE%
echo ------------------------------------------
echo AlignIns参数:
echo   严格阈值: %STRICT_THRESHOLD%
echo   标准阈值: %STANDARD_THRESHOLD%
echo   可疑客户端权重: %SUSPICIOUS_WEIGHT%
echo ------------------------------------------
echo FedUP参数:
echo   基础剪枝比例: %PRUNING_RATIO%
echo   自适应剪枝: %ADAPTIVE_PRUNING%
echo ------------------------------------------
echo 系统参数:
echo   随机种子: %SEED%
echo   计算设备: %DEVICE%
echo   保存结果: %SAVE_RESULTS%
echo   结果目录: %RESULTS_DIR%
echo ==========================================

REM 确认是否继续
set /p CONFIRM="是否使用以上配置开始实验? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo 实验已取消
    pause
    exit /b 1
)

REM =============================================================================
REM 检查环境和依赖
REM =============================================================================

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python
    pause
    exit /b 1
)

echo 检查必要文件...
if not exist "run_alignins_fedup_correct_example.py" (
    echo 错误: 未找到运行脚本 run_alignins_fedup_correct_example.py
    pause
    exit /b 1
)

if not exist "src" (
    echo 错误: 未找到src目录
    pause
    exit /b 1
)

REM 创建结果目录
if "%SAVE_RESULTS%"=="true" (
    if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
    echo 结果将保存到: %RESULTS_DIR%
)

REM =============================================================================
REM 构建运行命令
REM =============================================================================

echo 构建运行命令...

set PYTHON_CMD=python run_alignins_fedup_correct_example.py
set PYTHON_CMD=%PYTHON_CMD% --dataset %DATASET%
set PYTHON_CMD=%PYTHON_CMD% --model %MODEL%
set PYTHON_CMD=%PYTHON_CMD% --num_agents %NUM_AGENTS%
set PYTHON_CMD=%PYTHON_CMD% --num_corrupt %NUM_CORRUPT%
set PYTHON_CMD=%PYTHON_CMD% --attack_type %ATTACK_TYPE%
set PYTHON_CMD=%PYTHON_CMD% --rounds %ROUNDS%
set PYTHON_CMD=%PYTHON_CMD% --local_epochs %LOCAL_EPOCHS%
set PYTHON_CMD=%PYTHON_CMD% --lr %LEARNING_RATE%
set PYTHON_CMD=%PYTHON_CMD% --batch_size %BATCH_SIZE%
set PYTHON_CMD=%PYTHON_CMD% --alignins_strict_threshold %STRICT_THRESHOLD%
set PYTHON_CMD=%PYTHON_CMD% --alignins_standard_threshold %STANDARD_THRESHOLD%
set PYTHON_CMD=%PYTHON_CMD% --suspicious_weight %SUSPICIOUS_WEIGHT%
set PYTHON_CMD=%PYTHON_CMD% --fedup_pruning_ratio %PRUNING_RATIO%
set PYTHON_CMD=%PYTHON_CMD% --fedup_adaptive %ADAPTIVE_PRUNING%
set PYTHON_CMD=%PYTHON_CMD% --seed %SEED%
set PYTHON_CMD=%PYTHON_CMD% --save_results %SAVE_RESULTS%
set PYTHON_CMD=%PYTHON_CMD% --results_dir %RESULTS_DIR%
set PYTHON_CMD=%PYTHON_CMD% --log_level %LOG_LEVEL%
set PYTHON_CMD=%PYTHON_CMD% --log_interval %LOG_INTERVAL%

if not "%DEVICE%"=="auto" (
    set PYTHON_CMD=%PYTHON_CMD% --device %DEVICE%
)

REM =============================================================================
REM 开始实验
REM =============================================================================

echo ==========================================
echo 开始运行实验...
echo 命令: %PYTHON_CMD%
echo ==========================================

REM 记录开始时间
echo 实验开始时间: %date% %time%

REM 运行实验
%PYTHON_CMD%

REM 记录结束时间
echo ==========================================
echo 实验完成!
echo 结束时间: %date% %time%

REM 显示结果文件
if "%SAVE_RESULTS%"=="true" (
    if exist "%RESULTS_DIR%" (
        echo ==========================================
        echo 实验结果文件:
        dir "%RESULTS_DIR%\*.pt" 2>nul || echo 未找到结果文件
    )
)

echo ==========================================
echo 按任意键退出...
pause >nul