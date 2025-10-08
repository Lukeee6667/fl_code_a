@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM 用户自定义配置运行脚本 (Windows版本)
REM 基于用户常用的配置：两张GPU，特定的联邦学习参数
REM =============================================================================

echo ==========================================
echo 用户自定义配置 - AlignIns+FedUP实验
echo ==========================================

REM =============================================================================
REM 用户常用配置参数
REM =============================================================================

REM GPU配置
set CUDA_VISIBLE_DEVICES=0,1

REM 基础实验参数 (基于用户提供的配置)
set POISON_FRAC=0.3
set NUM_CORRUPT=10
set NUM_AGENTS=40
set DATA=cifar10
set ATTACK=badnet
set NON_IID=--non_iid
set BETA=0.5

REM 聚合方法选择
set AGGR_METHOD=alignins_fedup_correct

REM 其他参数
set LOCAL_EP=2
set BS=64
set CLIENT_LR=0.1
set SERVER_LR=1
set ROUNDS=100

REM AlignIns参数
set ALIGNINS_STRICT_THRESHOLD=0.7
set ALIGNINS_STANDARD_THRESHOLD=0.85
set SUSPICIOUS_WEIGHT=0.3

REM FedUP参数
set FEDUP_PRUNING_RATIO=0.1
set FEDUP_P_MAX=0.15
set FEDUP_P_MIN=0.01
set FEDUP_GAMMA=5
set FEDUP_SENSITIVITY_THRESHOLD=0.5

REM =============================================================================
REM 显示配置选项
REM =============================================================================

:show_configs
echo ==========================================
echo 可用的配置选项:
echo ==========================================
echo 1. config_user_original   - 用户原始配置 + 正确实现
echo 2. config_enhanced        - 增强版配置 (更多轮数)
echo 3. config_high_attack     - 高攻击强度配置
echo 4. config_dba_attack      - DBA攻击配置
echo 5. config_cifar100        - CIFAR-100配置
echo 6. config_compare_hybrid  - 对比实验 (混合方法)
echo 7. config_custom          - 自定义参数配置
echo ==========================================
echo 当前GPU配置: CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo 当前聚合方法: %AGGR_METHOD%
echo ==========================================
echo 使用方法:
echo   run_user_config.bat                    # 交互式选择
echo   run_user_config.bat 1                  # 直接运行配置1
echo ==========================================
goto :eof

REM =============================================================================
REM 配置1：用户原始配置 + 正确实现
REM =============================================================================

:config_user_original
echo === 用户原始配置 + 正确AlignIns+FedUP实现 ===

python src/federated.py ^
    --poison_frac %POISON_FRAC% ^
    --num_corrupt %NUM_CORRUPT% ^
    --num_agents %NUM_AGENTS% ^
    --aggr %AGGR_METHOD% ^
    --data %DATA% ^
    --attack %ATTACK% ^
    %NON_IID% ^
    --beta %BETA% ^
    --local_ep %LOCAL_EP% ^
    --bs %BS% ^
    --client_lr %CLIENT_LR% ^
    --server_lr %SERVER_LR%

goto :eof

REM =============================================================================
REM 配置2：增强版配置 (更多轮数)
REM =============================================================================

:config_enhanced
echo === 增强版配置 (更多轮数) ===

python src/federated.py ^
    --poison_frac %POISON_FRAC% ^
    --num_corrupt %NUM_CORRUPT% ^
    --num_agents %NUM_AGENTS% ^
    --aggr %AGGR_METHOD% ^
    --data %DATA% ^
    --attack %ATTACK% ^
    %NON_IID% ^
    --beta %BETA% ^
    --local_ep 5 ^
    --bs %BS% ^
    --client_lr %CLIENT_LR% ^
    --server_lr %SERVER_LR% ^
    --rounds 200

goto :eof

REM =============================================================================
REM 配置3：高攻击强度配置
REM =============================================================================

:config_high_attack
echo === 高攻击强度配置 ===

python src/federated.py ^
    --poison_frac 0.5 ^
    --num_corrupt 15 ^
    --num_agents %NUM_AGENTS% ^
    --aggr %AGGR_METHOD% ^
    --data %DATA% ^
    --attack %ATTACK% ^
    %NON_IID% ^
    --beta %BETA% ^
    --local_ep %LOCAL_EP% ^
    --bs %BS% ^
    --client_lr %CLIENT_LR% ^
    --server_lr %SERVER_LR%

goto :eof

REM =============================================================================
REM 配置4：DBA攻击配置
REM =============================================================================

:config_dba_attack
echo === DBA攻击配置 ===

python src/federated.py ^
    --poison_frac %POISON_FRAC% ^
    --num_corrupt %NUM_CORRUPT% ^
    --num_agents %NUM_AGENTS% ^
    --aggr %AGGR_METHOD% ^
    --data %DATA% ^
    --attack DBA ^
    %NON_IID% ^
    --beta %BETA% ^
    --local_ep %LOCAL_EP% ^
    --bs %BS% ^
    --client_lr %CLIENT_LR% ^
    --server_lr %SERVER_LR%

goto :eof

REM =============================================================================
REM 配置5：CIFAR-100配置
REM =============================================================================

:config_cifar100
echo === CIFAR-100配置 ===

python src/federated.py ^
    --poison_frac %POISON_FRAC% ^
    --num_corrupt %NUM_CORRUPT% ^
    --num_agents %NUM_AGENTS% ^
    --aggr %AGGR_METHOD% ^
    --data cifar100 ^
    --attack %ATTACK% ^
    %NON_IID% ^
    --beta %BETA% ^
    --local_ep %LOCAL_EP% ^
    --bs %BS% ^
    --client_lr %CLIENT_LR% ^
    --server_lr %SERVER_LR%

goto :eof

REM =============================================================================
REM 配置6：对比实验 - 使用混合方法
REM =============================================================================

:config_compare_hybrid
echo === 对比实验 - 混合方法 ===

python src/federated.py ^
    --poison_frac %POISON_FRAC% ^
    --num_corrupt %NUM_CORRUPT% ^
    --num_agents %NUM_AGENTS% ^
    --aggr alignins_fedup_hybrid ^
    --data %DATA% ^
    --attack %ATTACK% ^
    %NON_IID% ^
    --beta %BETA% ^
    --local_ep %LOCAL_EP% ^
    --bs %BS% ^
    --client_lr %CLIENT_LR% ^
    --server_lr %SERVER_LR%

goto :eof

REM =============================================================================
REM 主程序
REM =============================================================================

REM 检查GPU状态
echo 检查GPU状态...
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | findstr /n "^" | findstr "^[12]:"
) else (
    echo 警告: 未找到nvidia-smi命令，无法检查GPU状态
)

echo 当前CUDA_VISIBLE_DEVICES: %CUDA_VISIBLE_DEVICES%
echo.

REM 根据参数决定运行方式
if "%1"=="" (
    REM 无参数，交互式选择
    call :show_configs
    echo.
    set /p choice=请选择配置 (1-7): 
    
    if "!choice!"=="1" call :config_user_original
    if "!choice!"=="2" call :config_enhanced
    if "!choice!"=="3" call :config_high_attack
    if "!choice!"=="4" call :config_dba_attack
    if "!choice!"=="5" call :config_cifar100
    if "!choice!"=="6" call :config_compare_hybrid
    if "!choice!"=="7" (
        echo 自定义配置需要修改脚本中的参数
        echo 或者使用Linux版本的脚本进行更灵活的配置
    )
    if "!choice!" gtr "7" echo 无效选择，请输入1-7之间的数字
    if "!choice!" lss "1" echo 无效选择，请输入1-7之间的数字
) else (
    REM 直接运行指定配置
    if "%1"=="1" call :config_user_original
    if "%1"=="2" call :config_enhanced
    if "%1"=="3" call :config_high_attack
    if "%1"=="4" call :config_dba_attack
    if "%1"=="5" call :config_cifar100
    if "%1"=="6" call :config_compare_hybrid
    if "%1"=="7" (
        echo 自定义配置需要修改脚本中的参数
        echo 或者使用Linux版本的脚本进行更灵活的配置
    )
)

pause