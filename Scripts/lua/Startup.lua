-- 获取文件参数 DUT_VERSION = ...
DUT_VERSION = ...

-- RT3 安装路径
RSTD_PATH = RSTD.GetRstdPath()

-- DUT 版本选项
DUT_VER = {AR1xxx = 1}

-- 声明加载函数
dofile(RSTD_PATH .. "\\Scripts\\AR1xFunctions.lua")

-- 设置目标设备
DUT_VERSION = DUT_VERSION or DUT_VER.AR1xxx

-- 设置自动化模式开/关（无消息框）
local automation_mode = false

-- 在输出/日志中显示时间戳
RSTD.SetAndTransmit ("/Settings/Scripter/Display DateTime" , "1")
RSTD.SetAndTransmit ("/Settings/Scripter/DateTime Format" , "HH:mm:ss")

-- 载入 AR1x 客户端
Load_AR1x_Client(automation_mode)

-- 设置 TESTING 变量为 false，并在日志中输出
TESTING = false
WriteToLog("TESTING = " .. tostring(TESTING) .. "\n", "green")

-- 打开 mmWaveStudio 端口
RSTD.NetStart()