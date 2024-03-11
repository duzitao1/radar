--------------------------------------------------------------------------------
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


------------------------------ 配置 ----------------------------------
-- 使用 "DCA1000" 用于与 DCA1000 配合使用
capture_device  = "DCA1000"

-- SOP 模式
SOP_mode        = 2

-- RS232 连接波特率
baudrate        = 115200
-- RS232 COM 端口号
uart_com_port   = 7
-- 超时时间（毫秒）
timeout         = 1000

-- BSS 固件路径
bss_path        = "..\\..\\rf_eval_firmware\\radarss\\xwr12xx_xwr14xx_radarss.bin"
-- MSS 固件路径
mss_path        = "..\\..\\rf_eval_firmware\\masterss\\xwr12xx_xwr14xx_masterss.bin"

adc_data_path   = "..\\..\\out\\data.bin"

------------------------- 连接选项卡设置 ---------------------------------
-- 选择捕获设备
ret=ar1.SelectCaptureDevice(capture_device)
if(ret~=0)
then
    print("******* 错误的捕获设备 *******")
    return
end

-- SOP 模式
ret=ar1.SOPControl(SOP_mode)
RSTD.Sleep(timeout)
if(ret~=0)
then
    print("******* SOP 失败 *******")
    return
end

-- RS232 连接
ret=ar1.Connect(uart_com_port,baudrate,timeout)
RSTD.Sleep(timeout)
if(ret~=0)
then
    print("******* 连接失败 *******")
    return
end

-- 下载 BSS 固件
ret=ar1.DownloadBSSFw(bss_path)
RSTD.Sleep(2*timeout)
if(ret~=0)
then
    print("******* BSS 加载失败 *******")
    return
end

-- 下载 MSS 固件
ret=ar1.DownloadMSSFw(mss_path)
RSTD.Sleep(2*timeout)
if(ret~=0)
then
    print("******* MSS 加载失败 *******")
    return
end

-- SPI 连接
ar1.PowerOn(1, 1000, 0, 0)

-- RF 上电
ar1.RfEnable()

ar1.ChanNAdcConfig(1, 0, 0, 1, 1, 1, 1, 2, 1, 0)

ar1.LPModConfig(0, 0)

ar1.RfInit()
RSTD.Sleep(1000)

ar1.DataPathConfig(513, 1216644097, 0)

ar1.LvdsClkConfig(1, 1)

ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)

ar1.SetTestPatternConfig(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

ar1.ProfileConfig(0, 77, 100, 6, 40, 0, 0, 0, 0, 0, 0, 99.987, 0, 64, 2500, 0, 0, 30)
-- 配置chirp
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)

ar1.DisableTestSource(0)
-- 配置帧
ar1.FrameConfig(0, 0, 30, 255, 40, 0, 0, 1)

ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)

ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)

ar1.CaptureCardConfig_PacketDelay(25)

