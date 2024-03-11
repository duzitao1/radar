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
bss_path        = "I:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr12xx_xwr14xx_radarss.bin"
-- MSS 固件路径
mss_path        = "I:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr12xx_xwr14xx_masterss.bin"

adc_data_path   = "K:\\aio_radar\\out\\bin\\test_data.bin"

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
