--开始记录 ADC 数据
ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
RSTD.Sleep(1000)

--触发帧
ar1.StartFrame()
RSTD.Sleep(5000)