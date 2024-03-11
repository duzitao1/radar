ar1.ChanNAdcConfig(1, 1, 0, 1, 1, 1, 1, 2, 1, 0)

ar1.LPModConfig(0, 0)

ar1.RfInit()
RSTD.Sleep(1000)

ar1.DataPathConfig(1, 1, 0)

ar1.LvdsClkConfig(1, 1)

ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)

ar1.SetTestSource(4, 3, 0, 0, 0, 0, -327, 0, -327, 327, 327, 327, -2.5, 327, 327, 0, 0, 0, 0, -327, 0, -327, 
                      327, 327, 327, -95, 0, 0, 0.5, 0, 1, 0, 1.5, 0, 0, 0, 0, 0, 0, 0)
                      
ar1.ProfileConfig(0, 77, 100, 6, 60, 0, 0, 0, 0, 0, 0, 29.982, 0, 256, 10000, 0, 0, 30)
-- 配置chirp
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 1, 0)

ar1.EnableTestSource(1)
-- 配置帧
ar1.FrameConfig(0, 0, 8, 128, 40, 0, 0, 1)

ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)

ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)

ar1.CaptureCardConfig_PacketDelay(25)
