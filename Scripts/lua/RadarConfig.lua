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
ar1.FrameConfig(0, 0, 8, 255, 40, 0, 0, 1)

ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4099)

ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)

ar1.CaptureCardConfig_PacketDelay(25)
