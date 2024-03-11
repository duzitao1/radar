% 定义函数开始采集(一个参数:输入路径)
function [ErrStatus,command] = start_record(strFilename)
    % strFilename = 'ar1.CaptureCardConfig_StartRecord("..\\..\\out\\data.bin", 1)';
    command = sprintf('ar1.CaptureCardConfig_StartRecord("%s", 1); ar1.StartFrame();', strFilename);
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(command);
end