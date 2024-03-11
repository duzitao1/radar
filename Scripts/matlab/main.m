% clc
% clear
% close all

% 连接到mmwavestudio
% 将当前目录及其子目录添加到 MATLAB 搜索路径
addpath(genpath('.\'))
% 初始化 mmWaveStudio .NET 连接
RSTD_DLL_Path ='I:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
% 调用初始化函数，建立连接
ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
% 检查初始化是否成功
if (ErrStatus ~= 30000)
    disp('Init_RSTD_Connection 函数内部出现错误');
    return;
end

% %% 初始化DCA1000
% strFilename = 'K:\\aio_radar\\Scripts\\lua\\DCA1000_SetupScript.lua';
% Lua_String = sprintf('dofile("%s")',strFilename);
% ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
% if(ErrStatus ~= 30000)
%     disp('DCA1000_SetupScript.lua 函数内部出现错误');
%     return;
% end

% %% 配置雷达参数
% % 例子 Lua 命令
% strFilename = 'K:\\aio_radar\\Scripts\\lua\\RadarConfig.lua';
% Lua_String = sprintf('dofile("%s")',strFilename);
% ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
% if(ErrStatus ~= 30000)
%     disp('DCA1000_SetupScript.lua 函数内部出现错误');
%     return;
% end



% %% 开始记录
% strFilename = 'K:\\aio_radar\\Scripts\\lua\\StartADCRecording.lua';
% Lua_String = sprintf('dofile("%s")',strFilename);
% ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
% % disp("1")
% 
%%
% disp(1)
% strFilename = 'ar1.CaptureCardConfig_StartRecord("..\\..\\out\\data.bin", 1)';
% Lua_String = sprintf('%s',strFilename);
% ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
start_record("..\\..\\out\\data.bin")
% disp(1)
%% 结束











