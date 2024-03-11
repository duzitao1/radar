function ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path)
    % 该脚本用于建立与 mmWaveStudio 软件的连接
    % 先决条件：
    % 在运行脚本之前在 mmWaveStudio Luashell 中键入 RSTD.NetStart()。这将打开端口 2777
    % 如果没有错误，返回 30000。
    
    % 第一次在打开 MATLAB 后运行代码时
    if (strcmp(which('RtttNetClientAPI.RtttNetClient.IsConnected'), ''))
        disp('添加 RSTD Assembly');
        RSTD_Assembly = NET.addAssembly(RSTD_DLL_Path);
        if ~strcmp(RSTD_Assembly.Classes{1}, 'RtttNetClientAPI.RtttClient')
            disp('RSTD Assembly 加载不正确。检查 DLL 路径');
            ErrStatus = -10;
            return;
        end
        Init_RSTD_Connection = 1;
    % 不是第一次运行，但端口断开连接
    else 
        if ~RtttNetClientAPI.RtttNetClient.IsConnected()
            % 原因：
            % 在 Init 调用之前，IsConnected 将被重置。因此，在 Init 之前应检查 IsConnected
            % 但是，由于在 MATLAB 打开后第一次调用之前从未调用过 Init，因此 IsConnected 第一次返回 null
            Init_RSTD_Connection = 1;
        else    
            Init_RSTD_Connection = 0;
        end
    end
    
    % 如果需要初始化连接
    if Init_RSTD_Connection
        disp('初始化 RSTD 客户端');
        ErrStatus = RtttNetClientAPI.RtttNetClient.Init();
        if (ErrStatus ~= 0)
            disp('无法初始化 NetClient DLL');
            return;
        end
        disp('连接到 RSTD 客户端');
        ErrStatus = RtttNetClientAPI.RtttNetClient.Connect('127.0.0.1', 2777);
        if (ErrStatus ~= 0)
            disp('无法连接到 mmWaveStudio');
            disp('在 mmWaveStudio 中重新打开端口。键入 RSTD.NetClose()，然后键入 RSTD.NetStart()')
            return;
        end
        pause(1); % 等待 1 秒，不是必须的。
    end
    
    disp('向 RSTD 发送测试消息');
    Lua_String = 'WriteToLog("从 MATLAB 运行脚本\n", "green")';
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
    if (ErrStatus ~= 30000)
        disp('mmWaveStudio 连接失败');
    end
    disp('测试消息发送成功');
end
