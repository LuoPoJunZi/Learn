% HV.m
% 计算种群的超体积（Hypervolume）指标
% 输入:
%   PopObj - 种群的目标函数值矩阵，每一行代表一个个体的多个目标值
%   PF     - 真实Pareto前沿的目标函数值矩阵
% 输出:
%   Score  - 种群的超体积值
%   PopObj - 经过归一化处理后的种群目标函数值矩阵

function [Score, PopObj] = HV(PopObj, PF)

    % 获取种群和Pareto前沿的维数
    [N, M] = size(PopObj);  % N为种群个体数，M为目标维数

    % 计算种群的最小值，取每个目标的最小值和0中的较小值
    fmin = min(min(PopObj, [], 1), zeros(1, M));

    % 计算Pareto前沿的最大值，作为归一化的上界
    fmax = max(PF, [], 1);

    % 对种群进行归一化处理
    % 归一化公式: (x - fmin) / ((fmax - fmin) * 1.1)
    PopObj = (PopObj - repmat(fmin, N, 1)) ./ repmat((fmax - fmin) * 1.1, N, 1);

    % 移除任何在归一化后超过1的个体
    PopObj(any(PopObj > 1, 2), :) = [];

    % 设置参考点为所有目标维度的1
    RefPoint = ones(1, M);

    if isempty(PopObj)
        % 如果种群为空，则超体积为0
        Score = 0;
    elseif M < 4
        % 当目标维数小于4时，计算精确的超体积值

        % 按照所有目标从小到大排序种群
        pl = sortrows(PopObj);

        % 初始化S集合，包含权重和排序后的点
        S = {1, pl};

        % 对每个维度进行切片
        for k = 1 : M-1
            S_ = {};  % 临时存储新的S集合

            for i = 1 : size(S, 1)
                % 对当前维度进行切片
                Stemp = Slice(cell2mat(S(i, 2)), k, RefPoint);

                for j = 1 : size(Stemp, 1)
                    % 更新权重和切片后的点
                    temp(1) = {cell2mat(Stemp(j, 1)) * cell2mat(S(i, 1))};
                    temp(2) = Stemp(j, 2);
                    % 将新的切片结果加入临时集合
                    S_ = Add(temp, S_);
                end
            end

            % 更新S集合
            S = S_;
        end

        % 初始化超体积得分
        Score = 0;

        for i = 1 : size(S, 1)
            % 获取当前切片的头部点
            p = Head(cell2mat(S(i, 2)));
            % 计算超体积贡献
            Score = Score + cell2mat(S(i, 1)) * abs(p(M) - RefPoint(M));
        end
    else
        % 当目标维数大于等于4时，使用蒙特卡洛方法估计超体积值

        % 设置采样点数量
        SampleNum = 1000000;

        % 设置最大值和最小值用于采样
        MaxValue = RefPoint;
        MinValue = min(PopObj, [], 1);

        % 生成均匀分布的采样点
        Samples = unifrnd(repmat(MinValue, SampleNum, 1), repmat(MaxValue, SampleNum, 1));

        if gpuDeviceCount > 0
            % 如果有GPU设备可用，使用GPU加速
            Samples = gpuArray(single(Samples));
            PopObj  = gpuArray(single(PopObj));
        end

        for i = 1 : size(PopObj, 1)
            drawnow();  % 更新图形窗口，防止MATLAB界面冻结

            % 初始化支配标记，所有采样点初始为被支配
            domi = true(size(Samples, 1), 1);
            m = 1;

            while m <= M && any(domi)
                % 检查每个目标维度是否被当前个体支配
                domi = domi & PopObj(i, m) <= Samples(:, m);
                m = m + 1;
            end

            % 移除被当前个体支配的采样点
            Samples(domi, :) = [];
        end

        % 计算超体积得分
        Score = prod(MaxValue - MinValue) * (1 - size(Samples, 1) / SampleNum);
    end
end

% Slice 函数
% 对给定点集进行切片操作
% 输入:
%   pl        - 点集
%   k         - 当前维度
%   RefPoint  - 参考点
% 输出:
%   S         - 切片结果

function S = Slice(pl, k, RefPoint)
    p  = Head(pl);    % 获取点集的第一个点
    pl = Tail(pl);    % 获取点集的剩余部分
    ql = [];          % 初始化切片后的点集
    S  = {};          % 初始化切片结果集合

    while ~isempty(pl)
        ql  = Insert(p, k+1, ql);  % 插入点到切片集合
        p_  = Head(pl);            % 获取下一个点
        cell_(1,1) = {abs(p(k) - p_(k))};  % 计算当前维度的差值
        cell_(1,2) = {ql};                  % 关联切片后的点集
        S   = Add(cell_, S);                % 添加到切片结果集合
        p   = p_;                            % 更新当前点
        pl  = Tail(pl);                      % 更新点集
    end

    ql = Insert(p, k+1, ql);  % 插入最后一个点
    cell_(1,1) = {abs(p(k) - RefPoint(k))};  % 计算与参考点的差值
    cell_(1,2) = {ql};                        % 关联切片后的点集
    S  = Add(cell_, S);                      % 添加到切片结果集合
end

% Insert 函数
% 将点插入到切片集合中，并处理点的顺序和支配关系
% 输入:
%   p  - 当前点
%   k  - 当前维度
%   pl - 切片后的点集
% 输出:
%   ql - 更新后的切片点集

function ql = Insert(p, k, pl)
    flag1 = 0;
    flag2 = 0;
    ql    = [];
    hp    = Head(pl);  % 获取切片点集的第一个点

    % 将小于当前点的点加入到切片点集中
    while ~isempty(pl) && hp(k) < p(k)
        ql = [ql; hp];
        pl = Tail(pl);
        hp = Head(pl);
    end

    ql = [ql; p];  % 插入当前点

    m  = length(p);  % 点的维数

    while ~isempty(pl)
        q = Head(pl);  % 获取下一个点
        for i = k : m
            if p(i) < q(i)
                flag1 = 1;
            else
                if p(i) > q(i)
                    flag2 = 1;
                end
            end
        end

        % 如果当前点p不完全支配点q，则保留点q
        if ~(flag1 == 1 && flag2 == 0)
            ql = [ql; Head(pl)];
        end

        pl = Tail(pl);  % 更新切片点集
    end
end

% Head 函数
% 获取点集的第一个点
% 输入:
%   pl - 点集
% 输出:
%   p  - 第一个点

function p = Head(pl)
    if isempty(pl)
        p = [];
    else
        p = pl(1, :);  % 返回第一个点
    end
end

% Tail 函数
% 获取点集的剩余部分（去除第一个点）
% 输入:
%   pl - 点集
% 输出:
%   ql - 剩余点集

function ql = Tail(pl)
    if size(pl, 1) < 2
        ql = [];
    else
        ql = pl(2:end, :);  % 返回除第一个点外的所有点
    end
end

% Add 函数
% 将切片结果添加到切片集合中，处理权重的累加
% 输入:
%   cell_ - 当前切片的权重和点集
%   S     - 现有的切片集合
% 输出:
%   S_    - 更新后的切片集合

function S_ = Add(cell_, S)
    n = size(S, 1);  % 当前切片集合的大小
    m = 0;
    for k = 1 : n
        if isequal(cell_(1,2), S(k,2))
            % 如果当前点集已存在于切片集合中，则累加权重
            S(k,1) = {cell2mat(S(k,1)) + cell2mat(cell_(1,1))};
            m = 1;
            break;
        end
    end
    if m == 0
        % 如果当前点集不存在于切片集合中，则添加新的切片
        S(n+1, :) = cell_(1, :);
    end
    S_ = S;  % 返回更新后的切片集合
end
