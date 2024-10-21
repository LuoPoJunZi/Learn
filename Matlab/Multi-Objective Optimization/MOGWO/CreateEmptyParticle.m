%___________________________________________________________________%
%  多目标灰狼优化算法 (MOGWO)                                       %
%  源代码演示版本 1.0                                                %
%                                                                   %
%  开发于 MATLAB R2011b(7.13)                                       %
%                                                                   %
%  作者与开发者: Seyedali Mirjalili                                  %
%                                                                   %
%         邮箱: ali.mirjalili@gmail.com                             %
%               seyedali.mirjalili@griffithuni.edu.au               %
%                                                                   %
%       主页: http://www.alimirjalili.com                           %
%                                                                   %
%  主要论文:                                                        %
%                                                                   %
%    S. Mirjalili, S. Saremi, S. M. Mirjalili, L. Coelho,           %
%    Multi-objective grey wolf optimizer: A novel algorithm for     %
%    multi-criterion optimization, Expert Systems with Applications,%
%    即将发表，DOI: http://dx.doi.org/10.1016/j.eswa.2015.10.039     %
%                                                                   %
%___________________________________________________________________%

% 本版本的MOGWO代码参考了以下代码的很大部分:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MATLAB 代码                                                      %
%                                                                   %
%  多目标粒子群优化算法 (MOPSO)                                     %
%  版本 1.0 - 2011年2月                                             %
%                                                                   %
%  根据以下文献:                                                    %
%  Carlos A. Coello Coello 等人,                                     %
%  "用粒子群优化处理多目标," IEEE进化计算交易, 第8卷第3期,          %
%  页码: 256-279, 2004年6月.                                        %
%                                                                   %
%  使用 MATLAB R2009b (版本 7.9) 开发                               %
%                                                                   %
%  编程者: S. Mostapha Kalami Heris                                  %
%                                                                   %
%         邮箱: sm.kalami@gmail.com                                 %
%               kalami@ee.kntu.ac.ir                                %
%                                                                   %
%       主页: http://www.kalami.ir                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 函数 CreateEmptyParticle
% 用于创建一个或多个空的粒子结构体

function particle=CreateEmptyParticle(n)
    
    % 如果没有传入参数 n，默认创建一个粒子
    if nargin<1
        n=1;
    end

    % 创建一个空粒子结构体
    empty_particle.Position=[];      % 粒子的位置向量
    empty_particle.Velocity=[];      % 粒子的速度向量
    empty_particle.Cost=[];          % 粒子的目标函数值（成本）
    empty_particle.Dominated=false;  % 是否被其他粒子支配的标志
    empty_particle.Best.Position=[]; % 粒子迄今为止的最佳位置
    empty_particle.Best.Cost=[];     % 粒子迄今为止的最佳目标函数值
    empty_particle.GridIndex=[];     % 粒子所在网格的索引
    empty_particle.GridSubIndex=[];  % 粒子所在网格的子索引
    
    % 使用 repmat 函数复制 n 个粒子结构体
    particle=repmat(empty_particle,n,1);
    
end
