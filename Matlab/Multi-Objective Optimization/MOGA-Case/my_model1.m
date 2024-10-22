function [y1,y2] = my_model1(x)
% 此函数模拟了一个结合PEMFC(质子交换膜燃料电池)和AGMD(空气间隙膜蒸馏)的混合系统
% 输入参数x为优化变量数组
% 输出参数:
% y1: 负的混合系统功率密度
% y2: 负的混合系统㶲效率

% 将优化变量映射到实际物理参数范围
T = 343 + x(1)*(363 - 343);           % 操作温度(K)
P = 2 + x(2)*(5 - 2);                 % 操作压力(bar)
sigma_mem = 0.1 + x(3)*(0.3 - 0.1);   % 膜电导率(S/cm)
tmem = 0.02 + x(4)*(0.05 - 0.02);     % 膜厚度(cm)
dp = 1.00E-07 + x(5)*(6.00E-07 - 1.00E-07);  % 孔径(m)
dm = 0.0001 + x(6)*(0.0005 - 0.0001);        % 膜厚度(m)
eps = 0.5 + x(7)*(0.9 - 0.5);                % 孔隙率
da = 0.001 + x(8)*(0.006 - 0.001);           % 气隙厚度(m)
Tc = 300 + x(9)*(308 - 300);                 % 冷却温度(K)
c3 = 0.002;                                   % 常数
j = 0.1 + x(10)*(2 - 0.1);                   % 电流密度(A/cm²)

% 基本常数定义
T0 = 298.15;    % 参考温度(K)

% 计算热力学参数
detaH = -285830 + 75.44*(T-T0) - 0.5*(29.96*(T-T0)+0.5*0.00418*(T^2-T0^2)+167000*(1/T-1/T0)) ...
    - (27.28*(T-T0)+0.5*0.00326*(T^2-T0^2)-50000*(1/T-1/T0));  % 焓变(J/mol)
detaS = 69.95 + 75.44*log(T/T0) - (130.68+0.00326*(T-T0)+27.28*log(T/T0)-0.5*50000*(T^(-2)-T0^(-2))) ...
    - 0.5*(205+29.96*log(T/T0)+0.00418*(T-T0)+0.5*167000*(T^(-2)-T0^(-2)));  % 熵变(J/mol·K)
detaG = detaH - T*detaS;  % 吉布斯自由能变(J/mol)

% PEMFC基本参数
Rg = 8.314;      % 气体常数(J/mol·K)
ne = 2;          % 电子转移数
A = 16;          % PEMFC有效面积(cm²)
F = 96485;       % 法拉第常数(C/mol)
jL = 2;          % 极限电流密度(A/cm²)
c1 = 0.001;      % 常数1
c2 = 0.001;      % 常数2

% 计算水蒸气饱和压力和摩尔分数
t = T - 273.15;  % 转换为摄氏度
PsatH2O = 10^(-2.1794+0.02953*t-9.1837*10^(-5)*t^2+1.4454*10^(-7)*t^3);  % 水蒸气饱和压力(bar)
xsatH2O = PsatH2O/P;  % 水蒸气饱和摩尔分数

% 热交换器参数
ue = 0.02;        % 海水流速(m/s)
Th_ei = 300;      % 海水入口温度(K)
Cp_sea = 4064;    % 海水比热容(J/kg·K)
At = 2*(10)^(-5); % 管道截面积(m²)
Rolp_sea = 1013;  % 海水密度(kg/m³)
me = ue*Rolp_sea*At;  % 海水质量流量(kg/s)

% AGMD系统参数
Dh = 1*10^(-3);   % 水力直径(m)
tau_m = (2-eps)^2/eps;  % 膜的弯曲度
Lm = 0.07;        % 膜长度(m)
Am = 0.002;       % 膜有效面积(m²)
Rg = 8.3145;      % 气体常数(J/mol·K)
KB = 1.38*10^(-23);  % 玻尔兹曼常数(J/K)
alpha_m = 0.3;    % 膜吸收系数
Pair = 101*10^3;  % 大气压(Pa)
Km = 0.2;         % 膜导热系数(W/m·K)
hm = Km/dm;       % 膜传热系数(W/m²·K)

molw = 0.018;     % 水的摩尔质量(kg/mol)

% 进料侧参数
Uf = 0.02;        % 进料流速(m/s)
muf = 0.000749;   % 海水粘度(Pa·s)
Kf = 0.638624;    % 海水导热系数(W/m·K)
Cpf = 4064;       % 海水比热容(J/kg·K)
rhof = 1013;      % 海水密度(kg/m³)
Prf = muf*Cpf/Kf; % 普朗特数
Ref = Uf*rhof*Dh/muf;  % 雷诺数
Nuf = 1.86*(Ref*Prf*Dh/Lm)^0.33;  % 努塞尔数
hf = Nuf*Kf/Dh;   % 进料侧传热系数(W/m²·K)

% 计算海水成分
xw = (1000/18)/(1000/18+13/58.5);    % 水的摩尔分数
xNa = (13/58.5)/(1000/18+13/58.5);   % NaCl的摩尔分数
yw = 1-xNa/2-10*xNa^2;               % 活度系数

% 气隙参数
Cpcd = 2446.3;    % 水蒸气比热容(J/kg·K)
Kg = 0.025;       % 气相导热系数(W/m·K)
ha = Kg/da;       % 气隙传热系数(W/m²·K)

% 冷却侧参数
Up = 0.02;        % 冷却水流速(m/s)
mup = 0.000868;   % 冷凝水粘度(Pa·s)
Kp = 0.633201;    % 冷凝水导热系数(W/m·K)
Cpp = 4182;       % 冷凝水比热容(J/kg·K)
Rolp = 996;       % 冷凝水密度(kg/m³)
Prp = mup*Cpp/Kp; % 冷却侧普朗特数
Rep = Up*Rolp*Dh/mup;  % 冷却侧雷诺数
Nup = 1.86*(Rep*Prp*Dh/Lm)^0.33;  % 冷却侧努塞尔数
hc = Nup*Kp/Dh;   % 冷却侧传热系数(W/m²·K)

% 冷凝水参数
g = 9.8;          % 重力加速度(m/s²)
Rold = Rolp;      % 冷凝水密度(kg/m³)
Kd = Kp;          % 冷凝水导热系数(W/m·K)

% 冷凝板参数
Kcp = 401;        % 冷凝板导热系数(W/m·K)
dcp = 1*10^(-3);  % 冷凝板厚度(m)
hcp = Kcp/dcp;    % 冷凝板传热系数(W/m²·K)
mud = mup;        % 冷凝水粘度(Pa·s)
Hm = 0.02;        % 冷凝板高度(m)

% 计算PEMFC性能参数
% 计算理论电压V0
PH2 = 0.5*PsatH2O*((exp((1.653*j)/T^1.334)*xsatH2O)^(-1)-1);  % 氢气分压
lambd_air = 2;    % 空气化学计量比
xN2in = 0.79*(1-xsatH2O);  % 入口氮气摩尔分数
xN2out = (1-xsatH2O)/(1+((lambd_air-1)/lambd_air)*(0.21/0.79));  % 出口氮气摩尔分数
xchannelN2 = (xN2in-xN2out)/(log(xN2in/xN2out));  % 通道内平均氮气摩尔分数
PO2 = P*(1-xsatH2O-xchannelN2*exp((0.291*j)/T^0.832));  % 氧气有效分压
PH2O = P*xsatH2O;  % 水蒸气分压
V0 = 1.229-8.5*10^(-4)*(T-T0)+4.3085*10^(-5)*T*log(PH2*PO2^0.5/PH2O);  % 理论电压

% 计算活化过电位Vact
CO2conc = 1.97*10^(-7)*PO2*exp(498/T);  % 氧气浓度
CH2conc = 9.174*10^(-7)*PH2*exp(-77/T); % 氢气浓度
ksi = 0.00286+0.0002*log(A)+0.000043*log(CO2conc);  % 经验常数
Vact = 0.9514-ksi*T+0.000187*T*log(j*A)-0.000074*T*log(CO2conc);  % 活化过电位

% 计算欧姆过电位Vohm
Vohm = j*tmem*(sigma_mem)^(-1);  % 欧姆过电位

% 计算浓差过电位Vcon
P1 = PO2/0.1173+PsatH2O;
if P1<2
    beta1 = (7.16*10^(-4)*T-0.622)*P1-1.45*10^(-3)*T+1.68;
else
    beta1 = (8.66*10^(-5)*T-0.068)*P1-1.6*10^(-4)*T+0.54;
end
beta2 = 2;
Vcon = j*((beta1*j)/jL)^beta2;  % 浓差过电位

% 计算实际电压和性能参数
V = V0-Vact-Vohm-Vcon;  % 实际电压

if V>=0  % 当电压为正值时进行计算
    % 计算PEMFC性能指标
    PPEMC = V*j;  % 功率密度(W/cm²)
    EFPEM = (ne*F*V)/(-detaH);  % 效率
    EXDall = -j*A*detaG/(ne*F)-T*j*A*detaS*(1-T0/T)/(ne*F);  % 总㶲损
    EXDPEM = -j*detaG/(ne*F)-T*j*detaS*(1-T0/T)/(ne*F)-j*V;  % PEMFC㶲损
    XPEM = PPEMC/(EXDall/A);  % PEMFC㶲效率
    
    % 计算热量传递
    qh = (-A*detaH/(ne*F))*((1-EFPEM)*j-(ne*F*(c3)*(T-T0))/(-detaH));  % 产生的热量
    qw = qh;  % 废热量
    Th_eo = qw/(me*Cp_sea)+Th_ei;  % 出口温度
    Tf = Th_eo;  % 进料温度
    
    % 初始化膜面温度
    Tmf = Tf-1;
    Tcd = Tc+1;
    Tmf_cal = (Tmf + Tcd)/2;
    Tcd_cal = (Tmf + Tcd)/2;
    
    % 迭代计算膜面温度
    count = 0;
    eqs = 1e-6;  % 收敛精度
    while count < 100
        % 计算膜两侧压力
        Pmf = exp(1)^(23.328 - 3841/(Tmf - 45));
        Pcd = exp(1)^(23.328 - 3841/(Tcd - 45));
        T_ave = (Tmf + Tcd)/2;
        
        % 计算扩散系数
        PDwa = 1.895*10^(-5)*Tmf^2.072;
        Dk = (2/3*(8/pi)^0.5*(molw/(Rg*T_ave))^0.5*(eps*dp))/(tau_m*dm+da);
        DM = (PDwa*eps*molw/(Pair*Rg*T_ave))/(dm*tau_m+da); % 计算扩散系数（续）
        Dm = (1/Dk + 1/DM)^(-1);  % 有效扩散系数(m²/s)

        % 计算渗透通量
        Fs = Dm*(xw*yw*Pmf - Pcd);  % 膜渗透通量(kg/m²·s)
        Fh = Fs*3600;  % 转换为小时通量(kg/m²·h)
        Hv = (1.7535*T_ave + 2024.3)*1000;  % 蒸发潜热(J/kg)

        % 计算热传递
        Qf = (hf+Fs*Cpf)*(Tf-Tmf);  % 进料侧热传递(W/m²)

        % 计算膜的等效导热系数
        KM = (eps/Kg+(1-eps)/Km)^(-1);  % 膜等效导热系数(W/m·K)
        Qc = (Tmf-Tcd)/(dm/KM+da/Kg);   % 通过膜和气隙的传导热量(W/m²)

        % 计算冷凝板温度
        Tpc = Tc+Qc/hc;      % 冷凝板冷侧温度(K)
        Tpa = Tpc+Qc/hcp;    % 冷凝板热侧温度(K)

        % 计算传热系数
        hcd = Fs*Cpcd/(1-exp(-Fs*Cpcd/ha));  % 气隙传热系数(W/m²·K)
        hd = (g*Rold^2*Hv*Kd^3/(Hm*mud*(Tcd-Tpa)))^(1/4);  % 冷凝膜传热系数(W/m²·K)
        hp = 1/(1/hd+1/Kcp+1/hc);  % 总传热系数(W/m²·K)

        % 计算热通量
        Qv = Fs*Hv;  % 蒸发热通量(W/m²)
        Qcd = hcd*(Tmf-Tcd)+Fs*Hv;  % 冷凝侧总热通量(W/m²)

        % 更新膜面温度
        Tmf_cal = Tf-(1/(hf*(1/hf+1/hcd+1/hp)))*(Tf-Tc+Fs*Hv/hcd);
        Tcd_cal = Tc+(1/(hf*(1/hf+1/hcd+1/hp)))*(Tf-Tc+Fs*Hv/hcd);

        count = count+1;  % 迭代计数

        % 检查收敛性
        if abs(Tmf-Tmf_cal)<eqs && abs(Tcd-Tcd_cal)<eqs 
            break;
        end
        
        % 更新温度值
        Tmf = Tmf_cal;
        Tcd = Tcd_cal;
    end

    % 获取真实值并更新计算结果
    CCount = count;  % 记录迭代次数
    Fs = real(Fs);  % 实际渗透通量
    Fh = real(Fh);  % 小时通量
    Tmf = real(Tmf);  % 膜面温度
    Tcd = real(Tcd);  % 冷凝温度

    % 更新平均温度和蒸发潜热
    T_ave = (Tmf + Tcd)/2;
    Hv = (1.7535*T_ave + 2024.3)*1000;
    Pmf = real(Pmf);
    Pcd = real(Pcd);

    % 计算AGMD系统性能
    Qv = Fs*Hv;  % 蒸发热通量(W/m²)
    PAGMD = Qv*Am;  % AGMD功率(W)
    PDAGMD = Qv*Am/A;  % AGMD功率密度(W/cm²)
    
    % 计算传热量
    Qc = (Tmf-Tcd)/(dm/KM+da/Kg);  % 导热量(W/m²)
    Qcd = hcd*(Tmf-Tcd)+Fs*Hv;  % 总传热量(W/m²)
    
    % 计算克努森数
    de = 3.66*10^(-10);  % 分子直径(m)
    lamda = KB*Tmf/(sqrt(2)*pi*Pair*de^2);  % 平均自由程(m)
    KN = real(lamda/dp);  % 克努森数
    
    % 计算能量平衡
    see = (Qv+Qc)*Am;  % AGMD利用的废热量(W)
    com = qh-(Qv+Qc)*Am;  % 剩余废热量(W)

    % 当满足能量守恒时计算系统性能
    if com>=0 && PDAGMD>=0
        % 计算㶲分析参数
        EX_in = qh*(1-T0/T);  % 输入㶲(W)
        EX_out = PDAGMD*A;    % 输出㶲(W)
        EXD_AGMD = (EX_in-EX_out)/A;  % AGMD㶲损(W/cm²)
        X_AGMD = EX_out/EX_in;  % AGMD㶲效率
        
        % 计算混合系统性能
        Pd_hy = V*j+PDAGMD;  % 混合系统功率密度(W/cm²)
        efficiency_hy = (2*F*Pd_hy)/(-j*detaH);  % 混合系统效率
        efficiency_AGMD = PAGMD/qh;  % AGMD效率
        EXD_hy = EXDall/A-PPEMC-EX_out/A;  % 混合系统㶲损(W/cm²)
        Xhy = (PPEMC*A+EX_out)/EXDall;  % 混合系统㶲效率
        Pd_agmd = PDAGMD;  % AGMD功率密度(W/cm²)
    else
        % 当不满足能量守恒时，只考虑PEMFC性能
        Pd_hy = PPEMC;
        efficiency_AGMD = 0;
        Pd_agmd = 0;
        EXD_AGMD = 0;
        X_AGMD = 0;
        efficiency_hy = EFPEM;
        EXD_hy = EXDPEM;
        Xhy = XPEM;
    end

    % 计算系统经济性指标
    Am1 = Am*100;  % 转换膜面积(cm²)
    A1 = A*100;    % 转换电池面积(cm²)

    % 计算各项成本
    C1 = 2.5*PPEMC*A1;  % PEMFC成本
    C2 = 36*Am1;        % 膜成本
    C3 = 19.3*3;        % 其他材料成本
    C4 = 120;           % 固定成本
    C5 = 33.13;         % 其他成本
    
    % 计算热回收相关参数
    Qre = c2*A1*(T-T0);  % 回收热量
    LTMDRE = (T-T0)/(log(T/T0));  % 对数平均温差
    Qw = 37.649423870095780;  % 最大废热量
    LTMDIHE = (Th_eo-Th_ei)/(log(Th_eo/Th_ei));  % 换热器对数平均温差
    
    % 计算热交换面积
    Are = Qre/LTMDRE;    % 回收热交换面积
    Aihe = Qw/LTMDIHE;   % 换热器面积
    
    % 计算热交换器成本
    C6 = 2681*(Are)^0.59;   % 回收热交换器成本
    C7 = 2143*(Aihe)^0.514; % 换热器成本
    
    LCinv = C1+C2+C3+C4+C5+C6+C7;  % 总投资成本

else
    % 当电压为负值时，系统无效
    Pd_hy = 0;   % 功率密度为0
    Xhy = 0;     % 㶲效率为0
    LCinv = LCinv;  % 保持原投资成本
end

% 输出优化目标值（负值用于最小化问题）
y1 = -Pd_hy;  % 负的混合系统功率密度
y2 = -Xhy;    % 负的混合系统㶲效率

end
