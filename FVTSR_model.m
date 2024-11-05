%%
%相较于10_15_1版本，Ct =（比值等式解 + 函数与x轴交点）* 倍率（经验值），倍率改变为 （0.4 * 比值等式解 + 0.6 * 函数与x轴交点 ）
clear all;
clc;


P_overall = xlsread('E:\文献汇总\荧光检测系统\ct值算法论文\2023年Ct值算法论文\初稿\NAR稿\NAR投稿\论文用图-NAR用图\重复性数据\baseline标注--用于Ct值计算\2024-8-14 FAM 伯乐数据Ct值计算.xlsx',1,'E2:AY63');
%P = xlsread('D:\Desktop\research on optical detection method\ct_20220505测试数据.xlsx',1,'D70:AV70');
x = 1:47;
% P = P';

s_column = size(P_overall,1);
%s_row = size(P_overall,2);


v = [];
zengyi_set = [];
mark2_set = [];

for s_size = 1:s_column;
    P = P_overall(s_size,3:47);
    base_start = P_overall(s_size,1);
    base_end = P_overall(s_size,2);
    x = 1 :45;
    
    figure(1)
    plot(P,'*');
    hold on
    plot(P);

    %%
    %平滑
    %P_smooth = P;
    P_smooth = smooth(P);
    P_smooth = P_smooth';
    % P_smooth = smoothdata(P);
    figure(2)
    plot(P_smooth);
    
    %%
    %归一化
    P_min = min(P_smooth);
    P_max = max(P_smooth);
    P_decrease = P_smooth - P_min;
    P_normalisation = P_decrease / P_max + 1;
    
    

   %%
   %阴阳性判定（动态微分法）
   
   P_normalisation_min = min(P_normalisation);
   P_normalisation_base_min = P_normalisation - P_normalisation_min;
   P_normalisation_max = max(P_normalisation_base_min);
   P_normalisation_double = P_normalisation_base_min + P_normalisation_max;
   
   x_judgement = 1:44;
   for x_judgement =1:44
       Dynamic_P_normalisation_single = P_normalisation_double(x_judgement + 1) / P_normalisation_double(x_judgement) - 1;
       Dynamic_P_normalisation(x_judgement) = Dynamic_P_normalisation_single;
   end                                      %求动态微分值
   figure(44)
   hold on;
   plot(Dynamic_P_normalisation);
   
   for x_judgement_2 =1:43
       Dynamic_P_normalisation_single_2 = Dynamic_P_normalisation(x_judgement_2 + 1) / Dynamic_P_normalisation(x_judgement_2);
       Dynamic_P_normalisation_2(x_judgement_2) = Dynamic_P_normalisation_single_2;
   end
   figure(45)
   hold on;
   plot(Dynamic_P_normalisation_2);
                  

    %%
    % 计算一阶差分
    d_Dynamic_P_normalisation = diff(Dynamic_P_normalisation);
    
%     max_Dynamic_P = max(Dynamic_P);
    [max_Dynamic_P_normalisation,max_Dynamic_P_normalisation_Index] = max(Dynamic_P_normalisation);

    % 初始化极大值计数
    countMax = 0;
    % 初始化曲线后段数据统计值
    count_dynamic_P_subset = 0;

    startIdx = 40;    % 起始下标
    endIdx = 44;      % 结束下标
    
    
    % 选择向量的子集--归一化P一阶微分的子集
    subset = Dynamic_P_normalisation(startIdx:endIdx);
    % 检查子集中是否所有值都大于阈值（曲线后期的数值是否不断上升）
    allGreaterThanThreshold = all(subset > 0);
    
    
    % 遍历差分向量，寻找极大值
    for i = 3:length(d_Dynamic_P_normalisation)-1
       % 判断：归一化P一阶微分的一阶差分在第(i-1)个>0 && 
       % 归一化P一阶微分的一阶差分在第i个>0 &&
       % 归一化P一阶微分的一阶差分在第i+1个<0 &&
       % i+1是否为归一化P一阶微分的最大值 && 
       % 归一化P的40-45均值 > 归一化P的1-5均值
        if d_Dynamic_P_normalisation(i-1) > 0 && d_Dynamic_P_normalisation(i) > 0 && d_Dynamic_P_normalisation(i+1) < 0 && i+1 == max_Dynamic_P_normalisation_Index && mean(P_normalisation(40:45)) > mean(P_normalisation(1:5))
            % 如果找到一个真正的极大值
            countMax = countMax + 1;
        end
    end
    
    if countMax == 0
      if  allGreaterThanThreshold == 1 && P_normalisation(45) == max(P_normalisation)
        countMax = countMax + 1;
      end
    end
    
    if countMax == 1;
%        disp('阳性');
         else
         countMax = 0;
         v(s_size,1) = countMax;
         disp('阴性');
         continue
    end
         
    
    %%
    %基线平移
%     if mark2 == 0
%     if zengyi < 5000  &&  mark2 == 0
        x_Linear = base_start : base_end ;
        P_Linear = (P_smooth(base_start:base_end));
        Linear_coefficient = polyfit(x_Linear,P_Linear,1);
        Linear = Linear_coefficient(1) * x + Linear_coefficient(2) - 1;
        P_linear = P_smooth - Linear;

        % for i=1:45
        %     if P_linear(i) < 0
        %         u = i;
        %         for i = 1:u
        %             P_linear(i) = 0;
        %         end
        %     end
        % end
        figure(11)
        plot(Linear);
        figure(3)
        plot(P_linear);
        hold on;
        plot([0,45],[0,0]);

        
    %%
    %二次平滑(不一定需要，目前还在用）
%     P2 = smooth(P_linear);
     P2 = P_linear;

    figure(4)
    plot(P2);

    % P2 = P2';

    %%
    %能量迁移定量模型
    R_0 = 5; % F?ster 距离，单位为纳米
    inte_a = 0.5 * R_0;
    inte_b = 1.09 * R_0;
    inte_c = R_0;
    
    Gain_midpoint = 0;
    Gain_midpoint = P2(45)*(1/2);        %增益中点
    
    %增益中点对应x值（左）
     for x = 12:45
            cs_D(x) = P2(x) - Gain_midpoint;
            if cs_D(x) <= 0
                Gain_midpoint_left = x;
            end
     end                                        
    
     %增益中点对应x值（右）
     Gain_midpoint_right =  Gain_midpoint_left +1;
     
     % 增益中点大于等于44，判为阴性
     if Gain_midpoint_left >= 44
         countMax = 0;
         v(s_size,1) = countMax;
         disp('阴性');
         continue
     end

     
%     cs_D = cs_D.';
    cs_x_1 = [Gain_midpoint_left-1;Gain_midpoint_left;Gain_midpoint_right;Gain_midpoint_right+1]';        %拟合范围
    cs_y_1 = P2(Gain_midpoint_left-1:Gain_midpoint_right+1);
    cs_coefficient1 = polyfit(cs_x_1,cs_y_1,1);
    cs_x_11 = (Gain_midpoint_right-40:1:Gain_midpoint_right+30);                               %拟合函数后取值范围（x值）
    % cs_x_11 = [cs_jz_z-4;cs_jz_z-3;cs_jz_z-2;cs_jz_z-1;cs_jz_z;cs_jz_y;cs_jz_y+1];     
    cs_y_11 = cs_coefficient1(1)*cs_x_11 + cs_coefficient1(2);                                %拟合函数后取值范围内拟合值求取
    
    cs_45 = cs_y_11(45);                                                                %拟合函数后45循环拟合值求取
   
    intersection_cs_y_11 = (0 - cs_coefficient1(2)) / cs_coefficient1(1);
     
    
    
%    floor_base_end = floor(base_end);
 
     syms xx
    % 计算FRET近似梯形比值: S1/S2（大面积（左）/小面积（右））
    area_E_specific = (0.5*(inte_c - inte_a)*((R_0.^6)/(R_0.^6 + inte_c.^6) + (R_0.^6)/(R_0.^6 + inte_a.^6)))/(0.5*(inte_b - inte_c)*((R_0.^6)/(R_0.^6 + inte_b.^6) + (R_0.^6)/(R_0.^6 + inte_c.^6)));
    % 计算荧光曲线近似三角形比值: 
      area_fluorescence = 0.5*(P2(Gain_midpoint_left) + cs_45) * (45 - Gain_midpoint_left) / (0.5 * (Gain_midpoint_left - xx) * (P2(Gain_midpoint_left)));
    % 求解等式未知数xx
    equation = area_fluorescence == area_E_specific;
    solution = solve(equation, xx);
    solution_decimal = vpa(solution, 6);%保留4位小数
    Ct = 0.4 * solution_decimal + 0.6 * intersection_cs_y_11;
    disp('Ct=');
    disp(solution_decimal);
     
v(s_size,1) = countMax;
v(s_size,2) = Ct;
end

%xlswrite('E:\核酸现场快速检测仪器研发\循环阈值计算方法\算法\算法优化草稿\202309优化\Ct值计算-20230902-HSL-1.xlsx',v,'sheet1','E2');



