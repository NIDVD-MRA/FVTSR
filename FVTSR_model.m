%%
%�����10_15_1�汾��Ct =����ֵ��ʽ�� + ������x�ύ�㣩* ���ʣ�����ֵ�������ʸı�Ϊ ��0.4 * ��ֵ��ʽ�� + 0.6 * ������x�ύ�� ��
clear all;
clc;


P_overall = xlsread('E:\���׻���\ӫ����ϵͳ\ctֵ�㷨����\2023��Ctֵ�㷨����\����\NAR��\NARͶ��\������ͼ-NAR��ͼ\�ظ�������\baseline��ע--����Ctֵ����\2024-8-14 FAM ��������Ctֵ����.xlsx',1,'E2:AY63');
%P = xlsread('D:\Desktop\research on optical detection method\ct_20220505��������.xlsx',1,'D70:AV70');
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
    %ƽ��
    %P_smooth = P;
    P_smooth = smooth(P);
    P_smooth = P_smooth';
    % P_smooth = smoothdata(P);
    figure(2)
    plot(P_smooth);
    
    %%
    %��һ��
    P_min = min(P_smooth);
    P_max = max(P_smooth);
    P_decrease = P_smooth - P_min;
    P_normalisation = P_decrease / P_max + 1;
    
    

   %%
   %�������ж�����̬΢�ַ���
   
   P_normalisation_min = min(P_normalisation);
   P_normalisation_base_min = P_normalisation - P_normalisation_min;
   P_normalisation_max = max(P_normalisation_base_min);
   P_normalisation_double = P_normalisation_base_min + P_normalisation_max;
   
   x_judgement = 1:44;
   for x_judgement =1:44
       Dynamic_P_normalisation_single = P_normalisation_double(x_judgement + 1) / P_normalisation_double(x_judgement) - 1;
       Dynamic_P_normalisation(x_judgement) = Dynamic_P_normalisation_single;
   end                                      %��̬΢��ֵ
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
    % ����һ�ײ��
    d_Dynamic_P_normalisation = diff(Dynamic_P_normalisation);
    
%     max_Dynamic_P = max(Dynamic_P);
    [max_Dynamic_P_normalisation,max_Dynamic_P_normalisation_Index] = max(Dynamic_P_normalisation);

    % ��ʼ������ֵ����
    countMax = 0;
    % ��ʼ�����ߺ������ͳ��ֵ
    count_dynamic_P_subset = 0;

    startIdx = 40;    % ��ʼ�±�
    endIdx = 44;      % �����±�
    
    
    % ѡ���������Ӽ�--��һ��Pһ��΢�ֵ��Ӽ�
    subset = Dynamic_P_normalisation(startIdx:endIdx);
    % ����Ӽ����Ƿ�����ֵ��������ֵ�����ߺ��ڵ���ֵ�Ƿ񲻶�������
    allGreaterThanThreshold = all(subset > 0);
    
    
    % �������������Ѱ�Ҽ���ֵ
    for i = 3:length(d_Dynamic_P_normalisation)-1
       % �жϣ���һ��Pһ��΢�ֵ�һ�ײ���ڵ�(i-1)��>0 && 
       % ��һ��Pһ��΢�ֵ�һ�ײ���ڵ�i��>0 &&
       % ��һ��Pһ��΢�ֵ�һ�ײ���ڵ�i+1��<0 &&
       % i+1�Ƿ�Ϊ��һ��Pһ��΢�ֵ����ֵ && 
       % ��һ��P��40-45��ֵ > ��һ��P��1-5��ֵ
        if d_Dynamic_P_normalisation(i-1) > 0 && d_Dynamic_P_normalisation(i) > 0 && d_Dynamic_P_normalisation(i+1) < 0 && i+1 == max_Dynamic_P_normalisation_Index && mean(P_normalisation(40:45)) > mean(P_normalisation(1:5))
            % ����ҵ�һ�������ļ���ֵ
            countMax = countMax + 1;
        end
    end
    
    if countMax == 0
      if  allGreaterThanThreshold == 1 && P_normalisation(45) == max(P_normalisation)
        countMax = countMax + 1;
      end
    end
    
    if countMax == 1;
%        disp('����');
         else
         countMax = 0;
         v(s_size,1) = countMax;
         disp('����');
         continue
    end
         
    
    %%
    %����ƽ��
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
    %����ƽ��(��һ����Ҫ��Ŀǰ�����ã�
%     P2 = smooth(P_linear);
     P2 = P_linear;

    figure(4)
    plot(P2);

    % P2 = P2';

    %%
    %����Ǩ�ƶ���ģ��
    R_0 = 5; % F?ster ���룬��λΪ����
    inte_a = 0.5 * R_0;
    inte_b = 1.09 * R_0;
    inte_c = R_0;
    
    Gain_midpoint = 0;
    Gain_midpoint = P2(45)*(1/2);        %�����е�
    
    %�����е��Ӧxֵ����
     for x = 12:45
            cs_D(x) = P2(x) - Gain_midpoint;
            if cs_D(x) <= 0
                Gain_midpoint_left = x;
            end
     end                                        
    
     %�����е��Ӧxֵ���ң�
     Gain_midpoint_right =  Gain_midpoint_left +1;
     
     % �����е���ڵ���44����Ϊ����
     if Gain_midpoint_left >= 44
         countMax = 0;
         v(s_size,1) = countMax;
         disp('����');
         continue
     end

     
%     cs_D = cs_D.';
    cs_x_1 = [Gain_midpoint_left-1;Gain_midpoint_left;Gain_midpoint_right;Gain_midpoint_right+1]';        %��Ϸ�Χ
    cs_y_1 = P2(Gain_midpoint_left-1:Gain_midpoint_right+1);
    cs_coefficient1 = polyfit(cs_x_1,cs_y_1,1);
    cs_x_11 = (Gain_midpoint_right-40:1:Gain_midpoint_right+30);                               %��Ϻ�����ȡֵ��Χ��xֵ��
    % cs_x_11 = [cs_jz_z-4;cs_jz_z-3;cs_jz_z-2;cs_jz_z-1;cs_jz_z;cs_jz_y;cs_jz_y+1];     
    cs_y_11 = cs_coefficient1(1)*cs_x_11 + cs_coefficient1(2);                                %��Ϻ�����ȡֵ��Χ�����ֵ��ȡ
    
    cs_45 = cs_y_11(45);                                                                %��Ϻ�����45ѭ�����ֵ��ȡ
   
    intersection_cs_y_11 = (0 - cs_coefficient1(2)) / cs_coefficient1(1);
     
    
    
%    floor_base_end = floor(base_end);
 
     syms xx
    % ����FRET�������α�ֵ: S1/S2�����������/С������ң���
    area_E_specific = (0.5*(inte_c - inte_a)*((R_0.^6)/(R_0.^6 + inte_c.^6) + (R_0.^6)/(R_0.^6 + inte_a.^6)))/(0.5*(inte_b - inte_c)*((R_0.^6)/(R_0.^6 + inte_b.^6) + (R_0.^6)/(R_0.^6 + inte_c.^6)));
    % ����ӫ�����߽��������α�ֵ: 
      area_fluorescence = 0.5*(P2(Gain_midpoint_left) + cs_45) * (45 - Gain_midpoint_left) / (0.5 * (Gain_midpoint_left - xx) * (P2(Gain_midpoint_left)));
    % ����ʽδ֪��xx
    equation = area_fluorescence == area_E_specific;
    solution = solve(equation, xx);
    solution_decimal = vpa(solution, 6);%����4λС��
    Ct = 0.4 * solution_decimal + 0.6 * intersection_cs_y_11;
    disp('Ct=');
    disp(solution_decimal);
     
v(s_size,1) = countMax;
v(s_size,2) = Ct;
end

%xlswrite('E:\�����ֳ����ټ�������з�\ѭ����ֵ���㷽��\�㷨\�㷨�Ż��ݸ�\202309�Ż�\Ctֵ����-20230902-HSL-1.xlsx',v,'sheet1','E2');



