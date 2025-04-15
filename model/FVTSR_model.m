clear all;
clc;

P_overall = xlsread('E:\***.xlsx',sheet *,'E2:AY63');
x = 1:47;

s_column = size(P_overall,1);


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
    %smooth
    P_smooth = smooth(P);
    P_smooth = P_smooth';
    figure(2)
    plot(P_smooth);
    
    %%
    % Amplitude normalization
    P_min = min(P_smooth);
    P_max = max(P_smooth);
    P_decrease = P_smooth - P_min;
    P_normalisation = P_decrease / P_max + 1;
    
    

   %%
   % Dynamic Differential Determination
   P_normalisation_min = min(P_normalisation);
   P_normalisation_base_min = P_normalisation - P_normalisation_min;
   P_normalisation_max = max(P_normalisation_base_min);
   P_normalisation_double = P_normalisation_base_min + P_normalisation_max;
   
   x_judgement = 1:44;
   for x_judgement =1:44
       Dynamic_P_normalisation_single = P_normalisation_double(x_judgement + 1) / P_normalisation_double(x_judgement) - 1;
       Dynamic_P_normalisation(x_judgement) = Dynamic_P_normalisation_single;
   end                                      
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
    % Calculate the first order difference
    d_Dynamic_P_normalisation = diff(Dynamic_P_normalisation);
    
    [max_Dynamic_P_normalisation,max_Dynamic_P_normalisation_Index] = max(Dynamic_P_normalisation);

    % Initialise the count of extremely large values
    countMax = 0;
    % Initialise the statistical values of the back-end data of the curve
    count_dynamic_P_subset = 0;

    startIdx = 40;    % starting subscript
    endIdx = 44;      % end subscript
    
    
    % Selection of subsets of vectors - subsets of normalised P first order differentials
    subset = Dynamic_P_normalisation(startIdx:endIdx);
    %Check that all values in the subset are greater than the threshold 
    allGreaterThanThreshold = all(subset > 0);
    
    
    % Iterate over the difference vectors to find extreme values
    for i = 3:length(d_Dynamic_P_normalisation)-1
       % Judgement, the first-order difference of the normalised P first-order differential is at the (i-1)th > 0 && 
       % The first-order difference of the normalised P first-order differential at i > 0 &&
       % The first-order difference of the normalised P first-order differential at the i+1st < 0 &&
       % Whether i+1 is the maximum value of the normalised P first order differentiation && 
       % 40-45 mean of normalised P > 1-5 mean of normalised P ?
        if d_Dynamic_P_normalisation(i-1) > 0 && d_Dynamic_P_normalisation(i) > 0 && d_Dynamic_P_normalisation(i+1) < 0 && i+1 == max_Dynamic_P_normalisation_Index && mean(P_normalisation(40:45)) > mean(P_normalisation(1:5))
            % If we find a truly extreme value
            countMax = countMax + 1;
        end
    end
    
    if countMax == 0
      if  allGreaterThanThreshold == 1 && P_normalisation(45) == max(P_normalisation)
        countMax = countMax + 1;
      end
    end
    
    if countMax == 1;
         else
         countMax = 0;
         v(s_size,1) = countMax;
         disp('negative');
         continue
    end
         
    
    %%
    % Background value processing
        x_Linear = base_start : base_end ;
        P_Linear = (P_smooth(base_start:base_end));
        Linear_coefficient = polyfit(x_Linear,P_Linear,1);
        Linear = Linear_coefficient(1) * x + Linear_coefficient(2) - 1;
        P_linear = P_smooth - Linear;

        figure(11)
        plot(Linear);
        figure(3)
        plot(P_linear);
        hold on;
        plot([0,45],[0,0]);

        
    %%
    % smooth
     P2 = P_linear;

    figure(4)
    plot(P2);


    %%
    % Quantitative algorithms for energy migration
    R_0 = 5; % Distance in nanometres
    inte_a = 0.5 * R_0;
    inte_b = 1.09 * R_0;
    inte_c = R_0;
    
    Gain_midpoint = 0;
    Gain_midpoint = P2(45)*(1/2);        % Midpoint of gain
    
    % Gain midpoint corresponds to x-value (left)
     for x = 12:45
            cs_D(x) = P2(x) - Gain_midpoint;
            if cs_D(x) <= 0
                Gain_midpoint_left = x;
            end
     end                                        
    
     % Gain midpoint corresponds to x-value (right)
     Gain_midpoint_right =  Gain_midpoint_left +1;
     
     % A gain midpoint greater than or equal to 44 is considered negative.
     if Gain_midpoint_left >= 44
         countMax = 0;
         v(s_size,1) = countMax;
         disp('negative');
         continue
     end

     
%     cs_D = cs_D.';
    cs_x_1 = [Gain_midpoint_left-1;Gain_midpoint_left;Gain_midpoint_right;Gain_midpoint_right+1]';        %Fitting range
    cs_y_1 = P2(Gain_midpoint_left-1:Gain_midpoint_right+1);
    cs_coefficient1 = polyfit(cs_x_1,cs_y_1,1);
    cs_x_11 = (Gain_midpoint_right-40:1:Gain_midpoint_right+30);                               %Range of values after fitting the function (x-values)
    cs_y_11 = cs_coefficient1(1)*cs_x_11 + cs_coefficient1(2);                                % After fitting the function, take the value of the range of the fitted value to find
    
    cs_45 = cs_y_11(45);                                                                % After fitting the function, the 45-loop fit values are derived
   
    intersection_cs_y_11 = (0 - cs_coefficient1(2)) / cs_coefficient1(1);
     
 
     syms xx
    % Calculation of FRET approximate trapezoidal ratio: S1/S2 (large area (left)/small area (right))
    area_E_specific = (0.5*(inte_c - inte_a)*((R_0.^6)/(R_0.^6 + inte_c.^6) + (R_0.^6)/(R_0.^6 + inte_a.^6)))/(0.5*(inte_b - inte_c)*((R_0.^6)/(R_0.^6 + inte_b.^6) + (R_0.^6)/(R_0.^6 + inte_c.^6)));
    % Calculation of the approximate triangular ratio of fluorescence curves: 
      area_fluorescence = 0.5*(P2(Gain_midpoint_left) + cs_45) * (45 - Gain_midpoint_left) / (0.5 * (Gain_midpoint_left - xx) * (P2(Gain_midpoint_left)));
    % Solve the equation for the unknown xx
    equation = area_fluorescence == area_E_specific;
    solution = solve(equation, xx);
    solution_decimal = vpa(solution, 6);% Retain 4 decimal places
    Key point = 0.4 * solution_decimal + 0.6 * intersection_cs_y_11;
    disp('Key point =');
    disp(solution_decimal);
     
v(s_size,1) = countMax;
v(s_size,2) = Key point;
end




