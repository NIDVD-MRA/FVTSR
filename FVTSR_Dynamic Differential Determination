clear all;
clc;


P_overall = xlsread('*******.xlsx',1,'J2:BB53');
x = 1:45;


s_column = size(P_overall,1);



v = [];
zengyi_set = [];
mark2_set = [];

for s_size = 1:s_column;
    P = P_overall(s_size,1:45);

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
    %normalisation
    P_min = min(P_smooth);
    P_decrease = P_smooth - P_min;
    P_max = max(P_decrease);
    
    P_normalisation = P_decrease / P_max +1;
    
    

   %%
   %Dynamic Differential Determination
   P_normalisation_min = min(P_normalisation);
   P_normalisation_base_min = P_normalisation - P_normalisation_min;
   P_normalisation_max = max(P_normalisation_base_min);
   P_normalisation_double = P_normalisation_base_min + P_normalisation_max;
   
   x_judgement = 1:44;
   for x_judgement =1:44
       Dynamic_P_normalisation_single = P_normalisation_double(x_judgement + 1) / P_normalisation_double(x_judgement) -1;
       Dynamic_P_normalisation(x_judgement) = Dynamic_P_normalisation_single;
   end                                      %Find the value of the dynamic differential
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
    
%     max_Dynamic_P = max(Dynamic_P);
    [max_Dynamic_P_normalisation,max_Dynamic_P_normalisation_Index] = max(Dynamic_P_normalisation);

    % Initialising the maximal value count
    countMax = 0;
    % Initialise back-of-curve statistics
    count_dynamic_P_subset = 0;

    startIdx = 40;    % starting subscript
    endIdx = 44;      % end subscript
    
    
    % Selection of subsets of vectors - subsets of normalised P first order differentials
    subset = Dynamic_P_normalisation(startIdx:endIdx);
    % Check that all values in the subset are greater than the threshold
    allGreaterThanThreshold = all(subset > 0);
    
    
    % Iterate over the difference vectors to find extreme values
   countMax_1 = 0;
    for i = 3:length(d_Dynamic_P_normalisation)-1
       % Judgement: the first-order difference of the normalised P first-order differential is at the (i-1)th > 0 && 
       % The first-order difference of the normalised P first-order differential at i > 0 &&
       % The first-order difference of the normalised P first-order differential at the i+1st < 0 &&
       % Whether i+1 is the maximum value of the normalised P first order differentiation && 
       % 40-45 mean of normalised P > 1-5 mean of normalised P
        if d_Dynamic_P_normalisation(i-1) > 0 && d_Dynamic_P_normalisation(i) > 0 && d_Dynamic_P_normalisation(i+1) < 0 && i+1 == max_Dynamic_P_normalisation_Index && mean(P_normalisation(40:45)) > mean(P_normalisation(1:5))
            % If we find a truly extreme value
            countMax_1 = countMax + 1;
        end
    end
    
     %%
    %Strong positive judgement
    mark = 0;
    mark1 = 0;
    zengyi = P(45) - P(1);
    early_zengyi = P(45) - min(P(10:45));
    if early_zengyi > 3000
        mark = 1;
    end

    
     %%
    %Goodness of fit，R^2
    xr = 1:35;
    xrmean = mean(xr);
    P_smooth_r = (P_normalisation(11:45));
    pmean = mean(P_smooth_r);
    sumx2 = (xr-xrmean)*(xr-xrmean)';
    sumxp = (P_smooth_r - pmean)*( xr - xrmean)';
    a = sumxp/sumx2;
    b = pmean-a*xrmean;

    px = linspace(0,35,50);
    py = a*px+b;
    pxx = 1:35;
    pyy = a*pxx+b;
    mean_P_smooth = mean(P_smooth_r);
    sum_P_smooth = 0;
    sum_pyy = 0;
    size = length(xr);
    for k = 1:size;
        sum_P_smooth = (pyy(k) - mean_P_smooth)^2 + sum_P_smooth;
        sum_pyy = (P_smooth_r(k)-mean_P_smooth)^2+sum_pyy;
    end
    r = sum_P_smooth/sum_pyy;
    
    
    %%
    %Determining the plateau period
    mean_platform = mean(P_normalisation(40:45));
    std_platform = std(P_normalisation(40:45));
    CV_platform = std_platform / mean_platform;
    mark2 = 0;
    if CV_platform < 0.02
        mark2 = 1;
    end

    %Determine if it is a normal positive fluorescence curve by the number of intersections with the curve
    x_line = 1:45;
    k_line = (P_normalisation(45) - P_normalisation(1))/(45-1);
    b_line =  P_normalisation(1) - k_line;
    line_first_end = k_line * x_line + b_line;

    %Subtraction of first and last straight and curved lines
    x_line = 1:45;
    line_first_end_data = (line_first_end(1:45));
    line_first_end_sub = line_first_end_data - P_normalisation;

    %Retrieve the number of intersections
    temp = 0;
    temp_1 = 0;
    temp_2 = 0;
    for i = 1:42
        if line_first_end_sub(i) * line_first_end_sub(i+1) < 0
            temp_1 = temp_1 + 1;
            if line_first_end_sub(i) * line_first_end_sub(i+1) == 0    %The intersection point is exactly the fluorescence value point 
                temp_2 = temp_2 + 1;
            end
        end
    end
    temp = temp_1 + 0.5 * temp_2; 
   
    
    %%
    %judgements
   countMax_2 = 0;
    if mark1 == 1 
       countMax = countMax + 1;
       v(s_size,1) = countMax;
       continue
   end
   
   if countMax_1 == 1 
       if r >= 0.9301 
           countMax = 0;
        v(s_size,1) = countMax;
        continue
       end
       if r < 0.87 || (temp <= 1 && r >= 0.87)
       countMax = countMax + 1;
        v(s_size,1) = countMax;
       continue
       end   
   end
       
   
    if countMax == 0
     if mark1 == 0
        if allGreaterThanThreshold == 1 && P_normalisation(45) == max(P_normalisation) && r <= 0.87 && temp <= 1
            countMax = countMax + 1;
            v(s_size,1) = countMax;
        else
           countMax = 0;
        end 
     end
    end
    
 
v(s_size,1) = countMax;

end
