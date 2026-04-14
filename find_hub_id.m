function [hub_id, rate, c] = find_hub_id(L_tr, thresh)
    % 输入: L_tr - 原始数据矩阵 (每行一个样本)
    %       thresh - 阈值，用于筛选枢纽
    % 输出: hub_id - 原始数据中枢纽行的索引
    %       rate   - 每个唯一行的加权正交比例
    %       c      - 原始行到唯一行的映射
    
    [a, b, c] = unique(L_tr, 'rows');
    a = NormalizeFea(a, 1);
    m = size(a, 1);                   
    
    tol = 1e-10;           
    S = a * a';               
    L = abs(S) > tol;             
    L = sparse(double(L));       
    
    b = b(:);                      
    s = L * b;         
    
    denom = s.^2;
    [I, J] = find(L);
    V = b(I) .* b(J);                  
    M2 = sparse(I, J, V, m, m);        
    
    T = L * M2;                         
    num2 = sum(T .* L, 2);               
   
    rate = zeros(m, 1);
    valid = denom > 0;                  
    rate(valid) = 1 - num2(valid) ./ denom(valid);
    
    temp_id = find(rate > thresh);
    
    hub_id = [];
    for i = 1:numel(temp_id)
        id = find(c == temp_id(i));
        hub_id = [hub_id; id];
    end
end