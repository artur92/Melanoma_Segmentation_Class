function S = SubBytes(C)                                     
S = zeros(4,4);                       
for j = 1:4                           
    for l = 1:4
        I = multinv(C(l,j));
        S(l,j) = pmult257(I,31); 
        S(l,j) = bitxor(S(l,j),99);   
    end                               
end                