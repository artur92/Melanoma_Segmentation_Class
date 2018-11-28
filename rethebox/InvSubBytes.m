function C = InvSubBytes(I)                                     
S = zeros(4,4);                       
for j = 1:4                           
    for l = 1:4
        S(l,j) = pmult257(I(l,j),74); 
        S(l,j) = bitxor(S(l,j),5);
        C(l,j) = multinv(S(l,j));
    end                               
end                