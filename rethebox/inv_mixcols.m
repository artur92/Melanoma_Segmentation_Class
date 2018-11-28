function So = inv_mixcols(S)
M = [14,11,13,9;9,14,11,13;13,9,14,11;11,13,9,14];
for i = 1:4
    for j = 1:4
        x = 0;
        for k = 1:4
            p = pmult283(M(j,k), S(k,i));
            x = bitxor(x,p);
        end
        y(j,i) = x;
    end
end
So = y;
end