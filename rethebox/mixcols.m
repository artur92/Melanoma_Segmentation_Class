function So = mixcols(S)
M = [2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2];
for i = 1:4
    for j = 1:4
        x = 0;
        for k = 1:4
            p = pmultshort(M(j,k), S(k,i));
            x = bitxor(x,p);
        end
        y(j,i) = x;
    end
end
So = y;
end