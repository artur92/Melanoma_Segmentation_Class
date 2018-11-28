function I = multinv(X) % A is a number between 0 and 255
I = 0;
for j = 1:255
    if pmult283(j,X) == 1
        I = j;
        break
    end
end