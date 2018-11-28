function kr = keyexpand(k)
rcon = zeros(4,10);
rcon(1,1) = 1;
for i = 2:10
   rcon(1,i) = pmultshort(2,rcon(1,i-1));
end
kr = k;
for i = 1:10
   s = directread2(kr(1:4,length(kr)-3:length(kr)));
   x=s(1,4); s(1,4)=s(2,4); s(2,4)=s(3,4); s(3,4)=s(4,4); s(4,4)=x;
   for j = 1:4
       kr(j,4*i+1) = kr(j,4*(i-1)+1);
       kr(j,4*i+1) = bitxor(kr(j,4*i+1),s(j,4));
       kr(j,4*i+1) = bitxor(kr(j,4*i+1),rcon(j,i));
       for n = 2:4
           kr(j,(4*i)+n) = bitxor(kr(j,(4*(i-1))+n),kr(j,(4*i)+n-1));
       end
   end
end