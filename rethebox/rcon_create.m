function rcon_create
rcon = zeros(4,10);
rcon(1,1) = 1;
for i = 2:10
   rcon(1,i) = pmultshort(2,rcon(1,i-1));
end