
function step3 = shiftRows(a)
b=a(2,:);
b=circshift(b',3)';
c=a(3,:);
c=circshift(c',2)';
d=a(4,:);
d=circshift(d',1)';
e=a(1,:);
a=[e;b;c;d];
step3 = a;
end