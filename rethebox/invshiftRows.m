function step3 = invshiftRows(a)
b=a(2,:);
b=circshift(b',1)';
c=a(3,:);
c=circshift(c',2)';
d=a(4,:);
d=circshift(d',3)';
e=a(1,:);
a=[e;b;c;d];
step3 = a;
end