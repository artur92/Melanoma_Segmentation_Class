
%********************************************************************
%********************************************************************

%***								  ***

%*** 			AES ENCRYPTION				  ***

%***								  ***
%********************************************************************
%********************************************************************

function ciphertext = encrypt(plaintext,key)

% Reading the input to be encrypted and the ciper key in decimal


%Input = [25,160,154,233; 61,244,198,248; 227,226,141,72; 190,43,42,08]
   

A = [25,160,154,233;61,244,198,248;227,226,141,72;190,43,42,08];
%A = [0,0,0,0;0,0,0,0;0,0,0,0;0,0,0,01];
key = [43,40,171,9; 126,174,247,207; 21,210,21,79; 22,166,136,60];

% S-BOX which is a standard Look up table is generated in a function 

%********************************************************************

%***                    INITIAL ROUND                             ***
%********************************************************************

A = plaintext;

% KEY - EXPANSION OR KEY WHITENING STEP

key = keyexpand(key);


% ADD ROUND KEY 

A = add_round_key(A, key(1:4,1:4));




%********************************************************************

%***                    NINE ROUNDS                              ****
%********************************************************************


for i = 1:9



	%BYTE SUBSTITUTION  

	% Reading all the input values which is a 4x4 matrix and maping them 

	% with S-Box values

	A= SubBytes(A);
	
% SHIFT ROWS

	% This step provies the diffusion by cyclic shifting of rows

	% second row once, third row twice and forth row three times.

	A = shiftRows(A);	
% MIX COLUMNS
% This step provied diffusion too by performing polynomial 

	% multiplication with c(x)
        A = mixcols(A);

% ADD ROUND KEY 
	
	% it simply add (XORs) the key of the previous step to the current state
 round2 = key(:,(4*i+1):(4*i+4));
	A = add_round_key(A(:,:),key(:,(4*i+1):(4*i+4)));

end


% The final round does not involve mix columns	

A = SubBytes(A);
A = shiftRows(A);
A = add_round_key(A,key(:,41:44));
ciphertext = A

