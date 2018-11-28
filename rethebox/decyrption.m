
%********************************************************************
%********************************************************************
%***								                              ***
%*** 			AES DECRYPTION				                      ***
%***                                                              ***
%********************************************************************
%********************************************************************

% Reading the input to be encrypted and the ciper key in decimal

%ciper_text = [87,52,174,199; 18,177,244,114; 125,190,102,111; 64,191,185,198];
ciper_text = [62 214 50 106; 157 92 133 0; 153 11 182 78; 215 166 136 181];
key = [43,40,171,9; 126,174,247,207; 21,210,21,79; 22,166,136,60];

% S-BOX which is a standard Look up table is generated in a function 

%********************************************************************
%***                    INITIAL ROUND                             ***
%********************************************************************

% KEY - EXPANSION OR KEY WHITENING STEP
key = keyexpand(key);
% ADD ROUND KEY 
ciper_text = add_round_key(ciper_text, key(1:4,41:44));

ciper_text = invshiftRows(ciper_text);

ciper_text = InvSubBytes(ciper_text);

%********************************************************************
%***                    NINE ROUNDS                              ****
%********************************************************************

for i = 9:-1:1
	    
    % ADD ROUND KEY 
	
    round_key = key(:,(4*i+1):(4*i+4));
	%A = add_round_key(A,key(:,(4*i+1):(4*i+4)))
    ciper_text = add_round_key(ciper_text,round_key);
    
    % MIX COLUMNS
    
	ciper_text = inv_mixcols(ciper_text);
    
    % SHIFT ROWS
	

	ciper_text = invshiftRows(ciper_text);
    
    %BYTE SUBSTITUTION  
	
    ciper_text = InvSubBytes(ciper_text);
	
end

% The final round does not involve mix columns	

plain_text = add_round_key(ciper_text,key(:,1:4))