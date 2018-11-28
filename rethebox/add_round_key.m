function out = add_round_key(A, key) 
    out = bitxor (A, key);