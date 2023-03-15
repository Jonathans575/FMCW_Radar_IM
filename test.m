close all; clear; clc;

% parameters setting
f_s = 12e6;

% ISTFT parameters setting
W_len = 256;
Win = hamming(W_len,'periodic');
hop = 1;
Overlap_len = W_len-hop;

% file_for_test
num_SNR = 9;  % number of SNR
num_SINR = 12;  % number of SINR
num_class_samples = 20;  % number of samples

filename  = 'dataset/FMCW_test.hdf5';
inp_real  = h5read(filename,'/X_real');
inp_imag  = h5read(filename,'/X_imag');
oral_real = h5read(filename,'/Y_real');
oral_imag = h5read(filename,'/Y_imag');

% Open hdf5 file
file_name = 'model_compare_model_cnn_256';
filename  = ['dataset/' file_name '.hdf5'];
pred_real = h5read(filename,'/Y_real');
pred_imag = h5read(filename,'/Y_imag');

%% Test
% save results
SINR_inp = []; SINR_dl = [];
P_inp = []; P_dl = [];
RAD_inp = []; RAD_dl = [];

for p = 1:num_SINR

count = 0;
sinr_inp = 0; sinr_dl = 0; 
p_inp = 0; p_dl = 0; 
rad_inp = 0; rad_dl = 0;  

a = max(p-3,1);
for i = a:num_SNR
    starting = 0;
    temp = 4;
    for hh = 2:i
        starting = starting + temp;
        temp = temp + 1;
    end
    starting = starting * num_class_samples;
    
    for j = num_class_samples*(p-1)+1 + starting : num_class_samples*p + starting
        count = count + 1;

        sig_TF_inp  = inp_real(:,:,j)  + 1i * inp_imag(:,:,j);
        sig_TF_oral = oral_real(:,:,j) + 1i * oral_imag(:,:,j);
        sig_TF_pred = pred_real(:,:,j) + 1i * pred_imag(:,:,j);

        % ISTFT
        [sig_inp,  t_inp]  = istft(sig_TF_inp, f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
        [sig_oral, t_oral] = istft(sig_TF_oral,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
        [sig_pred, t_pred] = istft(sig_TF_pred,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);

        % calculate SINR
        temp = 10*log10(sum(power(abs(sig_oral),2))/sum(power(abs(sig_inp - sig_oral),2)));
        sinr_inp = sinr_inp + temp;

        temp = 10*log10(sum(power(abs(sig_oral),2))/sum(power(abs(sig_pred - sig_oral),2)));
        sinr_dl = sinr_dl + temp;
        
        % calculate p and rad
        temp = (sig_inp' * sig_oral)/...
               (sqrt(sum(sig_oral .* conj(sig_oral))) * sqrt(sum(sig_inp .* conj(sig_inp))));
        p_inp = p_inp + abs(temp);
        rad_inp = rad_inp + abs(angle(temp));
        
        temp = (sig_pred' * sig_oral)/...
               (sqrt(sum(sig_oral .* conj(sig_oral))) * sqrt(sum(sig_pred .* conj(sig_pred))));
        p_dl = p_dl + abs(temp);
        rad_dl = rad_dl + abs(angle(temp));
    end
end

SINR_inp(p) = sinr_inp/count; SINR_dl(p) = sinr_dl/count;
P_inp(p) = p_inp/count; P_dl(p) = p_dl/count;
RAD_inp(p) = rad_inp/count; RAD_dl(p) = rad_dl/count;
end

fprintf(['SINR' file_name(6:length(file_name)) '=[']);
for i = 1:1:11
    fprintf('%.4f', SINR_dl(i)); fprintf(',');
end
fprintf('%.4f];', SINR_dl(12));
fprintf(' %% ');
fprintf('%.4f\n', mean(SINR_dl)); 

fprintf(['p' file_name(6:length(file_name)) '=[']);
for i = 1:1:11
    fprintf('%.4f', P_dl(i)); fprintf(',');
end
fprintf('%.4f];', P_dl(12));
fprintf(' %% ');
fprintf('%.4f\n', mean(P_dl)); 

fprintf(['rad' file_name(6:length(file_name)) '=[']);
for i = 1:1:11
    fprintf('%.4f', RAD_dl(i)); fprintf(',');
end
fprintf('%.4f];', RAD_dl(12));
fprintf(' %% ');
fprintf('%.4f\n', mean(RAD_dl)); 
