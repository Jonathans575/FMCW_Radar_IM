close all; clear; clc;
n = 0;  % counting

%% STFT parameters setting
W_len = 256;
Win = hamming(W_len,'periodic');
hop = 1;
Overlap_len = W_len-hop;
    
%% Create hdf5 file
number = 12;
class = 'chimney';

filename = strcat('../dataset/',class,'_small.hdf5');
h5create(filename,'/X_real',[256 16129 Inf],'ChunkSize',[256 16129 1]);
h5create(filename,'/X_imag',[256 16129 Inf],'ChunkSize',[256 16129 1]);

data1 = zeros(256,16129,number);
data2 = zeros(256,16129,number);

for i = 1:number
    %% load data
    path = strcat('../../PARSAX measurements interference/HV-signal30MHz-chimney/Mat/sig_HV_',int2str(i),'.mat');
    a = load(path);
    a = a.complex;
    size(a)
    
    %% preprocessing
    len = length(a);
    a_fft = fft(a,32768);
    a_fft(1:30) = 0;  % error data
    a = ifft(a_fft,32768);
    a = a(1:len);

    %% STFT
    [S,F,T] = stft(a,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
    
    n = n + 1;
    disp(n);
    data1(:,:,n) = real(S);
    data2(:,:,n) = imag(S);
end

%% write data to hdf5 file
data = zeros(256,16129,2);
% *******************************************************************
for inum = 1:number
      data(:,:,1) = data1(:,:,inum);
      start = [1 1 inum];
      count = [256 16129 1];
      h5write(filename,'/X_real',data(:,:,1),start,count);
end
disp("writing to '/X_real' is over");
% *******************************************************************
for inum = 1:number
      data(:,:,1) = data2(:,:,inum);
      start = [1 1 inum];
      count = [256 16129 1];
      h5write(filename,'/X_imag',data(:,:,1),start,count);
end
disp("writing to '/X_imag' is over");
% *******************************************************************
disp("Data storage completed");
disp("--------------------------------");
h5disp(filename);

close all;