close all; clear; clc;
set(0, 'defaultTextInterpreter', 'latex'); 
Ftsz = 12;
mkdir('dataset/pic');
mkdir('dataset/pic_label');
n = 0;  % counting

%% Create hdf5 file
num_class_samples = 80;  % 80 | 20
num_total_samples = 72 * num_class_samples;

filename = 'dataset/FMCW_train.hdf5';  % FMCW_train.hdf5 | FMCW_test.hdf5
h5create(filename,'/X_real',[256 256 Inf],'ChunkSize',[256 256 1]);
h5create(filename,'/X_imag',[256 256 Inf],'ChunkSize',[256 256 1]);
h5create(filename,'/Y_real',[256 256 Inf],'ChunkSize',[256 256 1]);
h5create(filename,'/Y_imag',[256 256 Inf],'ChunkSize',[256 256 1]);

data1 = zeros(256,256,num_total_samples);
data2 = zeros(256,256,num_total_samples);
data3 = zeros(256,256,num_total_samples);
data4 = zeros(256,256,num_total_samples);

%% STFT parameters setting
W_len = 256;
Win = hamming(W_len,'periodic');
hop = 1;
Overlap_len = W_len-hop;

%% Constants setting
c = 3e8;                              % speed of light

%% FMCW radar system parameters setting
f_c = 3e9;                            % center frequency
T_sw = 400e-6;                        % duration of a sweep
BW = 40e6;                            % bandwidth
sweep_slope = -BW/T_sw;               % chirp rate
f_s = 12e6;                           % sampling frequency of beat signals
R_max = 8e3;                          % maximum detection distance
v_s = 30;                             % the velocity of radar platform

%% Parameters computed using the parameters above
tau_max = 2*R_max/c;                  % maximum delay time related to maximum detection distance
fb_max = abs(sweep_slope) * tau_max;  % maximum beat frequency

%% Data generation
num_SNR = 9;
for kk = 1:num_SNR
a = [-20 -15 -10 -5 0 5 10 15 20];  % SNR: (-20dB,20dB)
SNR = a(kk);
num_SINR = 4+(SNR+20)/5;

for hh = 1:num_SINR  % SINR: (-40dB,20dB)
disp(['Generating...SNR:' num2str(SNR) ...
      'dB,SINR:' num2str(-40+5*(hh-1)) 'dB~' num2str(-35+5*(hh-1)) 'dB']);
count = 0;  % counting

while count < num_class_samples
   %% Parameters of targets
    num_tar = round(20*rand(1, 1));                     % number of targets
    if num_tar ~= 0
        d_tar = (R_max-8)*rand(1, num_tar)+8;           % distances of targets (>=8m)
        amp_tar = 3*rand(1, num_tar);                   % amplitudes (or moduli) of targets' scattering coefficients
        phi_tar = 2*pi*rand(1, num_tar);                % phase of targets' scattering coefficients
        scat_coeff_tar = amp_tar .* exp(1i*phi_tar);    % complex scattering coefficients of targets
        v_t = 80*rand(1, num_tar);                      % the velocities of targets; positive along the direction
    else
        d_tar = 0;
        amp_tar = 0;
        phi_tar = 0;
        scat_coeff_tar = amp_tar .* exp(1i*phi_tar);
        v_t = 0;
    end

   %% Synthesize beat signals of targets
    t = 0:1/f_s:T_sw;
    sig_Rx_clean = beatSig_FMCW_mov(scat_coeff_tar, d_tar, v_t,...
                                    v_s, t, f_c, T_sw, sweep_slope, c);
    sig_Rx = awgn(sig_Rx_clean, SNR, 'measured');

   %% Parameters of interference
    num_intf = round(19*rand(1, 1))+1;                         % number of interference (>=1)
    amp_intf = 3*rand(1, num_intf);                            % amplitude of interference
    fc_intf = f_c*ones(1, num_intf);                           % center frequency of interference
    sweep_slope_intf = 4*sweep_slope*(rand(1, num_intf)-0.5);  % chirp rate of interference
    T_sw_intf = T_sw*rand(1, num_intf);                        % duration of interference
    t_d_intf = T_sw*(rand(1, num_intf)-0.5);                   % delay time of interference relative to the dechirp ref signal

   %% Synthesize beat signals related to interference
    sig_Int = beatInterfer_FMCW(amp_intf, fc_intf, sweep_slope_intf, T_sw_intf, t_d_intf,...
                                t, f_c, sweep_slope, T_sw, fb_max);

   %% Full signal after reception (beat signals of targets + noise + interference)
    sig_full = sig_Rx + sig_Int;

   %% Truncation (To keep the same range resolution for all targets)
    I_tWin = rectpuls(t-T_sw/2-tau_max,T_sw)>0.5;  % indices of time samples of the truncation window

    sig_full_trc = sig_full(I_tWin);
    sig_Rx_clean_trc = sig_Rx_clean(I_tWin);
    
   %% Calculate SINR
    for i = 1:15
        sig_full_trc_small = sig_full_trc(1+256*(i-1):511+256*(i-1));
        sig_Rx_clean_trc_small = sig_Rx_clean_trc(1+256*(i-1):511+256*(i-1));
        SINR = 10*log10(sum(power(abs(sig_Rx_clean_trc_small),2))/sum(power(abs(sig_full_trc_small - sig_Rx_clean_trc_small),2)));
        if SINR >= (-40+5*(hh-1)) && SINR <= (-35+5*(hh-1))
            count = count + 1;
            n = n + 1;

           %% STFT (beat signals of targets + noise + interference)
            [S_full,F,T] = stft(sig_full_trc,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
%             S_full_norm = S_full/max(abs(S_full(:)));

%             hf = figure;
%             imagesc(T*1e6,F/1e6,db(S_full_norm))
%             set(gca, 'YDir', 'normal')
%             xlabel('Time [$\mu$s]', 'interpreter', 'latex', 'fontsize', Ftsz);
%             ylabel('Frequency [MHz]', 'fontsize', Ftsz);
%             title('t-f diagram of interfered signal', 'fontsize', Ftsz);
%             set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
%             print(hf, '-dpdf', '-r300', ['dataset/pic/' num2str(n) '_3906.pdf'])
%             print(hf, '-dpng', '-r300', ['dataset/pic/' num2str(n) '_3906.png'])
%             saveas(hf, ['dataset/pic/' num2str(n) '_3906.fig'])

            S_full_small = S_full(:,1+256*(i-1):256*i);
%             S_full_small_norm = S_full_small/max(abs(S_full_small(:)));
% 
%             hf = figure;
%             imagesc(T*1e6,F/1e6,db(S_full_small_norm))
%             set(gca, 'YDir', 'normal')
%             xlabel('Time [$\mu$s]', 'interpreter', 'latex', 'fontsize', Ftsz);
%             ylabel('Frequency [MHz]', 'fontsize', Ftsz);
%             title('t-f diagram of interfered signal', 'fontsize', Ftsz);
%             set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
%             print(hf, '-dpdf', '-r300', ['dataset/pic/' num2str(n) '.pdf'])
%             print(hf, '-dpng', '-r300', ['dataset/pic/' num2str(n) '.png'])
%             saveas(hf, ['dataset/pic/' num2str(n) '.fig'])

            data1(:,:,n) = real(S_full_small);
            data2(:,:,n) = imag(S_full_small);
            
           %% STFT(beat signals of targets)
            [S_Rx_clean,F,T] = stft(sig_Rx_clean_trc,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
%             S_Rx_clean_norm = S_Rx_clean/max(abs(S_Rx_clean(:)));
% 
%             hf = figure;
%             imagesc(T*1e6,F/1e6,db(S_Rx_clean_norm))
%             set(gca, 'YDir', 'normal')
%             xlabel('Time [$\mu$s]', 'interpreter', 'latex', 'fontsize', Ftsz);
%             ylabel('Frequency [MHz]', 'fontsize', Ftsz);
%             title('t-f diagram of clean signal', 'fontsize', Ftsz);
%             set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
%             print(hf, '-dpdf', '-r300', ['dataset/pic_label/' num2str(n) '_3906.pdf'])
%             print(hf, '-dpng', '-r300', ['dataset/pic_label/' num2str(n) '_3906.png'])
%             saveas(hf, ['dataset/pic_label/' num2str(n) '_3906.fig'])

            S_Rx_clean_small = S_Rx_clean(:,1+256*(i-1):256*i);
%             S_Rx_clean_small_norm = S_Rx_clean_small/max(abs(S_Rx_clean_small(:)));
% 
%             hf = figure;
%             imagesc(T*1e6,F/1e6,db(S_Rx_clean_small_norm))
%             set(gca, 'YDir', 'normal')
%             xlabel('Time [$\mu$s]', 'interpreter', 'latex', 'fontsize', Ftsz);
%             ylabel('Frequency [MHz]', 'fontsize', Ftsz);
%             title('t-f diagram of clean signal', 'fontsize', Ftsz);
%             set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
%             print(hf, '-dpdf', '-r300', ['dataset/pic_label/' num2str(n) '.pdf'])
%             print(hf, '-dpng', '-r300', ['dataset/pic_label/' num2str(n) '.png'])
%             saveas(hf, ['dataset/pic_label/' num2str(n) '.fig'])

            data3(:,:,n) = real(S_Rx_clean_small);
            data4(:,:,n) = imag(S_Rx_clean_small);
            
%             close all;
            break;
        end
    end
end
end
end

disp("Data generation completed");
disp("--------------------------------");

%% write data to hdf5 file
data = zeros(256,256,2);
% *******************************************************************
for inum = 1:num_total_samples
      data(:,:,1) = data1(:,:,inum);
      start = [1 1 inum];
      count = [256 256 1];
      h5write(filename,'/X_real',data(:,:,1),start,count);
end
% *******************************************************************
for inum = 1:num_total_samples
      data(:,:,1) = data2(:,:,inum);
      start = [1 1 inum];
      count = [256 256 1];
      h5write(filename,'/X_imag',data(:,:,1),start,count);
end
% *******************************************************************
for inum = 1:num_total_samples
      data(:,:,1) = data3(:,:,inum);
      start = [1 1 inum];
      count = [256 256 1];
      h5write(filename,'/Y_real',data(:,:,1),start,count);
end
% *******************************************************************
for inum = 1:num_total_samples
      data(:,:,1) = data4(:,:,inum);
      start = [1 1 inum];
      count = [256 256 1];
      h5write(filename,'/Y_imag',data(:,:,1),start,count);
end
% *******************************************************************
disp("Data storage completed");
disp("--------------------------------");
h5disp(filename);

close all;