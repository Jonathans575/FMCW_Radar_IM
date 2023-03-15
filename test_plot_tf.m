close all; clear; clc;

set(0, 'defaultTextInterpreter', 'latex');
Ftsz = 12;

% radar parameters setting
c = 3e8;
T_sw = 400e-6;
BW = 40e6;
sweep_slope = BW/T_sw;
f_s = 12e6;

% ISTFT parameters setting
W_len = 256;
Win = hamming(W_len,'periodic');
hop = 1;
Overlap_len = W_len-hop;

flag_plot = 1;
dl_plot = 1;

% saving path
num_SNR = 9;  % number of SNR
num_SINR = 12;  % number of SINR
num_class_samples = 5;  % number of samples

filename  = 'dataset/FMCW_test_3906_5.hdf5';
inp_real  = h5read(filename,'/X_real');
inp_imag  = h5read(filename,'/X_imag');
oral_real = h5read(filename,'/Y_real');
oral_imag = h5read(filename,'/Y_imag');

% Open hdf5 file
filename  = 'dataset/model_b_0_fix_11_32_3_EighthData_3906.hdf5';
disp(file_name);
pred_real = h5read(filename,'/Y_real');
pred_imag = h5read(filename,'/Y_imag');

saving_path = ['plot/' file_name '/'];
mkdir(saving_path);

%% Test
SINR_inp = []; SINR_dl = [];
P_inp = []; P_dl = [];
RAD_inp = []; RAD_dl = [];

for p = 1:num_SINR

count = 0;
sinr_inp = 0; sinr_dl = 0; 
p_inp = 0; p_dl = 0; 
rad_inp = 0; rad_dl = 0;

file_folder = [saving_path int2str(-40+5*(p-1)) '_' int2str(-35+5*(p-1)) '/'];
mkdir(file_folder);
disp('------------------------------');

a = max(p-3,1);
for i = a:num_SNR
    disp(['正在处理...SINR:' num2str(-40+5*(p-1)) 'dB~' num2str(-35+5*(p-1)) 'dB,' ...
          'SNR:' num2str(-20+5*(i-1)) 'dB']);
    starting = 0;
    temp = 4;
    for hh = 2:i
        starting = starting + temp;
        temp = temp + 1;
    end
    starting = starting * num_class_samples;
    
    for j = num_class_samples*(p-1)+1 + starting : num_class_samples*p + starting
        
        count = count + 1;
        
        sig_TF_inp = inp_real(:,:,j) + 1i * inp_imag(:,:,j);
        sig_TF_oral = oral_real(:,:,j) + 1i * oral_imag(:,:,j);

        % ISTFT
        [sig_inp, t_inp]   = istft(sig_TF_inp,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
        [sig_oral, t_oral] = istft(sig_TF_oral,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);

        if flag_plot == true
            hf = figure;
            plot(t_inp*1e6,real(sig_inp));
            grid on
            set(gca,'XLim',[0 350]);
            xlabel('Time [$\mu$s]','fontsize',Ftsz);
            ylabel('Amplitude','fontsize',Ftsz)
            title('Real part','fontsize',Ftsz)
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
            print(hf, '-dpdf', '-r300', [file_folder 'sig_inp_RealPart_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_inp_RealPart_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_inp_RealPart_' num2str(j) '.fig'])

            hf = figure;
            plot(t_oral*1e6,real(sig_oral));
            grid on
            set(gca,'XLim',[0 350]);
            xlabel('Time [$\mu$s]','fontsize',Ftsz);
            ylabel('Amplitude','fontsize',Ftsz)
            title('Real part','fontsize',Ftsz)
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
            print(hf, '-dpdf', '-r300', [file_folder 'sig_oral_RealPart_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_oral_RealPart_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_oral_RealPart_' num2str(j) '.fig'])
        end
        
        % FFT
        sig_fft_inp  = flip(fft(sig_inp));
        sig_fft_inp_norm = abs(sig_fft_inp)/max(abs(sig_fft_inp));
        
        sig_fft_oral = flip(fft(sig_oral));
        sig_fft_oral_norm = abs(sig_fft_oral)/max(abs(sig_fft_oral));
        
        N_sig = length(sig_inp);
        r_axis = f_s/N_sig*(0:N_sig/2-1)/abs(sweep_slope)*c/2;
        
        if flag_plot == true         
            hf = figure;
            plot(r_axis/1e3, db(sig_fft_inp_norm(1:N_sig/2)), 'Color', '#FF0000','LineStyle','-'); hold on;
            plot(r_axis/1e3, db(sig_fft_oral_norm(1:N_sig/2)), 'Color', '#00008B','LineStyle','--'); hold on;
            grid on
            xlabel('Range [km]', 'fontsize', Ftsz)
            ylabel('Normalized amplitude [dB]', 'fontsize', Ftsz)
            title('Range profiles', 'fontsize', Ftsz)
            h = legend('interfered','ref sig',...
                       'location','southeast');
            set(h,'interpreter','latex')
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3]);
            print(hf, '-dpdf', '-r300', [file_folder 'sig_fft_oral_inp_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_fft_oral_inp_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_fft_oral_inp_' num2str(j) '.fig'])
        end
        
        % STFT domain
        [sig_TF_inp,f_slice,t_frame] =  stft(sig_inp,f_s,'Window',...
                                         Win,'OverlapLength',...
                                         Overlap_len,'FFTLength',...
                                         W_len);
        sig_TF_inp_norm = sig_TF_inp/max(abs(sig_TF_inp(:)));

        if flag_plot == true
            hf = figure;
            imagesc(t_frame*1e6,f_slice/1e6,db(sig_TF_inp_norm))
            colorbar;
            caxis([-100,0]);
            set(gca, 'YDir', 'normal')
            xlabel('Time [$\mu$s]', 'fontsize', Ftsz);
            ylabel('Frequency [MHz]', 'fontsize', Ftsz);
            title('t-f diagram of interfered signal', 'fontsize', Ftsz);
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
            print(hf, '-dpdf', '-r300', [file_folder 'sig_TF_inp_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_TF_inp_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_TF_inp_' num2str(j) '.fig'])
        end
        
        [sig_TF_oral,f_slice,t_frame] =  stft(sig_oral,f_s,'Window',...
                                         Win,'OverlapLength',...
                                         Overlap_len,'FFTLength',...
                                         W_len);      
        sig_TF_oral_norm = sig_TF_oral/max(abs(sig_TF_oral(:)));
        
        if flag_plot == true
            hf = figure;
            imagesc(t_frame*1e6,f_slice/1e6,db(sig_TF_oral_norm))
            caxis([-100,0]);
            set(gca, 'YDir', 'normal')
            xlabel('Time [$\mu$s]', 'fontsize', Ftsz);
            ylabel('Frequency [MHz]', 'fontsize', Ftsz);
            title('t-f diagram of clean signal', 'fontsize', Ftsz);
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
            print(hf, '-dpdf', '-r300', [file_folder 'sig_TF_oral_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_TF_oral_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_TF_oral_' num2str(j) '.fig'])
        end
       
       %% recovered signal
        sig_TF_pred = pred_real(:,:,j) + 1i * pred_imag(:,:,j);
        sig_TF_pred_norm = sig_TF_pred/max(abs(sig_TF_pred(:)));
       
        if dl_plot == true
            hf = figure;
            imagesc(t_frame*1e6,f_slice/1e6, db(sig_TF_pred_norm))
            caxis([-100,0]);
            set(gca, 'YDir', 'normal')
            xlabel('Time [$\mu$s]', 'fontsize', Ftsz);
            ylabel('Frequency [MHz]', 'fontsize', Ftsz);
            title(['recovered signal (CV-FCN $\lambda$=' b ')'], 'fontsize', Ftsz);
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
            print(hf, '-dpdf', '-r300', [file_folder 'sig_TF_pred_b_' b '_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_TF_pred_b_' b '_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_TF_pred_b_' b '_' num2str(j) '.fig'])
        end

        % ISTFT
        [sig_pred_cal, t_pred] = istft(sig_TF_pred,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);

        if dl_plot == true
            hf = figure;
            plot(t_pred*1e6,real(sig_pred_cal));
            grid on
            set(gca,'XLim',[0 350]);
            xlabel('Time [$\mu$s]','fontsize',Ftsz);
            ylabel('Amplitude','fontsize',Ftsz)
            title('Real part','fontsize',Ftsz)
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
            print(hf, '-dpdf', '-r300', [file_folder 'sig_pred_RealPart_b_' b '_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_pred_RealPart_b_' b '_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_pred_RealPart_b_' b '_' num2str(j) '.fig'])
        end
        
        % FFT
        sig_fft_pred  = flip(fft(sig_pred_cal));
        sig_fft_pred_norm = abs(sig_fft_pred)/max(abs(sig_fft_pred));
        
        if dl_plot == true
            hf = figure;
            plot(r_axis/1e3, db(sig_fft_pred_norm(1:N_sig/2)))
            grid on
            xlabel('Range [km]', 'fontsize', Ftsz)
            ylabel('Normalized amplitude [dB]', 'fontsize', Ftsz)
            title('Range profiles', 'fontsize', Ftsz)
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3]);
            print(hf, '-dpdf', '-r300', [file_folder 'sig_fft_pred_b_' b '_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_fft_pred_b_' b '_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_fft_pred_b_' b '_' num2str(j) '.fig'])
        end
        
       %% recovered signal
        sig_TF_pred = pred_real(:,:,j) + 1i * pred_imag(:,:,j);
        % ISTFT
        [sig_pred, t_pred] = istft(sig_TF_pred,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
        % FFT
        sig_fft_pred  = flip(fft(sig_pred));
        sig_fft_pred_norm_real = abs(sig_fft_pred)/max(abs(sig_fft_pred));
        
       %% Plot
        if dl_plot == true
            hf = figure;
            plot(r_axis/1e3, db(sig_fft_oral_norm(1:N_sig/2)),     'Color', '#FF0000','LineStyle','-'); hold on;
            plot(r_axis/1e3, db(sig_fft_pred_norm_real(1:N_sig/2)),'Color', '#FFB90F','LineStyle','--'); hold on;
            grid on
            h = legend('ref sig','CV-FCN','location','Northeast');
            set(h,'interpreter','latex')
            xlabel('Range [km]', 'fontsize', Ftsz)
            ylabel('Normalized amplitude [dB]', 'fontsize', Ftsz)
            title('Range profiles', 'fontsize', Ftsz)
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3]);
            print(hf, '-dpdf', '-r300', [file_folder 'sig_fft_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [file_folder 'sig_fft_' num2str(j) '.png'])
            saveas(hf, [file_folder 'sig_fft_' num2str(j) '.fig'])
        end
        
        % calculate SINR
        temp = 10*log10(sum(power(abs(sig_oral),2))/sum(power(abs(sig_inp - sig_oral),2)));
        sinr_inp = sinr_inp + temp;

        temp = 10*log10(sum(power(abs(sig_oral),2))/sum(power(abs(sig_pred_cal - sig_oral),2)));
        sinr_dl = sinr_dl + temp;
        
        % calculate p and rad
        temp = (sig_inp' * sig_oral)/...
               (sqrt(sum(sig_oral .* conj(sig_oral))) * sqrt(sum(sig_inp .* conj(sig_inp))));
        p_inp = p_inp + abs(temp);
        rad_inp = rad_inp + abs(angle(temp));
        
        temp = (sig_pred_cal' * sig_oral)/...
               (sqrt(sum(sig_oral .* conj(sig_oral))) * sqrt(sum(sig_pred_cal .* conj(sig_pred_cal))));
        p_dl = p_dl + abs(temp);
        rad_dl = rad_dl + abs(angle(temp));

        close all;
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

clear;
close all;