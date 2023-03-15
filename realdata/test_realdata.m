close all; clear; clc;
set(0, 'defaultTextInterpreter', 'latex');
Ftsz = 12;

plt_fft = 0;
flag_plot = 1;  % 1:plot figures; 0:not
scenario = 'windmill';

%% radar parameters setting
c = 3e8;
T_sw = 1e-3;
BW = 30e6;
sweep_slope = BW/T_sw;
f_s = 20e6;

%% ISTFT parameters setting
W_len = 256;
Win = hamming(W_len,'periodic');
hop = 1;
Overlap_len = W_len-hop;

%% Open hdf5 file
if strcmp(scenario, 'chimney')
    filename  = strcat('../dataset/realdata/',scenario,'_small.hdf5');
else
    filename  = strcat('../dataset/realdata/',scenario,'.hdf5');
end
inp_real  = h5read(filename,'/X_real');
inp_imag  = h5read(filename,'/X_imag');

b = '0';
filename  = ['../dataset/realdata/pred_' scenario '_' b '.hdf5'];
pred_real = h5read(filename,'/Y_real');
pred_imag = h5read(filename,'/Y_imag');

%% Saving path
class = [scenario '_b_' b];
mkdir(class);
mkdir([class '/t']);
mkdir([class '/fft']);

%% Plot figures
for j = 1:12
    %% measured radar signal (interfered)
    S = inp_real(:,:,j) + 1i * inp_imag(:,:,j);
    
    % ISTFT
    [sig_inp, t_inp] = istft(S,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
    if flag_plot == true
        hf = figure;
        plot(t_inp*1e6,real(sig_inp));
        grid on
        axis tight
        xlabel('Time [$\mu$s]','fontsize',Ftsz);
        ylabel('Amplitude','fontsize',Ftsz);
        title('Real part','fontsize',Ftsz);
        set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
        print(hf, '-dpdf', '-r300', [class '/t/' 'sig_inp_RealPart_' num2str(j) '.pdf'])
        print(hf, '-dpng', '-r300', [class '/t/' 'sig_inp_RealPart_' num2str(j) '.png'])
        saveas(hf, [class '/t/' 'sig_inp_RealPart_' num2str(j) '.fig'])
    end
    
    % fft
    sig_fft_inp = fft(sig_inp,32768);
    sig_fft_inp_norm = abs(sig_fft_inp)/max(abs(sig_fft_inp));
    sig_fft_inp_norm = sig_fft_inp_norm(1:8192);
    
    N_sig = length(sig_fft_inp);
    r_axis = f_s/N_sig*(0:8191)/abs(sweep_slope)*c/2;
    
    if flag_plot == true
        hf = figure;
        plot(r_axis/1e3, db(sig_fft_inp_norm))
        grid on
        xlabel('Range [km]', 'fontsize', Ftsz)
        ylabel('Normalized amplitude [dB]', 'fontsize', Ftsz)
        title('Range profiles', 'fontsize', Ftsz)
        set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3]);
        print(hf, '-dpdf', '-r300', [class '/fft/' 'sig_fft_inp_' num2str(j) '.pdf'])
        print(hf, '-dpng', '-r300', [class '/fft/' 'sig_fft_inp_' num2str(j) '.png'])
        saveas(hf, [class '/fft/' 'sig_fft_inp_' num2str(j) '.fig'])
    end
    
    % STFT domain
    [sig_TF_inp,f_slice,t_frame] =  stft(sig_inp,f_s,...
                                         'Window',Win,...
                                         'OverlapLength',Overlap_len,...
                                         'FFTLength',W_len);
    sig_TF_inp_norm = sig_TF_inp/max(abs(sig_TF_inp(:)));
    
    if flag_plot == true
        hf = figure;
        imagesc(t_frame*1e6,f_slice/1e6,db(sig_TF_inp_norm))
        colorbar;
        set(gca, 'YDir', 'normal')
        xlabel('Time [$\mu$s]', 'fontsize', Ftsz);
        ylabel('Frequency [MHz]', 'fontsize', Ftsz);
        title('t-f diagram of interfered signal', 'fontsize', Ftsz);
        set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
        print(hf, '-dpdf', '-r300', [class '/sig_TF_inp_' num2str(j) '.pdf'])
        print(hf, '-dpng', '-r300', [class '/sig_TF_inp_' num2str(j) '.png'])
        saveas(hf, [class '/sig_TF_inp_' num2str(j) '.fig'])
    end
    
    %% recovered radar signal
    S = pred_real_cal(:,:,j) + 1i * pred_imag_cal(:,:,j);
    
    % ISTFT
    [sig_pred, t_pred] = istft(S,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
    if flag_plot == true
        hf = figure;
        plot(t_pred*1e6,real(sig_pred));
        grid on
        axis tight
        xlabel('Time [$\mu$s]','fontsize',Ftsz);
        ylabel('Amplitude','fontsize',Ftsz)
        title('Real part','fontsize',Ftsz)
        set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
        print(hf, '-dpdf', '-r300', [class '/t/' 'sig_pred_RealPart_' num2str(j) '.pdf'])
        print(hf, '-dpng', '-r300', [class '/t/' 'sig_pred_RealPart_' num2str(j) '.png'])
        saveas(hf, [class '/t/' 'sig_pred_RealPart_' num2str(j) '.fig'])
    end
    
    % fft
    sig_fft_pred = fft(sig_pred,32768);
    sig_fft_pred_norm = abs(sig_fft_pred)/max(abs(sig_fft_pred));
    sig_fft_pred_norm = sig_fft_pred_norm(1:8192);
    
    N_sig = length(sig_fft_pred);
    r_axis = f_s/N_sig*(0:8191)/abs(sweep_slope)*c/2;
    
    if flag_plot == true
        hf = figure;
        plot(r_axis/1e3, db(sig_fft_pred_norm))
        grid on
        xlabel('Range [km]', 'fontsize', Ftsz)
        ylabel('Normalized amplitude [dB]', 'fontsize', Ftsz)
        title('Range profiles', 'fontsize', Ftsz)
        set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3]);
        print(hf, '-dpdf', '-r300', [class '/fft/' 'sig_fft_pred_' num2str(j) '.pdf'])
        print(hf, '-dpng', '-r300', [class '/fft/' 'sig_fft_pred_' num2str(j) '.png'])
        saveas(hf, [class '/fft/' 'sig_fft_pred_' num2str(j) '.fig'])
    end
    
    % STFT domain
    [sig_TF_pred,f_slice,t_frame] =  stft(sig_pred,f_s,...
                                         'Window',Win,...
                                         'OverlapLength',Overlap_len,...
                                         'FFTLength',W_len);
    sig_TF_pred_norm = sig_TF_pred/max(abs(sig_TF_pred(:)));
    
    if flag_plot == true
        hf = figure;
        imagesc(t_frame*1e6,f_slice/1e6,db(sig_TF_pred_norm))
        set(gca, 'YDir', 'normal')
        xlabel('Time [$\mu$s]', 'fontsize', Ftsz);
        ylabel('Frequency [MHz]', 'fontsize', Ftsz);
        title(['recovered signal (CV-FCN $\lambda$=' b ')'], 'fontsize', Ftsz);
        set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
        print(hf, '-dpdf', '-r300', [class '/sig_TF_pred_' num2str(j) '.pdf'])
        print(hf, '-dpng', '-r300', [class '/sig_TF_pred_' num2str(j) '.png'])
        saveas(hf, [class '/sig_TF_pred_' num2str(j) '.fig'])
    end
    
    if plt_fft == true
        %% recovered signal
        S = pred_real(:,:,j) + 1i * pred_imag(:,:,j);
        % ISTFT
        [sig_pred, t_pred] = istft(S,f_s,'Window',Win,'OverlapLength',Overlap_len,'FFTLength',W_len);
        % fft
        sig_fft_pred = fft(sig_pred,32768);
        sig_fft_pred_norm = abs(sig_fft_pred)/max(abs(sig_fft_pred));
        sig_fft_pred_norm = sig_fft_pred_norm(1:8192);
    
        % plot the Range profiles
        if flag_plot == true
            hf = figure;
            plot(r_axis/1e3, db(sig_fft_inp_norm), 'Color', '#FF0000','LineStyle','-'); hold on; % red
            plot(r_axis/1e3, db(sig_fft_pred_norm), 'Color', '#FFB90F','LineStyle','--'); hold on; % green
            grid on
            xlabel('Range [km]', 'fontsize', Ftsz)
            ylabel('Normalized amplitude [dB]', 'fontsize', Ftsz)
            title('Range profiles', 'fontsize', Ftsz)
            h = legend('Without IM','CV-FCN $\lambda$=0','location','south','NumColumns',2);
            set(h,'interpreter','latex')
            set(gcf, 'PaperUnits', 'inches', 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3]);
            print(hf, '-dpdf', '-r300', [class '/fft/' 'sig_fft_pred_' num2str(j) '.pdf'])
            print(hf, '-dpng', '-r300', [class '/fft/' 'sig_fft_pred_' num2str(j) '.png'])
            saveas(hf, [class '/fft/' 'sig_fft_pred_' num2str(j) '.fig'])
        end
    end
    
    close all;
end

close all;