# FMCW_Radar_IM  
The code about deep-learning based FMCW Radar interference mitigation.  

Our paper:  
[1] J. Wang, R. Li, Y. He and Y. Yang, "Prior-Guided Deep Interference Mitigation for FMCW Radars," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5118316.  
[2] R. Li, J. Wang, Y. He, Y. Yang and Y. Lang, "Deep Learning for Interference Mitigation in Time-Frequency Maps of FMCW Radars," 2021 CIE International Conference on Radar (Radar), Haikou, Hainan, China, 2021, pp. 1883-1886.  
To read the paper, see:  
[1] https://ieeexplore.ieee.org/document/9908588  
[2] https://ieeexplore.ieee.org/document/10028226  

The matlab scripts in this fold are used to generate the beat signals acquired with FMCW radar system.  
Details:  
[1] "beatSig_FMCW.m" synthesizes the useful beat signals scattered from point targets.  
[2] "beatSig_FMCW_mov.m" synthesizes the useful beat signals scattered from point moving targets.  
[3] "beatInterfer_FMCW.m" generates the beat signals related to interference after the dechirping and low-pass filtering.  
[4] "main.m" is a demon for the full signal generation, where the "beatSig_FMCW_mov" and "beatInterfer_FMCW" are used.  
[5] "test.m" calculates the SINR and correlation coefficient of interfered signal and predicted signal.  
[6] "test_plot_tf.m" is used to plot the signal's waveform, frequency spectrum, and t-f diagram, etc.  
[7] "./realdata/realdata_make.m" generates the dataset of real-world radar interfered signals for testing.  
[8] "./realdata/test_realdata.m" is similar to "test.m", but it is applicable to measured signals.  

The python scripts in this fold are used for training and testing the interference mitigation models.  
[1] "complexnn/*" includes a variety of the basic complex-valued modules.  
[2] "train_256.py" is used to train the interference mitigation models.  
[3] "test.py" is used for testing.  

%==========================================================  
% Contact: Jianping Wang,     J.Wang-4@tudelft.nl  
% Contact: Runlong Li,     lirunlong@bupt.edu.cn  
%==========================================================  
