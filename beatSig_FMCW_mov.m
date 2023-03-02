function sig = beatSig_FMCW_mov(scat_coeff, d, v_t, v_s, t, fc, T_sw, fr, c)
% beatSig_FMCW generates the beat signal of FMCW radar for point targets.
% 
% Parameters:
%   scat_coeff --- the scattering coefficients of targets 
%            d --- the distances of targets 
%          v_t --- the velocities of targets; positive along the direction
%          from radar to target
%          v_s --- the velocity of radar platform; direction as v_t;
%            t --- the time samples
%           fc --- the center frequency of the radar system
%         T_sw --- Time duration of an FMCW Sweep
%           fr --- Chirp rate: positive value for upchirp; negative for
%                              downchirp
%            c --- speed of light
%
% author: Jianping Wang @ MS3, TUDelft, the Netherlands
% email: jianpingwang87@gmail.com; or J.Wang-4@tudelft.nl
% 

if length(scat_coeff)~=length(d)
    error('Error. \n Unequal length vectors for target parameters');
end

NumTarget = length(d);

sig = zeros(1,length(t));
for k = 1:NumTarget
    tau = 2*( d(k) + (v_t(k) - v_s) * t )/c;
    sig = sig +  scat_coeff(k) * rectpuls(t-T_sw/2-tau, T_sw)...
        .* exp( 1i*2*pi*( -fc*tau + 0.5*fr*tau.^2 - fr * tau .* (t-T_sw/2) ) );
end