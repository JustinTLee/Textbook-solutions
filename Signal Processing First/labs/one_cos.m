function [x, t] = one_cos(A, w, phi, dur)
%ONE_COS Generates a sinusoidal signal and corresponding times
%   Inputs:
%     - amplitude (A)
%     - radial frequency (w)
%     - phase (phi)
%     - duration (dur)

t = linspace(0, dur, 20);
x = A*cos(w*t + phi);

end

