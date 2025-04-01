function [vi,v] = loop_filter(kp,ki,error,vi)

vp = kp*error;
vi = vi + ki*error;
v = vp + vi;