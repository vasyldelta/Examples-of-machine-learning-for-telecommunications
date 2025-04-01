function [out_complex] = demodulator_qpsk_lite(z,M,Fs,Fd)

N_samp_inp = Fs/Fd;
W_const = 2/N_samp_inp;

vi_phase = 0;
err_phase = 0;
nco_out = complex(1);
Psi = 0;

W = 1/N_samp_inp;
error = 0;
NCO = 1/N_samp_inp;
TEDBuff = complex([0 0]);
vi = 0;
symb_now = 1;
buff_y_in = complex([0 0 0]);
v = 0;

bT = 1e-3;
Kd = 0.17;

if M == 2
    Kd = 2.54;%2;%2.54
    bT= 1e-3;
elseif M == 4
    Kd = 0.17;%0.2042;%0.17
    bT = 0.5*10^-4;
end

Kd_phase = 1; % �������� ������� �������������
bT_phase = 8*10^-3;%1e-3;%8*10^-3; % ������������� ������� ������
[kp_phase,ki_phase] = loop_filter_coef(bT_phase,2,Kd_phase);
[kp,ki] = loop_filter_coef(bT,N_samp_inp,Kd);%%

len_buffer_out_max = 10000000;
y_out_complex = complex(zeros(len_buffer_out_max,1));
len_z = length(z);

ind_comp = 1;

% z = z/std(z);

for nnn = 1:len_z
    
    y_in = z(nnn);
    
    if NCO <= 0
        
        mu = NCO/W+1;
        
        NCO = mod(NCO,1);
        if NCO==0
            NCO=1;
        end
        
        %---------------------������������----------------------------
        int_out = farrow(mu, [y_in, buff_y_in(1:3)]);
        %--------------------------------------------------------------
        
        %% -----------------bursts �������������------------------------
        
        int_out_2 = int_out*nco_out; % ���������
        
        %% ----------------------------------------------------------------
        % �������� �� ������ ������� (������������ � 2 ������/������ � 1 ������/������)
        if symb_now == 1
            
            error = (real(TEDBuff(1)) * (sign(real(TEDBuff(2))) - sign(real(int_out))) +...
                imag(TEDBuff(1)) * (sign(imag(TEDBuff(2))) - sign(imag(int_out))));
            
            Q = imag(int_out_2);
            I = real(int_out_2);
            
            if M == 2
                err_phase = Q*I;
            elseif M == 4
                err_phase = (Q^3*I-I^3*Q);
            end

            y_out_complex(ind_comp) = int_out_2;
            ind_comp = ind_comp + 1;
            
            %-----------��������� ������ - ������---------------------
            symb_now = 0;
            %---------------------------------------------------------
            
        else
            %-----------��������� ������ - �������������--------------
            symb_now = 1;
            %---------------------------------------------------------
        end
        %----������ � ����� ���������� 3-� �������� ��� ���������� ������ �����������------
        TEDBuff = [int_out, TEDBuff(1)];
        %--------------------------------------------------------------
        
        %------------�������� ������ - 1 (��� ����� ��������)--------
        [vi,v] = loop_filter(kp,ki,error,vi);
        %------------------------------------------------------------
        
        %------------�������� ������ (��� ����� ������� �������������)--------
        [vi_phase,v_phase] = loop_filter(kp_phase,ki_phase,err_phase,vi_phase);
        %----------------------------------------------------------
        
        %-------------------NCO ����� ������� �������������--------------
        Psi = Psi+v_phase;
        nco_out = exp(1i*Psi);
        %----------------------------------------------------------------
    end
    
    
    % ����� �������� ��������� 3-� ������� ��������
    buff_y_in = [y_in,buff_y_in(1:2)];
    
    %---------------���� ���������� + NCO ����� ��������---------------
    % ����������� �����
    W = W_const + v;
    
    % ������� �������� NCO
    NCO = NCO - W;
    %------------------------------------------------------------------
end

out_complex = y_out_complex(1:ind_comp-1);