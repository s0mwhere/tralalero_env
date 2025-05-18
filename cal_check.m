clear all; close all; clc;

% ------------------------ SET DATA ----------------------------
usr1_distance=62.62969346980914;
usr1_num_path=13;
usr1_AoD=[0.23274911236094167,0.6151019656739172,2.6037537055030975,0.6321919004247489,2.0716982204954384,1.815880081920995,1.9093069368421298,0.43911948714130616,2.599664871389113,2.644413454862484,0.8396979693644971,1.6897059808751103,2.9226021200761623];
usr1_path_gain_distribution=[(0.0003334802026710823-0.0006536918556126807j),(-0.00032893048752469787+0.0009382752474995068j),(0.00017699304060362164+0.0005846907829913553j),(-0.00011686797744041862-0.00016053965596708525j),(0.0007849963873072333+0.0005814536433202831j),(-0.00098337346368133-0.0007739258213561682j),(-0.00031370352163051366-0.00014684920225323004j),(0.0007429011060502619+0.00033498280733969445j),(-0.0004435728589403585-0.0007955973564085109j),(0.0018132956585964228-0.0006010622893542323j),(0.0007044975100375842+6.903660909956658e-05j),(0.0001222472907889994-2.8543106561250242e-06j),(-8.133969428708005e-05-0.00012452931971109358j)];
targ1_AoD=1.0471975511965976;
targ1_atten_coeff=0.00030395264106727123;
targ1_doppler_freq=0.00030395264106727123;
targ1_rx_posit=0.31557369248965583;
sampling_period =1e-06;
clut_channel_modl=[(-0.00019493314084519006-3.269822149949792e-05j),(-0.00018712052163702593+6.367113530704668e-05j),(-0.00015283439323717703-0.00012533854687535223j),(-0.00018821644146206065+6.035457109747055e-05j),;(0.00012388384048066248-0.00022227647213183556j),(0.00023763941592073402-9.100298894248078e-05j),(0.00023748388011948885-9.140811057562204e-05j),(0.00021533330745095655+0.00013559352011746257j),;];
antenna_array=[0.0,0.24,0.51,0.74];
beam_matrix =[;(0.09045898162131781-0.15039101010919137j),(0.12610977385020034+0.002195687005684229j),(0.1020117942853089+0.052084224852607414j),(-0.14899729655351432-0.45034703985245594j),;(-0.14987097430332497-0.16327135039900442j),(0.10394388422470885+0.2000192689999228j),(-0.14345842677075585+0.02138589096350165j),(0.09138918419116324-0.4443998968789158j),;(0.14491636433288121+0.18298301688640442j),(-0.434124496416333+0.2443916706455454j),(0.006751968436352727+0.04844752175903683j),(0.5186364905986183-0.215476312025967j),;(-0.08512957939657984+0.32693735083979836j),(0.15806721888439101-0.04810075395867694j),(0.3254147910128548+0.2152773534836409j),(0.05648402550594795+0.1975945969729462j)];
split_fact = 0.39479991683034055;

num_of_tx_antenna = 4;
num_or_user = 3;
path_loss_at_ref_distance_1m = 1;
alpha_path_loss_exponent = 2.8;
lambda_wave = 0.1; %[w]
sigma_noise = 1e-11;
dlambda = 0.1;   %wave length in [m]
A_antennaRange = 10*dlambda;  % movement range of antenna elements in [m]
D0_spacingAntenna = dlambda/2;  % minimum space between 2 antenna element in [m]

P0_basestation_power = 1.5848931924611136; 
% ---------------------------- Calculation

% --- user 1 cal infor
usr1_peak_gain = cal_peak_gain(usr1_distance, usr1_num_path);

usr1_arr_response_vector = cal_ARV(usr1_num_path, num_of_tx_antenna, antenna_array, usr1_AoD);

usr1_channel_vector = cal_channel_vector(usr1_num_path, usr1_arr_response_vector, usr1_path_gain_distribution);

% --- clutter 1 cal
targ1_arr_response_vector = cal_ARV(1, num_of_tx_antenna, antenna_array,targ1_AoD);

targ1_channel_modl = targ1_atten_coeff*exp(1j*2*pi*targ1_doppler_freq*sampling_period)*exp(1j*(2*pi/dlambda)*targ1_rx_posit*cos(targ1_AoD))*targ1_arr_response_vector;

% user 1 data rate
usr1_channel_vector_h_transpose = usr1_channel_vector';

numerator = (1-split_fact)*(norm(dot(usr1_channel_vector_h_transpose,reshape(beam_matrix(:,1),num_or_user+1,1))))^2;

denominator = 0 ;
for i = 2:num_or_user
    denominator = (1-split_fact)*(norm(dot(usr1_channel_vector_h_transpose,reshape(beam_matrix(:,i),num_or_user+1,1))))^2 +sigma_noise;
end
usr1_data_rate = log2(1+ numerator/denominator);

%SCNR
SCNR_denom1 = 0;
SCNR_denom2 = 0;

SCNR_numer = (norm(dot(conj(targ1_channel_modl), reshape((beam_matrix(:, end)),num_or_user+1, 1))))^2;

for i = 1:num_or_user
    SCNR_denom1 = SCNR_denom1 + (norm(dot(conj(targ1_channel_modl), reshape((beam_matrix(:, i)),num_or_user+1, 1))))^2;
end

for i = 1:2
    for n = 1: num_or_user+1
        SCNR_denom2 = SCNR_denom2 + (norm(dot(conj(clut_channel_modl(1,:)), reshape((beam_matrix(:, i)),num_or_user+1, 1))))^2;
    end
end
SCNR = SCNR_numer/(SCNR_denom1+SCNR_denom2+sigma_noise);


cal_beam_power = trace(conj(beam_matrix')*beam_matrix);

% ----------------------- Cal functions

function peak_gain = cal_peak_gain(distance, number_of_path)      %path_gain_var
    the_path_loss_at_ref_distance_1m = 1;
    the_alpha_path_loss_exponent = 2.8;
    peak_gain = the_path_loss_at_ref_distance_1m * (distance.^(-1*the_alpha_path_loss_exponent) / number_of_path );
end

function array_response_vector = cal_ARV(num_path, num_antenna, antenna_array, AoD_array)
    array_response_vector = [];
    temp_single_angle_array = [];
    lambda_wave = 0.1;
    for i=1:num_path
        for j=1:num_antenna 
            temp_single_angle_array(j) = exp(  (1i*2*pi*antenna_array(j)*cos(AoD_array(i)))/ lambda_wave);
        end
        array_response_vector = [array_response_vector;temp_single_angle_array];
    end
end

function channel_vector = cal_channel_vector(number_of_path, array_response_vector, path_gain_distribution)
    channel_vector = array_response_vector(1,:)*path_gain_distribution(1);
    for i=2:number_of_path
        channel_vector = channel_vector + (array_response_vector(i,:)*path_gain_distribution(i));
    end
end