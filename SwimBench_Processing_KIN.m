% SWIM BENCH PROCESSING - Kat Webster (last modified: MAY 23, 2023)
% INPUTS = Raw cal & task files
%   rotates to ISB, recreates markers using LCS based, filters, pads,
%   combines processed data into new table & exports
%   averages 3 setting trials into 1 & exports
% OUTPUT = Processed task trials (individual & averaged)

% CLEAR COMMAND WINDOW & WORKSPACE
clear
clc

% INPUT SUBJECT
subject_num = "S01M";

% DEFINE SAMPLE RATE
    fs = 150; % mocap sample rate = 150 Hz

%% ------------------------- CALIBRATION ------------------------------- %%
% SET UP CAL DIRECTORY  
    CAL_dir = "C:\Users\kathr\Documents\Waterloo MSc\Thesis\Data\KINEMATICS\" ...
        + subject_num + "_KIN\CAL\";
    
    CAL_file_list = dir (CAL_dir + subject_num + "*.csv");

% READ IN CAL FILE
    CAL_raw = readmatrix(CAL_dir + subject_num + '_CAL');

% DEFINE CAL FRAME WITH ALL MARKERS AVAILABLE
    % Input for each participant from SB_Vicon Processing doc
    CAL_frame = 1 + 3; % row corresponding to good frame + 3 empty frames

% DEFINE CAL MARKERS
    % 3rd Distal Phalanx
    R_DP3_cal_raw = CAL_raw(CAL_frame, 3:5);
    L_DP3_cal_raw = CAL_raw(CAL_frame, 102:104);

    % 2nd Metacarpal
    R_MCP2_cal_raw = CAL_raw(CAL_frame, 6:8);
    L_MCP2_cal_raw = CAL_raw(CAL_frame, 99:101);

    % 5th Metacarpal
    R_MCP5_cal_raw = CAL_raw(CAL_frame, 9:11);
    L_MCP5_cal_raw = CAL_raw(CAL_frame, 96:98);

    % Radial Styloid
    R_WRS_cal_raw = CAL_raw(CAL_frame, 21:23);
    L_WRS_cal_raw = CAL_raw(CAL_frame, 84:86);

    % Ulnar Styloid
    R_WUS_cal_raw = CAL_raw(CAL_frame, 24:26);
    L_WUS_cal_raw = CAL_raw(CAL_frame, 81:83); 

    % Hand Cluster - right = 1, middle/point = 2, left = 3
    R_HND_1_cal_raw = CAL_raw(CAL_frame, 12:14);
    R_HND_2_cal_raw = CAL_raw(CAL_frame, 15:17);
    R_HND_3_cal_raw = CAL_raw(CAL_frame, 18:20);

    L_HND_1_cal_raw = CAL_raw(CAL_frame, 87:89);
    L_HND_2_cal_raw = CAL_raw(CAL_frame, 90:92);
    L_HND_3_cal_raw = CAL_raw(CAL_frame, 93:95);

    % Forearm Cluster - superior = 3, middle/point = 2, inferior = 1
    R_FA_1_cal_raw = CAL_raw(CAL_frame, 27:29);
    R_FA_2_cal_raw = CAL_raw(CAL_frame, 30:32);
    R_FA_3_cal_raw = CAL_raw(CAL_frame, 33:35);

    L_FA_1_cal_raw = CAL_raw(CAL_frame, 78:80);
    L_FA_2_cal_raw = CAL_raw(CAL_frame, 75:77);
    L_FA_3_cal_raw = CAL_raw(CAL_frame, 72:74);

    % Lateral Epicondyle
    R_LEC_cal_raw = CAL_raw(CAL_frame, 36:38);
    L_LEC_cal_raw = CAL_raw(CAL_frame, 66:68);

    % Medial Epicondyle
    R_MEC_cal_raw = CAL_raw(CAL_frame, 39:41);
    L_MEC_cal_raw = CAL_raw(CAL_frame, 69:71);

    % Upper Arm Cluster - superior = 3, middle/point = 2, inferior = 1
    R_UA_1_cal_raw = CAL_raw(CAL_frame, 42:44);
    R_UA_2_cal_raw = CAL_raw(CAL_frame, 45:47);
    R_UA_3_cal_raw = CAL_raw(CAL_frame, 48:50);
    
    L_UA_1_cal_raw = CAL_raw(CAL_frame, 63:65);
    L_UA_2_cal_raw = CAL_raw(CAL_frame, 60:62);
    L_UA_3_cal_raw = CAL_raw(CAL_frame, 57:59);

    % Acromion 
    R_AP_cal_raw = CAL_raw(CAL_frame, 51:53);
    L_AP_cal_raw = CAL_raw(CAL_frame, 54:56);

    % Supra Sternal Notch
    SS_cal_raw = CAL_raw(CAL_frame, 141:143);

    % Xiphoid Process
    XP_cal_raw = CAL_raw(CAL_frame, 144:146);
      
    % C7
    C7_cal_raw = CAL_raw(CAL_frame, 105:107);

    % T8
    T8_cal_raw = CAL_raw(CAL_frame, 108:110);

    % Thoracic Cluster - top right = 1, bottome right = 2, bottom left = 3, top left = 4
    THOR_1_cal_raw = CAL_raw(CAL_frame, 111:113);
    THOR_2_cal_raw = CAL_raw(CAL_frame, 114:116);
    THOR_3_cal_raw = CAL_raw(CAL_frame, 117:119);
    THOR_4_cal_raw = CAL_raw(CAL_frame, 120:122);

    % Lumbar Cluster - right = 1, middle/point = 2, left = 3
    LMB_1_cal_raw = CAL_raw(CAL_frame, 123:125);
    LMB_2_cal_raw = CAL_raw(CAL_frame, 126:128);
    LMB_3_cal_raw = CAL_raw(CAL_frame, 129:131);

    % L5
    L5_cal_raw = CAL_raw(CAL_frame, 135:137);

    % Posterior Superior Iliac Spine
    R_PSIS_cal_raw = CAL_raw(CAL_frame, 132:134);
    L_PSIS_cal_raw = CAL_raw(CAL_frame, 138:140);

    % Anterior Superior Iliac Spine
    R_ASIS_cal_raw = CAL_raw(CAL_frame, 147:149);
    L_ASIS_cal_raw = CAL_raw(CAL_frame, 150:152);

% ROTATE TO ISB AXES
    % Define axes rotation matrix
        % Vicon/lab conventions: +X = left, +Y = forwards, +Z = up
        % In lab set up = wand at top right corner
        % ISB conventions: +X = foward, +Y = up, +Z = right
    ISB_X = [0 1 0]; % +Y VICON = +X ISB
    ISB_Y = [0 0 1]; % +Z VICON = +Y ISB
    ISB_Z = [-1 0 0]; % -X VICON = +Z ISB (negative so positive goes L to R)
    ISB = [ISB_X; ISB_Y; ISB_Z];

    % Rotate CAL by ISB
        % Multiply ISB to columns of each landmark
    
    % 3rd Distal Phalanx
    R_DP3_cal_isb = (ISB*R_DP3_cal_raw')';
    L_DP3_cal_isb = (ISB*L_DP3_cal_raw')';

    % 2nd Metacarpal
    R_MCP2_cal_isb = (ISB*R_MCP2_cal_raw')';
    L_MCP2_cal_isb = (ISB*L_MCP2_cal_raw')';

    % 5th Metacarpal
    R_MCP5_cal_isb = (ISB*R_MCP5_cal_raw')';
    L_MCP5_cal_isb = (ISB*L_MCP5_cal_raw')';

    % Radial Styloid
    R_WRS_cal_isb = (ISB*R_WRS_cal_raw')';
    L_WRS_cal_isb = (ISB*L_WRS_cal_raw')';

    % Ulnar Styloid
    R_WUS_cal_isb = (ISB*R_WUS_cal_raw')';
    L_WUS_cal_isb = (ISB*L_WUS_cal_raw')'; 

    % Hand Cluster
    R_HND_1_cal_isb = (ISB*R_HND_1_cal_raw')';
    R_HND_2_cal_isb = (ISB*R_HND_2_cal_raw')';
    R_HND_3_cal_isb = (ISB*R_HND_3_cal_raw')';

    L_HND_1_cal_isb = (ISB*L_HND_1_cal_raw')';
    L_HND_2_cal_isb = (ISB*L_HND_2_cal_raw')';
    L_HND_3_cal_isb = (ISB*L_HND_3_cal_raw')';

    % Forearm Cluster
    R_FA_1_cal_isb = (ISB*R_FA_1_cal_raw')';
    R_FA_2_cal_isb = (ISB*R_FA_2_cal_raw')';
    R_FA_3_cal_isb = (ISB*R_FA_3_cal_raw')';

    L_FA_1_cal_isb = (ISB*L_FA_1_cal_raw')';
    L_FA_2_cal_isb = (ISB*L_FA_2_cal_raw')';
    L_FA_3_cal_isb = (ISB*L_FA_3_cal_raw')';

    % Lateral Epicondyle
    R_LEC_cal_isb = (ISB*R_LEC_cal_raw')';
    L_LEC_cal_isb = (ISB*L_LEC_cal_raw')';

    % Medial Epicondyle
    R_MEC_cal_isb = (ISB*R_MEC_cal_raw')';
    L_MEC_cal_isb = (ISB*L_MEC_cal_raw')';
    
    % Upper Arm Cluster
    R_UA_1_cal_isb = (ISB*R_UA_1_cal_raw')';
    R_UA_2_cal_isb = (ISB*R_UA_2_cal_raw')';
    R_UA_3_cal_isb = (ISB*R_UA_3_cal_raw')';
    
    L_UA_1_cal_isb = (ISB*L_UA_1_cal_raw')';
    L_UA_2_cal_isb = (ISB*L_UA_2_cal_raw')';
    L_UA_3_cal_isb = (ISB*L_UA_3_cal_raw')';

    % Acromion 
    R_AP_cal_isb = (ISB*R_AP_cal_raw')';
    L_AP_cal_isb = (ISB*L_AP_cal_raw')';

    % Supra Sternal Notch
    SS_cal_isb = (ISB*SS_cal_raw')';

    % Xiphoid Process
    XP_cal_isb = (ISB*XP_cal_raw')';
      
    % C7
    C7_cal_isb = (ISB*C7_cal_raw')';

    % T8
    T8_cal_isb = (ISB*T8_cal_raw')';

    % Thoracic Cluster
    THOR_1_cal_isb = (ISB*THOR_1_cal_raw')';
    THOR_2_cal_isb = (ISB*THOR_2_cal_raw')';
    THOR_3_cal_isb = (ISB*THOR_3_cal_raw')';
    THOR_4_cal_isb = (ISB*THOR_4_cal_raw')';

    % Lumbar Cluster
    LMB_1_cal_isb = (ISB*LMB_1_cal_raw')';
    LMB_2_cal_isb = (ISB*LMB_2_cal_raw')';
    LMB_3_cal_isb = (ISB*LMB_3_cal_raw')';

    % L5
    L5_cal_isb = (ISB*L5_cal_raw')';

    % Posterior Superior Iliac Spine
    R_PSIS_cal_isb = (ISB*R_PSIS_cal_raw')';
    L_PSIS_cal_isb = (ISB*L_PSIS_cal_raw')';

    % Anterior Superior Iliac Spine
    R_ASIS_cal_isb = (ISB*R_ASIS_cal_raw')';
    L_ASIS_cal_isb = (ISB*L_ASIS_cal_raw')';

% % VISUAL CHECK OF ALL CAL MARKER LOCATIONS
% figure('Name', 'CAL_isb', 'NumberTitle', 'off')
%     % Hand
%         scatter3(R_DP3_cal_isb(1,1), R_DP3_cal_isb(1,2), R_DP3_cal_isb(1,3),...
%         'r','filled');
%         hold on
%         scatter3(L_DP3_cal_isb(1,1), L_DP3_cal_isb(1,2), L_DP3_cal_isb(1,3),...
%         'r','filled'); 
% 
%     text(R_DP3_cal_isb(1,1),R_DP3_cal_isb(1,2),R_DP3_cal_isb(1,3),'   R_DP3');
%     text(L_DP3_cal_isb(1,1),L_DP3_cal_isb(1,2),L_DP3_cal_isb(1,3),'   L_DP3');
% 
%     scatter3(R_WUS_cal_isb(1,1), R_WUS_cal_isb(1,2), R_WUS_cal_isb(1,3),...
%         'r','filled'); 
%     scatter3(L_WUS_cal_isb(1,1), L_WUS_cal_isb(1,2), L_WUS_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_WRS_cal_isb(1,1), R_WRS_cal_isb(1,2), R_WRS_cal_isb(1,3),...
%         'r','filled'); 
%     scatter3(L_WRS_cal_isb(1,1), L_WRS_cal_isb(1,2), L_WRS_cal_isb(1,3),...
%         'r','filled'); 
% 
%     scatter3(R_MCP2_cal_isb(1,1),R_MCP2_cal_isb(1,2),R_MCP2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_MCP2_cal_isb(1,1),L_MCP2_cal_isb(1,2),L_MCP2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_MCP5_cal_isb(1,1),R_MCP5_cal_isb(1,2),R_MCP5_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_MCP5_cal_isb(1,1),L_MCP5_cal_isb(1,2),L_MCP5_cal_isb(1,3),...
%         'r','filled');
% 
%     text(R_WRS_cal_isb(1,1),R_WRS_cal_isb(1,2),R_WRS_cal_isb(1,3),'   R_WRS');
%     text(L_WRS_cal_isb(1,1),L_WRS_cal_isb(1,2),L_WRS_cal_isb(1,3),'   L_WRS');
%     text(R_WUS_cal_isb(1,1),R_WUS_cal_isb(1,2),R_WUS_cal_isb(1,3),'   R_WUS');
%     text(L_WUS_cal_isb(1,1),L_WUS_cal_isb(1,2),L_WUS_cal_isb(1,3),'   L_WUS');
% 
%     text(R_MCP2_cal_isb(1,1),R_MCP2_cal_isb(1,2),R_MCP2_cal_isb(1,3),...
%         '   R_MCP2');
%     text(L_MCP2_cal_isb(1,1),L_MCP2_cal_isb(1,2),L_MCP2_cal_isb(1,3),...
%         '   L_MCP2');
%     text(R_MCP5_cal_isb(1,1),R_MCP5_cal_isb(1,2),R_MCP5_cal_isb(1,3),...
%         '   R_MCP5');
%     text(L_MCP5_cal_isb(1,1),L_MCP5_cal_isb(1,2),L_MCP5_cal_isb(1,3),...
%         '   L_MCP5');
% 
%     line([R_WUS_cal_isb(1,1),R_WRS_cal_isb(1,1)],...
%         [R_WUS_cal_isb(1,2),R_WRS_cal_isb(1,2)],...
%         [R_WUS_cal_isb(1,3),R_WRS_cal_isb(1,3)],'Color','black')
%     line([R_WRS_cal_isb(1,1),R_MCP2_cal_isb(1,1)],...
%         [R_WRS_cal_isb(1,2),R_MCP2_cal_isb(1,2)],...
%         [R_WRS_cal_isb(1,3),R_MCP2_cal_isb(1,3)],'Color','black')
%     line([R_WUS_cal_isb(1,1),R_MCP5_cal_isb(1,1)],...
%         [R_WUS_cal_isb(1,2),R_MCP5_cal_isb(1,2)],...
%         [R_WUS_cal_isb(1,3),R_MCP5_cal_isb(1,3)],'Color','black')
%     line([R_MCP2_cal_isb(1,1),R_MCP5_cal_isb(1,1)],...
%         [R_MCP2_cal_isb(1,2),R_MCP5_cal_isb(1,2)],...
%         [R_MCP2_cal_isb(1,3),R_MCP5_cal_isb(1,3)],'Color','black')
%     line([R_MCP2_cal_isb(1,1),R_DP3_cal_isb(1,1)],...
%         [R_MCP2_cal_isb(1,2),R_DP3_cal_isb(1,2)],...
%         [R_MCP2_cal_isb(1,3),R_DP3_cal_isb(1,3)],'Color','black')
%     line([R_MCP5_cal_isb(1,1),R_DP3_cal_isb(1,1)],...
%         [R_MCP5_cal_isb(1,2),R_DP3_cal_isb(1,2)],...
%         [R_MCP5_cal_isb(1,3),R_DP3_cal_isb(1,3)],'Color','black')
%     line([R_WRS_cal_isb(1,1),R_LEC_cal_isb(1,1)],...
%         [R_WRS_cal_isb(1,2),R_LEC_cal_isb(1,2)],...
%         [R_WRS_cal_isb(1,3),R_LEC_cal_isb(1,3)],'Color','black')
% 
%     line([L_WUS_cal_isb(1,1),L_WRS_cal_isb(1,1)],...
%         [L_WUS_cal_isb(1,2),L_WRS_cal_isb(1,2)],...
%         [L_WUS_cal_isb(1,3),L_WRS_cal_isb(1,3)],'Color','black')
%     line([L_WRS_cal_isb(1,1),L_MCP2_cal_isb(1,1)],...
%         [L_WRS_cal_isb(1,2),L_MCP2_cal_isb(1,2)],...
%         [L_WRS_cal_isb(1,3),L_MCP2_cal_isb(1,3)],'Color','black')
%     line([L_WUS_cal_isb(1,1),L_MCP5_cal_isb(1,1)],...
%         [L_WUS_cal_isb(1,2),L_MCP5_cal_isb(1,2)],...
%         [L_WUS_cal_isb(1,3),L_MCP5_cal_isb(1,3)],'Color','black')
%     line([L_MCP2_cal_isb(1,1),L_MCP5_cal_isb(1,1)],...
%         [L_MCP2_cal_isb(1,2),L_MCP5_cal_isb(1,2)],...
%         [L_MCP2_cal_isb(1,3),L_MCP5_cal_isb(1,3)],'Color','black')
%     line([L_MCP2_cal_isb(1,1),L_DP3_cal_isb(1,1)],...
%         [L_MCP2_cal_isb(1,2),L_DP3_cal_isb(1,2)],...
%         [L_MCP2_cal_isb(1,3),L_DP3_cal_isb(1,3)],'Color','black')
%     line([L_MCP5_cal_isb(1,1),L_DP3_cal_isb(1,1)],...
%         [L_MCP5_cal_isb(1,2),L_DP3_cal_isb(1,2)],...
%         [L_MCP5_cal_isb(1,3),L_DP3_cal_isb(1,3)],'Color','black')
%     line([L_WRS_cal_isb(1,1),L_LEC_cal_isb(1,1)],...
%         [L_WRS_cal_isb(1,2),L_LEC_cal_isb(1,2)],...
%         [L_WRS_cal_isb(1,3),L_LEC_cal_isb(1,3)],'Color','black')
% 
%     scatter3(R_HND_1_cal_isb(1,1),R_HND_1_cal_isb(1,2),R_HND_1_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_HND_2_cal_isb(1,1),R_HND_2_cal_isb(1,2),R_HND_2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_HND_3_cal_isb(1,1),R_HND_3_cal_isb(1,2),R_HND_3_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_HND_1_cal_isb(1,1),L_HND_1_cal_isb(1,2),L_HND_1_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_HND_2_cal_isb(1,1),L_HND_2_cal_isb(1,2),L_HND_2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_HND_3_cal_isb(1,1),L_HND_3_cal_isb(1,2),L_HND_3_cal_isb(1,3),...
%         'r','filled');
% 
%     % Forearm
%     scatter3(R_LEC_cal_isb(1,1),R_LEC_cal_isb(1,2),R_LEC_cal_isb(1,3),'r',...
%         'filled');
%     scatter3(L_LEC_cal_isb(1,1),L_LEC_cal_isb(1,2),L_LEC_cal_isb(1,3),'r',...
%         'filled');
%     scatter3(R_MEC_cal_isb(1,1),R_MEC_cal_isb(1,2),R_MEC_cal_isb(1,3),'r',...
%         'filled');
%     scatter3(L_MEC_cal_isb(1,1),L_MEC_cal_isb(1,2),L_MEC_cal_isb(1,3),'r',...
%         'filled');
% 
%     text(R_LEC_cal_isb(1,1),R_LEC_cal_isb(1,2),R_LEC_cal_isb(1,3),'   R_LEC');
%     text(L_LEC_cal_isb(1,1),L_LEC_cal_isb(1,2),L_LEC_cal_isb(1,3),'   L_LEC');
%     text(R_MEC_cal_isb(1,1),R_MEC_cal_isb(1,2),R_MEC_cal_isb(1,3),'   R_MEC');
%     text(L_MEC_cal_isb(1,1),L_MEC_cal_isb(1,2),L_MEC_cal_isb(1,3),'   L_MEC');
% 
%     line([R_LEC_cal_isb(1,1),R_MEC_cal_isb(1,1)],...
%         [R_LEC_cal_isb(1,2),R_MEC_cal_isb(1,2)],...
%         [R_LEC_cal_isb(1,3),R_MEC_cal_isb(1,3)],'Color','black');
%     line([L_LEC_cal_isb(1,1),L_MEC_cal_isb(1,1)],...
%         [L_LEC_cal_isb(1,2),L_MEC_cal_isb(1,2)],...
%         [L_LEC_cal_isb(1,3),L_MEC_cal_isb(1,3)],'Color','black');
%     line([R_LEC_cal_isb(1,1),R_AP_cal_isb(1,1)],...
%         [R_LEC_cal_isb(1,2),R_AP_cal_isb(1,2)],...
%         [R_LEC_cal_isb(1,3),R_AP_cal_isb(1,3)],'Color','black');
%     line([L_LEC_cal_isb(1,1),L_AP_cal_isb(1,1)],...
%         [L_LEC_cal_isb(1,2),L_AP_cal_isb(1,2)],...
%         [L_LEC_cal_isb(1,3),L_AP_cal_isb(1,3)],'Color','black');
% 
%     scatter3(R_FA_1_cal_isb(1,1),R_FA_1_cal_isb(1,2),R_FA_1_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_FA_2_cal_isb(1,1),R_FA_2_cal_isb(1,2),R_FA_2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_FA_3_cal_isb(1,1),R_FA_3_cal_isb(1,2),R_FA_3_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_FA_1_cal_isb(1,1),L_FA_1_cal_isb(1,2),L_FA_1_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_FA_2_cal_isb(1,1),L_FA_2_cal_isb(1,2),L_FA_2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_FA_3_cal_isb(1,1),L_FA_3_cal_isb(1,2),L_FA_3_cal_isb(1,3),...
%         'r','filled');
% 
%     % Arm
%     scatter3(R_AP_cal_isb(1,1),R_AP_cal_isb(1,2),R_AP_cal_isb(1,3),'r',...
%         'filled');
%     scatter3(L_AP_cal_isb(1,1),L_AP_cal_isb(1,2),L_AP_cal_isb(1,3),'r',...
%         'filled');
% 
%     text(R_AP_cal_isb(1,1),R_AP_cal_isb(1,2),R_AP_cal_isb(1,3),'   R_AP');
%     text(L_AP_cal_isb(1,1),L_AP_cal_isb(1,2),L_AP_cal_isb(1,3),'   L_AP');
% 
%     scatter3(R_UA_1_cal_isb(1,1),R_UA_1_cal_isb(1,2),R_UA_1_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_UA_2_cal_isb(1,1),R_UA_2_cal_isb(1,2),R_UA_2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(R_UA_3_cal_isb(1,1),R_UA_3_cal_isb(1,2),R_UA_3_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_UA_1_cal_isb(1,1),L_UA_1_cal_isb(1,2),L_UA_1_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_UA_2_cal_isb(1,1),L_UA_2_cal_isb(1,2),L_UA_2_cal_isb(1,3),...
%         'r','filled');
%     scatter3(L_UA_3_cal_isb(1,1),L_UA_3_cal_isb(1,2),L_UA_3_cal_isb(1,3),...
%         'r','filled');
% 
%     % Torso
%     scatter3(C7_cal_isb(1,1),C7_cal_isb(1,2),C7_cal_isb(1,3),'k','filled');
% 
%     text(C7_cal_isb(1,1),C7_cal_isb(1,2),C7_cal_isb(1,3),'   C7');
% 
%     line([R_AP_cal_isb(1,1),C7_cal_isb(1,1)],...
%         [R_AP_cal_isb(1,2),C7_cal_isb(1,2)],...
%         [R_AP_cal_isb(1,3),C7_cal_isb(1,3)],'Color','black');
%     line([L_AP_cal_isb(1,1),C7_cal_isb(1,1)],...
%         [L_AP_cal_isb(1,2),C7_cal_isb(1,2)],...
%         [L_AP_cal_isb(1,3),C7_cal_isb(1,3)],'Color','black');
% 
%     scatter3(SS_cal_isb(1,1),SS_cal_isb(1,2),SS_cal_isb(1,3),'k','filled');
% 
%     text(SS_cal_isb(1,1),SS_cal_isb(1,2),SS_cal_isb(1,3),'   SS');
% 
%     line([C7_cal_isb(1,1),SS_cal_isb(1,1)],...
%         [C7_cal_isb(1,2),SS_cal_isb(1,2)],...
%         [C7_cal_isb(1,3),SS_cal_isb(1,3)],'Color','black');
% 
%     scatter3(XP_cal_isb(1,1),XP_cal_isb(1,2),XP_cal_isb(1,3),'k','filled');
% 
%     text(XP_cal_isb(1,1),XP_cal_isb(1,2),XP_cal_isb(1,3),'   XP');
% 
%     line([SS_cal_isb(1,1),XP_cal_isb(1,1)],...
%         [SS_cal_isb(1,2),XP_cal_isb(1,2)],...
%         [SS_cal_isb(1,3),XP_cal_isb(1,3)],'Color','black');
% 
%     scatter3(T8_cal_isb(1,1),T8_cal_isb(1,2),T8_cal_isb(1,3),'k','filled');
% 
%     text(T8_cal_isb(1,1),T8_cal_isb(1,2),T8_cal_isb(1,3),'   T8');
% 
%     line([T8_cal_isb(1,1),C7_cal_isb(1,1)],...
%         [T8_cal_isb(1,2),C7_cal_isb(1,2)],...
%         [T8_cal_isb(1,3),C7_cal_isb(1,3)],'Color','black');
% 
%     scatter3(THOR_1_cal_isb(1,1),THOR_1_cal_isb(1,2),THOR_1_cal_isb(1,3),...
%         'k','filled');
%     scatter3(THOR_2_cal_isb(1,1),THOR_2_cal_isb(1,2),THOR_2_cal_isb(1,3),...
%         'k','filled');
%     scatter3(THOR_3_cal_isb(1,1),THOR_3_cal_isb(1,2),THOR_3_cal_isb(1,3),...
%         'k','filled');
%     scatter3(THOR_4_cal_isb(1,1),THOR_4_cal_isb(1,2),THOR_4_cal_isb(1,3),...
%         'k','filled');
% 
%     scatter3(L5_cal_isb(1,1),L5_cal_isb(1,2),L5_cal_isb(1,3),'k','filled')
% 
%     text(L5_cal_isb(1,1),L5_cal_isb(1,2),L5_cal_isb(1,3),'   L5');
% 
%     line([L5_cal_isb(1,1),T8_cal_isb(1,1)],...
%         [L5_cal_isb(1,2),T8_cal_isb(1,2)],...
%         [L5_cal_isb(1,3),T8_cal_isb(1,3)],'Color','black');
% 
%     scatter3(LMB_1_cal_isb(1,1),LMB_1_cal_isb(1,2),LMB_1_cal_isb(1,3),...
%         'k','filled');
%     scatter3(LMB_2_cal_isb(1,1),LMB_2_cal_isb(1,2),LMB_2_cal_isb(1,3),...
%         'k','filled');
%     scatter3(LMB_3_cal_isb(1,1),LMB_3_cal_isb(1,2),LMB_3_cal_isb(1,3),...
%         'k','filled');
% 
%     % Pelvis
%     scatter3(R_ASIS_cal_isb(1,1),R_ASIS_cal_isb(1,2),R_ASIS_cal_isb(1,3),...
%         'k','filled');
%     scatter3(L_ASIS_cal_isb(1,1),L_ASIS_cal_isb(1,2),L_ASIS_cal_isb(1,3),...
%         'k','filled');
%     scatter3(R_PSIS_cal_isb(1,1),R_PSIS_cal_isb(1,2),R_PSIS_cal_isb(1,3),...
%         'k','filled');
%     scatter3(L_PSIS_cal_isb(1,1),L_PSIS_cal_isb(1,2),L_PSIS_cal_isb(1,3),...
%         'k','filled');
% 
%     text(R_ASIS_cal_isb(1,1),R_ASIS_cal_isb(1,2),R_ASIS_cal_isb(1,3),...
%         '   R_ASIS');
%     text(L_ASIS_cal_isb(1,1),L_ASIS_cal_isb(1,2),L_ASIS_cal_isb(1,3),...
%         '   L_ASIS');
%     text(R_PSIS_cal_isb(1,1),R_PSIS_cal_isb(1,2),R_PSIS_cal_isb(1,3),...
%         '   R_PSIS');
%     text(L_PSIS_cal_isb(1,1),L_PSIS_cal_isb(1,2),L_PSIS_cal_isb(1,3),...
%         '   L_PSIS');
% 
%     line([R_ASIS_cal_isb(1,1),L_ASIS_cal_isb(1,1)],...
%         [R_ASIS_cal_isb(1,2),L_ASIS_cal_isb(1,2)],...
%         [R_ASIS_cal_isb(1,3),L_ASIS_cal_isb(1,3)],'Color','black');
%     line([R_PSIS_cal_isb(1,1),L_PSIS_cal_isb(1,1)],...
%         [R_ASIS_cal_isb(1,2),L_PSIS_cal_isb(1,2)],...
%         [R_PSIS_cal_isb(1,3),L_PSIS_cal_isb(1,3)],'Color','black');
%     line([R_ASIS_cal_isb(1,1),R_PSIS_cal_isb(1,1)],...
%         [R_ASIS_cal_isb(1,2),R_PSIS_cal_isb(1,2)],...
%         [R_ASIS_cal_isb(1,3),R_PSIS_cal_isb(1,3)],'Color','black');
%     line([L_ASIS_cal_isb(1,1),L_PSIS_cal_isb(1,1)],...
%         [L_ASIS_cal_isb(1,2),L_PSIS_cal_isb(1,2)],...
%         [L_ASIS_cal_isb(1,3),L_PSIS_cal_isb(1,3)],'Color','black');
% 
%     line([R_PSIS_cal_isb(1,1),L5_cal_isb(1,1)],...
%         [R_PSIS_cal_isb(1,2),L5_cal_isb(1,2)],...
%         [R_PSIS_cal_isb(1,3),L5_cal_isb(1,3)],'Color','black');
%     line([L_PSIS_cal_isb(1,1),L5_cal_isb(1,1)],...
%         [L_PSIS_cal_isb(1,2),L5_cal_isb(1,2)],...
%         [L_PSIS_cal_isb(1,3),L5_cal_isb(1,3)],'Color','black');
% 
%     xlabel ('x-axis');
%     ylabel ('y-axis');
%     zlabel ('z-axis');

%% -------------------------- TASK FILE -------------------------------- %%
% SET UP TASK FILE DIRECTORIES
   TASK_dir = "C:\Users\kathr\Documents\Waterloo MSc\Thesis\Data\KINEMATICS\" ...
        + subject_num + "_KIN\";

   TASK_file_list = dir (TASK_dir + subject_num + "*.csv");

% READ IN RAW TASK FILES
for TASK_file_list_index = 1:length(TASK_file_list)

        TASK_file = TASK_file_list(TASK_file_list_index).name;
    
        disp ("start file:" + TASK_file)
 
    TASK_raw = readmatrix(TASK_dir + TASK_file); 

    % Set up file header mapping for task
    % Define marker corresponding to column number in file

    % 3rd Distal Phalanx
    R_DP3_task_raw = TASK_raw(6:end, 3:5);
    L_DP3_task_raw = TASK_raw(6:end, 102:104);

    % 2nd Metacarpal
    R_MCP2_task_raw = TASK_raw(6:end, 6:8);
    L_MCP2_task_raw = TASK_raw(6:end, 99:101);

    % 5th Metacarpal
    R_MCP5_task_raw = TASK_raw(6:end, 9:11);
    L_MCP5_task_raw = TASK_raw(6:end, 96:98);

    % Radial Styloid
    R_WRS_task_raw = TASK_raw(6:end, 21:23);
    L_WRS_task_raw = TASK_raw(6:end, 84:86);

    % Ulnar Styloid
    R_WUS_task_raw = TASK_raw(6:end, 24:26);
    L_WUS_task_raw = TASK_raw(6:end, 81:83); 

    % Hand Cluster - right = 1, middle/point = 2, left = 3
    R_HND_1_task_raw = TASK_raw(6:end, 12:14);
    R_HND_2_task_raw = TASK_raw(6:end, 15:17);
    R_HND_3_task_raw = TASK_raw(6:end, 18:20);

    L_HND_1_task_raw = TASK_raw(6:end, 87:89);
    L_HND_2_task_raw = TASK_raw(6:end, 90:92);
    L_HND_3_task_raw = TASK_raw(6:end, 93:95);

    % Forearm Cluster - superior = 3, middle/point = 2, inferior = 1
    R_FA_1_task_raw = TASK_raw(6:end, 27:29);
    R_FA_2_task_raw = TASK_raw(6:end, 30:32);
    R_FA_3_task_raw = TASK_raw(6:end, 33:35);

    L_FA_1_task_raw = TASK_raw(6:end, 78:80);
    L_FA_2_task_raw = TASK_raw(6:end, 75:77);
    L_FA_3_task_raw = TASK_raw(6:end, 72:74);

    % Lateral Epicondyle
    R_LEC_task_raw = TASK_raw(6:end, 36:38);
    L_LEC_task_raw = TASK_raw(6:end, 66:68);

    % Medial Epicondyle
    R_MEC_task_raw = TASK_raw(6:end, 39:41);
    L_MEC_task_raw = TASK_raw(6:end, 69:71);

    % Upper Arm Cluster - superior = 3, middle/point = 2, inferior = 1
    R_UA_1_task_raw = TASK_raw(6:end, 42:44);
    R_UA_2_task_raw = TASK_raw(6:end, 45:47);
    R_UA_3_task_raw = TASK_raw(6:end, 48:50);
    
    L_UA_1_task_raw = TASK_raw(6:end, 63:65);
    L_UA_2_task_raw = TASK_raw(6:end, 60:62);
    L_UA_3_task_raw = TASK_raw(6:end, 57:59);

    % Acromion 
    R_AP_task_raw = TASK_raw(6:end, 51:53);
    L_AP_task_raw = TASK_raw(6:end, 54:56);
    
    % C7
    C7_task_raw = TASK_raw(6:end, 105:107);

    % T8
    T8_task_raw = TASK_raw(6:end, 108:110);

    % Thoracic Cluster - top right = 1, bottome right = 2, bottom left = 3, top left = 4
    THOR_1_task_raw = TASK_raw(6:end, 111:113);
    THOR_2_task_raw = TASK_raw(6:end, 114:116);
    THOR_3_task_raw = TASK_raw(6:end, 117:119);
    THOR_4_task_raw = TASK_raw(6:end, 120:122);

    % Lumbar Cluster - right = 1, middle/point = 2, left = 3
    LMB_1_task_raw = TASK_raw(6:end, 123:125);
    LMB_2_task_raw = TASK_raw(6:end, 126:128);
    LMB_3_task_raw = TASK_raw(6:end, 129:131);

    % L5
    L5_task_raw = TASK_raw(6:end, 135:137);

    % Posterior Superior Iliac Spine
    R_PSIS_task_raw = TASK_raw(6:end, 132:134);
    L_PSIS_task_raw = TASK_raw(6:end, 138:140);

% ROTATE TASK TO ISB AXES
    % Multiply ISB to columns of each landmark
    
    % 3rd Distal Phalanx
    R_DP3_task_isb = (ISB*R_DP3_task_raw')';
    L_DP3_task_isb = (ISB*L_DP3_task_raw')';

    % 2nd Metacarpal
    R_MCP2_task_isb = (ISB*R_MCP2_task_raw')';
    L_MCP2_task_isb = (ISB*L_MCP2_task_raw')';

    % 5th Metacarpal
    R_MCP5_task_isb = (ISB*R_MCP5_task_raw')';
    L_MCP5_task_isb = (ISB*L_MCP5_task_raw')';

    % Radial Styloid
    R_WRS_task_isb = (ISB*R_WRS_task_raw')';
    L_WRS_task_isb = (ISB*L_WRS_task_raw')';

    % Ulnar Styloid
    R_WUS_task_isb = (ISB*R_WUS_task_raw')';
    L_WUS_task_isb = (ISB*L_WUS_task_raw')'; 

    % Hand Cluster
    R_HND_1_task_isb = (ISB*R_HND_1_task_raw')';
    R_HND_2_task_isb = (ISB*R_HND_2_task_raw')';
    R_HND_3_task_isb = (ISB*R_HND_3_task_raw')';

    L_HND_1_task_isb = (ISB*L_HND_1_task_raw')';
    L_HND_2_task_isb = (ISB*L_HND_2_task_raw')';
    L_HND_3_task_isb = (ISB*L_HND_3_task_raw')';

    % Forearm Cluster
    R_FA_1_task_isb = (ISB*R_FA_1_task_raw')';
    R_FA_2_task_isb = (ISB*R_FA_2_task_raw')';
    R_FA_3_task_isb = (ISB*R_FA_3_task_raw')';

    L_FA_1_task_isb = (ISB*L_FA_1_task_raw')';
    L_FA_2_task_isb = (ISB*L_FA_2_task_raw')';
    L_FA_3_task_isb = (ISB*L_FA_3_task_raw')';

    % Lateral Epicondyle
    R_LEC_task_isb = (ISB*R_LEC_task_raw')';
    L_LEC_task_isb = (ISB*L_LEC_task_raw')';

    % Medial Epicondyle
    R_MEC_task_isb = (ISB*R_MEC_task_raw')';
    L_MEC_task_isb = (ISB*L_MEC_task_raw')';
    
    % Upper Arm Cluster
    R_UA_1_task_isb = (ISB*R_UA_1_task_raw')';
    R_UA_2_task_isb = (ISB*R_UA_2_task_raw')';
    R_UA_3_task_isb = (ISB*R_UA_3_task_raw')';
    
    L_UA_1_task_isb = (ISB*L_UA_1_task_raw')';
    L_UA_2_task_isb = (ISB*L_UA_2_task_raw')';
    L_UA_3_task_isb = (ISB*L_UA_3_task_raw')';

    % Acromion 
    R_AP_task_isb = (ISB*R_AP_task_raw')';
    L_AP_task_isb = (ISB*L_AP_task_raw')';
      
    % C7
    C7_task_isb = (ISB*C7_task_raw')';

    % T8
    T8_task_isb = (ISB*T8_task_raw')';

    % Thoracic Cluster
    THOR_1_task_isb = (ISB*THOR_1_task_raw')';
    THOR_2_task_isb = (ISB*THOR_2_task_raw')';
    THOR_3_task_isb = (ISB*THOR_3_task_raw')';
    THOR_4_task_isb = (ISB*THOR_4_task_raw')';

    % Lumbar Cluster
    LMB_1_task_isb = (ISB*LMB_1_task_raw')';
    LMB_2_task_isb = (ISB*LMB_2_task_raw')';
    LMB_3_task_isb = (ISB*LMB_3_task_raw')';

    % L5
    L5_task_isb = (ISB*L5_task_raw')';

    % Posterior Superior Iliac Spine
    R_PSIS_task_isb = (ISB*R_PSIS_task_raw')';
    L_PSIS_task_isb = (ISB*L_PSIS_task_raw')';

% VISUAL CHECK OF ALL TASK MARKER LOCATIONS
figure('Name', TASK_file, 'NumberTitle', 'off')
    % Hand
    scatter3(R_DP3_task_isb(1,1), R_DP3_task_isb(1,2), R_DP3_task_isb(1,3),...
   'r','filled');
    hold on
    scatter3(L_DP3_task_isb(1,1), L_DP3_task_isb(1,2), L_DP3_task_isb(1,3),...
    'r','filled'); 

    text(R_DP3_task_isb(1,1),R_DP3_task_isb(1,2),R_DP3_task_isb(1,3),'   R_DP3');
    text(L_DP3_task_isb(1,1),L_DP3_task_isb(1,2),L_DP3_task_isb(1,3),'   L_DP3');

    scatter3(R_WUS_task_isb(1,1), R_WUS_task_isb(1,2), R_WUS_task_isb(1,3),...
        'r','filled'); 
    scatter3(L_WUS_task_isb(1,1), L_WUS_task_isb(1,2), L_WUS_task_isb(1,3),...
        'r','filled');
    scatter3(R_WRS_task_isb(1,1), R_WRS_task_isb(1,2), R_WRS_task_isb(1,3),...
        'r','filled'); 
    scatter3(L_WRS_task_isb(1,1), L_WRS_task_isb(1,2), L_WRS_task_isb(1,3),...
        'r','filled'); 

    scatter3(R_MCP2_task_isb(1,1),R_MCP2_task_isb(1,2),R_MCP2_task_isb(1,3),...
        'r','filled');
    scatter3(L_MCP2_task_isb(1,1),L_MCP2_task_isb(1,2),L_MCP2_task_isb(1,3),...
        'r','filled');
    scatter3(R_MCP5_task_isb(1,1),R_MCP5_task_isb(1,2),R_MCP5_task_isb(1,3),...
        'r','filled');
    scatter3(L_MCP5_task_isb(1,1),L_MCP5_task_isb(1,2),L_MCP5_task_isb(1,3),...
        'r','filled');

    text(R_WRS_task_isb(1,1),R_WRS_task_isb(1,2),R_WRS_task_isb(1,3),'   R_WRS');
    text(L_WRS_task_isb(1,1),L_WRS_task_isb(1,2),L_WRS_task_isb(1,3),'   L_WRS');
    text(R_WUS_task_isb(1,1),R_WUS_task_isb(1,2),R_WUS_task_isb(1,3),'   R_WUS');
    text(L_WUS_task_isb(1,1),L_WUS_task_isb(1,2),L_WUS_task_isb(1,3),'   L_WUS');

    text(R_MCP2_task_isb(1,1),R_MCP2_task_isb(1,2),R_MCP2_task_isb(1,3),...
        '   R_MCP2');
    text(L_MCP2_task_isb(1,1),L_MCP2_task_isb(1,2),L_MCP2_task_isb(1,3),...
        '   L_MCP2');
    text(R_MCP5_task_isb(1,1),R_MCP5_task_isb(1,2),R_MCP5_task_isb(1,3),...
        '   R_MCP5');
    text(L_MCP5_task_isb(1,1),L_MCP5_task_isb(1,2),L_MCP5_task_isb(1,3),...
        '   L_MCP5');

    line([R_WUS_task_isb(1,1),R_WRS_task_isb(1,1)],...
        [R_WUS_task_isb(1,2),R_WRS_task_isb(1,2)],...
        [R_WUS_task_isb(1,3),R_WRS_task_isb(1,3)],'Color','black')
    line([R_WRS_task_isb(1,1),R_MCP2_task_isb(1,1)],...
        [R_WRS_task_isb(1,2),R_MCP2_task_isb(1,2)],...
        [R_WRS_task_isb(1,3),R_MCP2_task_isb(1,3)],'Color','black')
    line([R_WUS_task_isb(1,1),R_MCP5_task_isb(1,1)],...
        [R_WUS_task_isb(1,2),R_MCP5_task_isb(1,2)],...
        [R_WUS_task_isb(1,3),R_MCP5_task_isb(1,3)],'Color','black')
    line([R_MCP2_task_isb(1,1),R_MCP5_task_isb(1,1)],...
        [R_MCP2_task_isb(1,2),R_MCP5_task_isb(1,2)],...
        [R_MCP2_task_isb(1,3),R_MCP5_task_isb(1,3)],'Color','black')
    line([R_MCP2_task_isb(1,1),R_DP3_task_isb(1,1)],...
        [R_MCP2_task_isb(1,2),R_DP3_task_isb(1,2)],...
        [R_MCP2_task_isb(1,3),R_DP3_task_isb(1,3)],'Color','black')
    line([R_MCP5_task_isb(1,1),R_DP3_task_isb(1,1)],...
        [R_MCP5_task_isb(1,2),R_DP3_task_isb(1,2)],...
        [R_MCP5_task_isb(1,3),R_DP3_task_isb(1,3)],'Color','black')
    line([R_WRS_task_isb(1,1),R_LEC_task_isb(1,1)],...
        [R_WRS_task_isb(1,2),R_LEC_task_isb(1,2)],...
        [R_WRS_task_isb(1,3),R_LEC_task_isb(1,3)],'Color','black')

    line([L_WUS_task_isb(1,1),L_WRS_task_isb(1,1)],...
        [L_WUS_task_isb(1,2),L_WRS_task_isb(1,2)],...
        [L_WUS_task_isb(1,3),L_WRS_task_isb(1,3)],'Color','black')
    line([L_WRS_task_isb(1,1),L_MCP2_task_isb(1,1)],...
        [L_WRS_task_isb(1,2),L_MCP2_task_isb(1,2)],...
        [L_WRS_task_isb(1,3),L_MCP2_task_isb(1,3)],'Color','black')
    line([L_WUS_task_isb(1,1),L_MCP5_task_isb(1,1)],...
        [L_WUS_task_isb(1,2),L_MCP5_task_isb(1,2)],...
        [L_WUS_task_isb(1,3),L_MCP5_task_isb(1,3)],'Color','black')
    line([L_MCP2_task_isb(1,1),L_MCP5_task_isb(1,1)],...
        [L_MCP2_task_isb(1,2),L_MCP5_task_isb(1,2)],...
        [L_MCP2_task_isb(1,3),L_MCP5_task_isb(1,3)],'Color','black')
    line([L_MCP2_task_isb(1,1),L_DP3_task_isb(1,1)],...
        [L_MCP2_task_isb(1,2),L_DP3_task_isb(1,2)],...
        [L_MCP2_task_isb(1,3),L_DP3_task_isb(1,3)],'Color','black')
    line([L_MCP5_task_isb(1,1),L_DP3_task_isb(1,1)],...
        [L_MCP5_task_isb(1,2),L_DP3_task_isb(1,2)],...
        [L_MCP5_task_isb(1,3),L_DP3_task_isb(1,3)],'Color','black')
    line([L_WRS_task_isb(1,1),L_LEC_task_isb(1,1)],...
        [L_WRS_task_isb(1,2),L_LEC_task_isb(1,2)],...
        [L_WRS_task_isb(1,3),L_LEC_task_isb(1,3)],'Color','black')

    scatter3(R_HND_1_task_isb(1,1),R_HND_1_task_isb(1,2),R_HND_1_task_isb(1,3),...
        'r','filled');
    scatter3(R_HND_2_task_isb(1,1),R_HND_2_task_isb(1,2),R_HND_2_task_isb(1,3),...
        'r','filled');
    scatter3(R_HND_3_task_isb(1,1),R_HND_3_task_isb(1,2),R_HND_3_task_isb(1,3),...
        'r','filled');
    scatter3(L_HND_1_task_isb(1,1),L_HND_1_task_isb(1,2),L_HND_1_task_isb(1,3),...
        'r','filled');
    scatter3(L_HND_2_task_isb(1,1),L_HND_2_task_isb(1,2),L_HND_2_task_isb(1,3),...
        'r','filled');
    scatter3(L_HND_3_task_isb(1,1),L_HND_3_task_isb(1,2),L_HND_3_task_isb(1,3),...
        'r','filled');

    % Forearm
    scatter3(R_LEC_task_isb(1,1),R_LEC_task_isb(1,2),R_LEC_task_isb(1,3),'r',...
        'filled');
    scatter3(L_LEC_task_isb(1,1),L_LEC_task_isb(1,2),L_LEC_task_isb(1,3),'r',...
        'filled');
    scatter3(R_MEC_task_isb(1,1),R_MEC_task_isb(1,2),R_MEC_task_isb(1,3),'r',...
        'filled');
    scatter3(L_MEC_task_isb(1,1),L_MEC_task_isb(1,2),L_MEC_task_isb(1,3),'r',...
        'filled');

    text(R_LEC_task_isb(1,1),R_LEC_task_isb(1,2),R_LEC_task_isb(1,3),'   R_LEC');
    text(L_LEC_task_isb(1,1),L_LEC_task_isb(1,2),L_LEC_task_isb(1,3),'   L_LEC');
    text(R_MEC_task_isb(1,1),R_MEC_task_isb(1,2),R_MEC_task_isb(1,3),'   R_MEC');
    text(L_MEC_task_isb(1,1),L_MEC_task_isb(1,2),L_MEC_task_isb(1,3),'   L_MEC');

    line([R_LEC_task_isb(1,1),R_MEC_task_isb(1,1)],...
        [R_LEC_task_isb(1,2),R_MEC_task_isb(1,2)],...
        [R_LEC_task_isb(1,3),R_MEC_task_isb(1,3)],'Color','black');
    line([L_LEC_task_isb(1,1),L_MEC_task_isb(1,1)],...
        [L_LEC_task_isb(1,2),L_MEC_task_isb(1,2)],...
        [L_LEC_task_isb(1,3),L_MEC_task_isb(1,3)],'Color','black');
    line([R_LEC_task_isb(1,1),R_AP_task_isb(1,1)],...
        [R_LEC_task_isb(1,2),R_AP_task_isb(1,2)],...
        [R_LEC_task_isb(1,3),R_AP_task_isb(1,3)],'Color','black');
    line([L_LEC_task_isb(1,1),L_AP_task_isb(1,1)],...
        [L_LEC_task_isb(1,2),L_AP_task_isb(1,2)],...
        [L_LEC_task_isb(1,3),L_AP_task_isb(1,3)],'Color','black');

    scatter3(R_FA_1_task_isb(1,1),R_FA_1_task_isb(1,2),R_FA_1_task_isb(1,3),...
        'r','filled');
    scatter3(R_FA_2_task_isb(1,1),R_FA_2_task_isb(1,2),R_FA_2_task_isb(1,3),...
        'r','filled');
    scatter3(R_FA_3_task_isb(1,1),R_FA_3_task_isb(1,2),R_FA_3_task_isb(1,3),...
        'r','filled');
    scatter3(L_FA_1_task_isb(1,1),L_FA_1_task_isb(1,2),L_FA_1_task_isb(1,3),...
        'r','filled');
    scatter3(L_FA_2_task_isb(1,1),L_FA_2_task_isb(1,2),L_FA_2_task_isb(1,3),...
        'r','filled');
    scatter3(L_FA_3_task_isb(1,1),L_FA_3_task_isb(1,2),L_FA_3_task_isb(1,3),...
        'r','filled');

    % Arm
    scatter3(R_AP_task_isb(1,1),R_AP_task_isb(1,2),R_AP_task_isb(1,3),'r',...
        'filled');
    scatter3(L_AP_task_isb(1,1),L_AP_task_isb(1,2),L_AP_task_isb(1,3),'r',...
        'filled');

    text(R_AP_task_isb(1,1),R_AP_task_isb(1,2),R_AP_task_isb(1,3),'   R_AP');
    text(L_AP_task_isb(1,1),L_AP_task_isb(1,2),L_AP_task_isb(1,3),'   L_AP');

    scatter3(R_UA_1_task_isb(1,1),R_UA_1_task_isb(1,2),R_UA_1_task_isb(1,3),...
        'r','filled');
    scatter3(R_UA_2_task_isb(1,1),R_UA_2_task_isb(1,2),R_UA_2_task_isb(1,3),...
        'r','filled');
    scatter3(R_UA_3_task_isb(1,1),R_UA_3_task_isb(1,2),R_UA_3_task_isb(1,3),...
        'r','filled');
    scatter3(L_UA_1_task_isb(1,1),L_UA_1_task_isb(1,2),L_UA_1_task_isb(1,3),...
        'r','filled');
    scatter3(L_UA_2_task_isb(1,1),L_UA_2_task_isb(1,2),L_UA_2_task_isb(1,3),...
        'r','filled');
    scatter3(L_UA_3_task_isb(1,1),L_UA_3_task_isb(1,2),L_UA_3_task_isb(1,3),...
        'r','filled');

    % Torso
    scatter3(C7_task_isb(1,1),C7_task_isb(1,2),C7_task_isb(1,3),'k','filled');

    text(C7_task_isb(1,1),C7_task_isb(1,2),C7_task_isb(1,3),'   C7');

    line([R_AP_task_isb(1,1),C7_task_isb(1,1)],...
        [R_AP_task_isb(1,2),C7_task_isb(1,2)],...
        [R_AP_task_isb(1,3),C7_task_isb(1,3)],'Color','black');
    line([L_AP_task_isb(1,1),C7_task_isb(1,1)],...
        [L_AP_task_isb(1,2),C7_task_isb(1,2)],...
        [L_AP_task_isb(1,3),C7_task_isb(1,3)],'Color','black');

    scatter3(T8_task_isb(1,1),T8_task_isb(1,2),T8_task_isb(1,3),'k','filled');

    text(T8_task_isb(1,1),T8_task_isb(1,2),T8_task_isb(1,3),'   T8');

    line([T8_task_isb(1,1),C7_task_isb(1,1)],...
        [T8_task_isb(1,2),C7_task_isb(1,2)],...
        [T8_task_isb(1,3),C7_task_isb(1,3)],'Color','black');

    scatter3(THOR_1_task_isb(1,1),THOR_1_task_isb(1,2),THOR_1_task_isb(1,3),...
        'k','filled');
    scatter3(THOR_2_task_isb(1,1),THOR_2_task_isb(1,2),THOR_2_task_isb(1,3),...
        'k','filled');
    scatter3(THOR_3_task_isb(1,1),THOR_3_task_isb(1,2),THOR_3_task_isb(1,3),...
        'k','filled');
    scatter3(THOR_4_task_isb(1,1),THOR_4_task_isb(1,2),THOR_4_task_isb(1,3),...
        'k','filled');

    scatter3(L5_task_isb(1,1),L5_task_isb(1,2),L5_task_isb(1,3),'k','filled')

    text(L5_task_isb(1,1),L5_task_isb(1,2),L5_task_isb(1,3),'   L5');

    line([L5_task_isb(1,1),T8_task_isb(1,1)],...
        [L5_task_isb(1,2),T8_task_isb(1,2)],...
        [L5_task_isb(1,3),T8_task_isb(1,3)],'Color','black');

    scatter3(LMB_1_task_isb(1,1),LMB_1_task_isb(1,2),LMB_1_task_isb(1,3),...
        'k','filled');
    scatter3(LMB_2_task_isb(1,1),LMB_2_task_isb(1,2),LMB_2_task_isb(1,3),...
        'k','filled');
    scatter3(LMB_3_task_isb(1,1),LMB_3_task_isb(1,2),LMB_3_task_isb(1,3),...
        'k','filled');

    % Pelvis
    scatter3(R_PSIS_task_isb(1,1),R_PSIS_task_isb(1,2),R_PSIS_task_isb(1,3),...
        'k','filled');
    scatter3(L_PSIS_task_isb(1,1),L_PSIS_task_isb(1,2),L_PSIS_task_isb(1,3),...
        'k','filled');

    text(R_PSIS_task_isb(1,1),R_PSIS_task_isb(1,2),R_PSIS_task_isb(1,3),...
        '   R_PSIS');
    text(L_PSIS_task_isb(1,1),L_PSIS_task_isb(1,2),L_PSIS_task_isb(1,3),...
        '   L_PSIS');

    line([R_PSIS_task_isb(1,1),L5_task_isb(1,1)],...
        [R_PSIS_task_isb(1,2),L5_task_isb(1,2)],...
        [R_PSIS_task_isb(1,3),L5_task_isb(1,3)],'Color','black');
    line([L_PSIS_task_isb(1,1),L5_task_isb(1,1)],...
        [L_PSIS_task_isb(1,2),L5_task_isb(1,2)],...
        [L_PSIS_task_isb(1,3),L5_task_isb(1,3)],'Color','black');

    xlabel ('x-axis');
    ylabel ('y-axis');
    zlabel ('z-axis');

% RECREATE SS, XP, ASIS, & MISSING MARKERS - LCS BASED INTERPOLATION
% 1. Define local coordinate system & unit vector of cal trial clusters
    % Hand
    R_HND_cal_z =...
        (R_HND_1_cal_isb - R_HND_3_cal_isb)...
        /norm(R_HND_1_cal_isb - R_HND_3_cal_isb);
    R_HND_cal_temp =...
        (R_HND_2_cal_isb - R_HND_3_cal_isb)...
        /norm (R_HND_2_cal_isb - R_HND_3_cal_isb);
    R_HND_cal_y =...
        cross (R_HND_cal_z, R_HND_cal_temp)...
        /norm (cross (R_HND_cal_z, R_HND_cal_temp));
    R_HND_cal_x =...
        cross (R_HND_cal_y, R_HND_cal_z)...
        /norm (cross(R_HND_cal_y, R_HND_cal_z));

    L_HND_cal_z =...
        (L_HND_1_cal_isb - L_HND_3_cal_isb)...
        /norm(L_HND_1_cal_isb - L_HND_3_cal_isb);
    L_HND_cal_temp =...
        (L_HND_2_cal_isb - L_HND_3_cal_isb)...
        /norm (L_HND_2_cal_isb - L_HND_3_cal_isb);
    L_HND_cal_y =...
        cross (L_HND_cal_z, L_HND_cal_temp)...
        /norm (cross (L_HND_cal_z, L_HND_cal_temp));
    L_HND_cal_x =...
        cross (L_HND_cal_y, L_HND_cal_z)...
        /norm (cross(L_HND_cal_y, L_HND_cal_z));

    % Forearm
    R_FA_cal_y =...
        (R_FA_3_cal_isb - R_FA_1_cal_isb)...
        /norm (R_FA_3_cal_isb - R_FA_1_cal_isb);
    R_FA_cal_temp =...
        (R_FA_2_cal_isb - R_FA_3_cal_isb)...
        /norm (R_FA_2_cal_isb - R_FA_3_cal_isb);
    R_FA_cal_z =...
        cross (R_FA_cal_y, R_FA_cal_temp)...
        /norm (cross (R_FA_cal_y, R_FA_cal_temp));
    R_FA_cal_x =...
        cross(R_FA_cal_z, R_FA_cal_y)...
        /norm (cross(R_FA_cal_z, R_FA_cal_y));

    L_FA_cal_y =...
        (L_FA_3_cal_isb - L_FA_1_cal_isb)...
        /norm (L_FA_3_cal_isb - L_FA_1_cal_isb);
    L_FA_cal_temp =...
        (L_FA_2_cal_isb - L_FA_3_cal_isb)...
        /norm (L_FA_2_cal_isb - L_FA_3_cal_isb);
    L_FA_cal_z =...
        cross (L_FA_cal_y, L_FA_cal_temp)...
        /norm (cross (L_FA_cal_y, L_FA_cal_temp));
    L_FA_cal_x =...
        cross(L_FA_cal_z, L_FA_cal_y)...
        /norm (cross(L_FA_cal_z, L_FA_cal_y));

    % Upper Arm - Using actual acromion marker, so do not need for SB
%     R_UA_cal_y =...
%         (R_UA_3_cal_isb - R_UA_1_cal_isb)...
%         /norm (R_UA_3_cal_isb - R_UA_1_cal_isb);
%     R_UA_cal_temp =...
%         (R_UA_2_cal_isb - R_UA_1_cal_isb)...
%         /norm (R_UA_2_cal_isb - R_UA_1_cal_isb);
%     R_UA_cal_z =...
%         cross (R_UA_cal_y, R_UA_cal_temp)...
%         /norm (cross (R_UA_cal_y, R_UA_cal_temp));
%     R_UA_cal_x =...
%         cross(R_UA_cal_z, R_UA_cal_y)...
%         /norm (cross(R_UA_cal_z, R_UA_cal_y));
% 
%     L_UA_cal_y =...
%         (L_UA_3_cal_isb - L_UA_1_cal_isb)...
%         /norm (L_UA_3_cal_isb - L_UA_1_cal_isb);
%     L_UA_cal_temp =...
%         (L_UA_2_cal_isb - L_UA_1_cal_isb)...
%         /norm (L_UA_2_cal_isb - L_UA_1_cal_isb);
%     L_UA_cal_z =...
%         cross (L_UA_cal_y, L_UA_cal_temp)...
%         /norm (cross (L_UA_cal_y, L_UA_cal_temp));
%     L_UA_cal_x =...
%         cross(L_UA_cal_z, L_UA_cal_y)...
%         /norm (cross(L_UA_cal_z, L_UA_cal_y));

    % Thoracic
    THOR_cal_z =...
        (THOR_1_cal_isb - THOR_4_cal_isb)...
        /norm (THOR_1_cal_isb - THOR_4_cal_isb);
    THOR_cal_temp =...
        (THOR_2_cal_isb - THOR_4_cal_isb)...
        /norm (THOR_2_cal_isb - THOR_4_cal_isb);
    THOR_cal_x =...
        cross (THOR_cal_z, THOR_cal_temp)...
        /norm (cross (THOR_cal_z, THOR_cal_temp));
    THOR_cal_y =...
        cross (THOR_cal_x, THOR_cal_z)...
        /norm (cross (THOR_cal_x, THOR_cal_z));

    % Lumbar
    LMB_cal_z =...
        (LMB_1_cal_isb - LMB_3_cal_isb)...
        /norm (LMB_1_cal_isb - LMB_3_cal_isb);
    LMB_cal_temp =...
        (LMB_2_cal_isb - LMB_3_cal_isb)...
        /norm (LMB_2_cal_isb - LMB_3_cal_isb);
    LMB_cal_x =...
        cross (LMB_cal_z, LMB_cal_temp)...
        /norm (cross (LMB_cal_z, LMB_cal_temp));
    LMB_cal_y =...
        cross (LMB_cal_x, LMB_cal_z)...
        /norm (cross (LMB_cal_x, LMB_cal_z));

% 2. Define rotation matrix (global to local) for cal clusters
    GLOBAL_x = [1 0 0];
    GLOBAL_y = [0 1 0];
    GLOBAL_z = [0 0 1];

    % Hand
    R_HND_cal_global_rot (1,1) = dot (R_HND_cal_x, GLOBAL_x);
    R_HND_cal_global_rot (1,2) = dot (R_HND_cal_x, GLOBAL_y);
    R_HND_cal_global_rot (1,3) = dot (R_HND_cal_x, GLOBAL_z);
    R_HND_cal_global_rot (2,1) = dot (R_HND_cal_y, GLOBAL_x);
    R_HND_cal_global_rot (2,2) = dot (R_HND_cal_y, GLOBAL_y);
    R_HND_cal_global_rot (2,3) = dot (R_HND_cal_y, GLOBAL_z);
    R_HND_cal_global_rot (3,1) = dot (R_HND_cal_z, GLOBAL_x);
    R_HND_cal_global_rot (3,2) = dot (R_HND_cal_z, GLOBAL_y);
    R_HND_cal_global_rot (3,3) = dot (R_HND_cal_z, GLOBAL_z);

    L_HND_cal_global_rot (1,1) = dot (L_HND_cal_x, GLOBAL_x);
    L_HND_cal_global_rot (1,2) = dot (L_HND_cal_x, GLOBAL_y);
    L_HND_cal_global_rot (1,3) = dot (L_HND_cal_x, GLOBAL_z);
    L_HND_cal_global_rot (2,1) = dot (L_HND_cal_y, GLOBAL_x);
    L_HND_cal_global_rot (2,2) = dot (L_HND_cal_y, GLOBAL_y);
    L_HND_cal_global_rot (2,3) = dot (L_HND_cal_y, GLOBAL_z);
    L_HND_cal_global_rot (3,1) = dot (L_HND_cal_z, GLOBAL_x);
    L_HND_cal_global_rot (3,2) = dot (L_HND_cal_z, GLOBAL_y);
    L_HND_cal_global_rot (3,3) = dot (L_HND_cal_z, GLOBAL_z);

    % Forearm
    R_FA_cal_global_rot (1,1) = dot (R_FA_cal_x, GLOBAL_x);
    R_FA_cal_global_rot (1,2) = dot (R_FA_cal_x, GLOBAL_y);
    R_FA_cal_global_rot (1,3) = dot (R_FA_cal_x, GLOBAL_z);
    R_FA_cal_global_rot (2,1) = dot (R_FA_cal_y, GLOBAL_x);
    R_FA_cal_global_rot (2,2) = dot (R_FA_cal_y, GLOBAL_y);
    R_FA_cal_global_rot (2,3) = dot (R_FA_cal_y, GLOBAL_z);
    R_FA_cal_global_rot (3,1) = dot (R_FA_cal_z, GLOBAL_x);
    R_FA_cal_global_rot (3,2) = dot (R_FA_cal_z, GLOBAL_y);
    R_FA_cal_global_rot (3,3) = dot (R_FA_cal_z, GLOBAL_z);

    L_FA_cal_global_rot (1,1) = dot (L_FA_cal_x, GLOBAL_x);
    L_FA_cal_global_rot (1,2) = dot (L_FA_cal_x, GLOBAL_y);
    L_FA_cal_global_rot (1,3) = dot (L_FA_cal_x, GLOBAL_z);
    L_FA_cal_global_rot (2,1) = dot (L_FA_cal_y, GLOBAL_x);
    L_FA_cal_global_rot (2,2) = dot (L_FA_cal_y, GLOBAL_y);
    L_FA_cal_global_rot (2,3) = dot (L_FA_cal_y, GLOBAL_z);
    L_FA_cal_global_rot (3,1) = dot (L_FA_cal_z, GLOBAL_x);
    L_FA_cal_global_rot (3,2) = dot (L_FA_cal_z, GLOBAL_y);
    L_FA_cal_global_rot (3,3) = dot (L_FA_cal_z, GLOBAL_z);

%     % Upper arm - Using actual acromion marker, so do not need for SB
%     R_UA_cal_global_rot (1,1) = dot (R_UA_cal_x, GLOBAL_x);
%     R_UA_cal_global_rot (1,2) = dot (R_UA_cal_x, GLOBAL_y);
%     R_UA_cal_global_rot (1,3) = dot (R_UA_cal_x, GLOBAL_z);
%     R_UA_cal_global_rot (2,1) = dot (R_UA_cal_y, GLOBAL_x);
%     R_UA_cal_global_rot (2,2) = dot (R_UA_cal_y, GLOBAL_y);
%     R_UA_cal_global_rot (2,3) = dot (R_UA_cal_y, GLOBAL_z);
%     R_UA_cal_global_rot (3,1) = dot (R_UA_cal_z, GLOBAL_x);
%     R_UA_cal_global_rot (3,2) = dot (R_UA_cal_z, GLOBAL_y);
%     R_UA_cal_global_rot (3,3) = dot (R_UA_cal_z, GLOBAL_z);
% 
%     L_UA_cal_global_rot (1,1) = dot (L_UA_cal_x, GLOBAL_x);
%     L_UA_cal_global_rot (1,2) = dot (L_UA_cal_x, GLOBAL_y);
%     L_UA_cal_global_rot (1,3) = dot (L_UA_cal_x, GLOBAL_z);
%     L_UA_cal_global_rot (2,1) = dot (L_UA_cal_y, GLOBAL_x);
%     L_UA_cal_global_rot (2,2) = dot (L_UA_cal_y, GLOBAL_y);
%     L_UA_cal_global_rot (2,3) = dot (L_UA_cal_y, GLOBAL_z);
%     L_UA_cal_global_rot (3,1) = dot (L_UA_cal_z, GLOBAL_x);
%     L_UA_cal_global_rot (3,2) = dot (L_UA_cal_z, GLOBAL_y);
%     L_UA_cal_global_rot (3,3) = dot (L_UA_cal_z, GLOBAL_z);
    
    % Thoracic
    THOR_cal_global_rot (1,1) = dot (THOR_cal_x, GLOBAL_x);
    THOR_cal_global_rot (1,2) = dot (THOR_cal_x, GLOBAL_y);
    THOR_cal_global_rot (1,3) = dot (THOR_cal_x, GLOBAL_z);
    THOR_cal_global_rot (2,1) = dot (THOR_cal_y, GLOBAL_x);
    THOR_cal_global_rot (2,2) = dot (THOR_cal_y, GLOBAL_y);
    THOR_cal_global_rot (2,3) = dot (THOR_cal_y, GLOBAL_z);
    THOR_cal_global_rot (3,1) = dot (THOR_cal_z, GLOBAL_x);
    THOR_cal_global_rot (3,2) = dot (THOR_cal_z, GLOBAL_y);
    THOR_cal_global_rot (3,3) = dot (THOR_cal_z, GLOBAL_z);

    % Lumbar
    LMB_cal_global_rot (1,1) = dot (LMB_cal_x, GLOBAL_x);
    LMB_cal_global_rot (1,2) = dot (LMB_cal_x, GLOBAL_y);
    LMB_cal_global_rot (1,3) = dot (LMB_cal_x, GLOBAL_z);
    LMB_cal_global_rot (2,1) = dot (LMB_cal_y, GLOBAL_x);
    LMB_cal_global_rot (2,2) = dot (LMB_cal_y, GLOBAL_y);
    LMB_cal_global_rot (2,3) = dot (LMB_cal_y, GLOBAL_z);
    LMB_cal_global_rot (3,1) = dot (LMB_cal_z, GLOBAL_x);
    LMB_cal_global_rot (3,2) = dot (LMB_cal_z, GLOBAL_y);
    LMB_cal_global_rot (3,3) = dot (LMB_cal_z, GLOBAL_z);

% 3. Define Relationship between clusters and markers for cal trial
    % HND vs. DP3, MCP2, MCP5, WRS, WUS
    R_DP3_vs_HND = R_HND_cal_global_rot*(R_DP3_cal_isb - R_HND_1_cal_isb)';
    R_MCP2_vs_HND = R_HND_cal_global_rot*(R_MCP2_cal_isb - R_HND_1_cal_isb)';
    R_MCP5_vs_HND = R_HND_cal_global_rot*(R_MCP5_cal_isb - R_HND_1_cal_isb)';
    R_WRS_vs_HND = R_HND_cal_global_rot*(R_WRS_cal_isb - R_HND_1_cal_isb)';
    R_WUS_vs_HND = R_HND_cal_global_rot*(R_WUS_cal_isb - R_HND_1_cal_isb)';

    L_DP3_vs_HND = L_HND_cal_global_rot*(L_DP3_cal_isb - L_HND_1_cal_isb)';
    L_MCP2_vs_HND = L_HND_cal_global_rot*(L_MCP2_cal_isb - L_HND_1_cal_isb)';
    L_MCP5_vs_HND = L_HND_cal_global_rot*(L_MCP5_cal_isb - L_HND_1_cal_isb)';
    L_WRS_vs_HND = L_HND_cal_global_rot*(L_WRS_cal_isb - L_HND_1_cal_isb)';
    L_WUS_vs_HND = L_HND_cal_global_rot*(L_WUS_cal_isb - L_HND_1_cal_isb)';
    
    % FA vs. WRS, WUS, LEC, MEC
    R_WRS_vs_FA = R_FA_cal_global_rot*(R_WRS_cal_isb - R_FA_1_cal_isb)';
    R_WUS_vs_FA = R_FA_cal_global_rot*(R_WUS_cal_isb - R_FA_1_cal_isb)';
    R_LEC_vs_FA = R_FA_cal_global_rot*(R_LEC_cal_isb - R_FA_1_cal_isb)';
    R_MEC_vs_FA = R_FA_cal_global_rot*(R_MEC_cal_isb - R_FA_1_cal_isb)';
    
    L_WRS_vs_FA = L_FA_cal_global_rot*(L_WRS_cal_isb - L_FA_1_cal_isb)';
    L_WUS_vs_FA = L_FA_cal_global_rot*(L_WUS_cal_isb - L_FA_1_cal_isb)';
    L_LEC_vs_FA = L_FA_cal_global_rot*(L_LEC_cal_isb - L_FA_1_cal_isb)';
    L_MEC_vs_FA = L_FA_cal_global_rot*(L_MEC_cal_isb - L_FA_1_cal_isb)';

    % UA vs. LEC, MEC, AP - - Using actual acromion marker, so do not need for SB
%     R_LEC_vs_UA = R_UA_cal_global_rot*(R_LEC_cal_isb - R_UA_1_cal_isb)';
%     R_MEC_vs_UA = R_UA_cal_global_rot*(R_MEC_cal_isb - R_UA_1_cal_isb)';
%     R_AP_vs_UA = R_UA_cal_global_rot*(R_AP_cal_isb - R_UA_3_cal_isb)';

%     L_LEC_vs_UA = L_UA_cal_global_rot*(L_LEC_cal_isb - L_UA_1_cal_isb)';
%     L_MEC_vs_UA = L_UA_cal_global_rot*(L_MEC_cal_isb - L_UA_1_cal_isb)';
%     L_AP_vs_UA = L_UA_cal_global_rot*(L_AP_cal_isb - L_UA_3_cal_isb)';

    % THOR vs. SS, XP, C7, T8, L5 (need to do for SS & XP regardless)
    SS_vs_THOR = THOR_cal_global_rot*(SS_cal_isb - THOR_1_cal_isb)';
    XP_vs_THOR = THOR_cal_global_rot*(XP_cal_isb - THOR_1_cal_isb)'; 
    C7_vs_THOR = THOR_cal_global_rot*(C7_cal_isb - THOR_1_cal_isb)';
    T8_vs_THOR = THOR_cal_global_rot*(T8_cal_isb - THOR_1_cal_isb)';
    
    L5_vs_THOR = THOR_cal_global_rot*(L5_cal_isb - THOR_1_cal_isb)';
    R_ASIS_vs_THOR = THOR_cal_global_rot*(R_ASIS_cal_isb - THOR_1_cal_isb)';
    L_ASIS_vs_THOR = THOR_cal_global_rot*(L_ASIS_cal_isb - THOR_1_cal_isb)';
    R_PSIS_vs_THOR = THOR_cal_global_rot*(R_PSIS_cal_isb - THOR_1_cal_isb)';
    L_PSIS_vs_THOR = THOR_cal_global_rot*(L_PSIS_cal_isb - THOR_1_cal_isb)';

    % LMB vs. T8, L5, ASIS, PSIS (need to do for ASIS regardless)
    T8_vs_LMB = LMB_cal_global_rot*(T8_cal_isb - LMB_1_cal_isb)';
    L5_vs_LMB = LMB_cal_global_rot*(L5_cal_isb - LMB_1_cal_isb)';
    R_ASIS_vs_LMB = LMB_cal_global_rot*(R_ASIS_cal_isb - LMB_1_cal_isb)';
    L_ASIS_vs_LMB = LMB_cal_global_rot*(L_ASIS_cal_isb - LMB_1_cal_isb)';
    R_PSIS_vs_LMB = LMB_cal_global_rot*(R_PSIS_cal_isb - LMB_1_cal_isb)';
    L_PSIS_vs_LMB = LMB_cal_global_rot*(L_PSIS_cal_isb - LMB_1_cal_isb)';

% 4. Define LCS & unit vectors for task clusters
% Use for loops to iteratively go through each frame/row
    TASK_size = size(TASK_raw(6:end,:));
    TASK_frames = (TASK_size(1)); 

    % Hand
    for frame=1:TASK_frames
        R_HND_task_z(frame,:) =...
            (R_HND_1_task_isb(frame,:) - R_HND_3_task_isb(frame,:))...
            /norm(R_HND_1_task_isb(frame,:) - R_HND_3_task_isb(frame,:));
        R_HND_task_temp(frame,:) =...
            (R_HND_2_task_isb(frame,:) - R_HND_3_task_isb(frame,:))...
            /norm (R_HND_2_task_isb(frame,:) - R_HND_3_task_isb(frame,:));
        R_HND_task_y(frame,:) =...
            cross (R_HND_task_z(frame,:), R_HND_task_temp(frame,:))...
            /norm (cross (R_HND_task_z(frame,:), R_HND_task_temp(frame,:)));
        R_HND_task_x(frame,:) =...
            cross (R_HND_task_y(frame,:), R_HND_task_z(frame,:))...
            /norm (cross(R_HND_task_y(frame,:), R_HND_task_z(frame,:)));
    end

    for frame=1:TASK_frames
        L_HND_task_z(frame,:) =...
            (L_HND_1_task_isb(frame,:) - L_HND_3_task_isb(frame,:))...
            /norm(L_HND_1_task_isb(frame,:) - L_HND_3_task_isb(frame,:));
        L_HND_task_temp(frame,:) =...
            (L_HND_2_task_isb(frame,:) - L_HND_3_task_isb(frame,:))...
            /norm (L_HND_2_task_isb(frame,:) - L_HND_3_task_isb(frame,:));
        L_HND_task_y(frame,:) =...
            cross (L_HND_task_z(frame,:), L_HND_task_temp(frame,:))...
            /norm (cross (L_HND_task_z(frame,:), L_HND_task_temp(frame,:)));
        L_HND_task_x(frame,:) =...
            cross (L_HND_task_y(frame,:), L_HND_task_z(frame,:))...
            /norm (cross(L_HND_task_y(frame,:), L_HND_task_z(frame,:)));
    end

    % Forearm
    for frame=1:TASK_frames
        R_FA_task_y(frame,:) =...
            (R_FA_3_task_isb(frame,:) - R_FA_1_task_isb(frame,:))...
            /norm (R_FA_3_task_isb(frame,:) - R_FA_1_task_isb(frame,:));
        R_FA_task_temp(frame,:) =...
            (R_FA_2_task_isb(frame,:) - R_FA_3_task_isb(frame,:))...
            /norm (R_FA_2_task_isb(frame,:) - R_FA_3_task_isb(frame,:));
        R_FA_task_z(frame,:) =...
            cross (R_FA_task_y(frame,:), R_FA_task_temp(frame,:))...
            /norm (cross (R_FA_task_y(frame,:), R_FA_task_temp(frame,:)));
        R_FA_task_x(frame,:) =...
            cross(R_FA_task_z(frame,:), R_FA_task_y(frame,:))...
            /norm (cross(R_FA_task_z(frame,:), R_FA_task_y(frame,:)));
    end
    
    for frame=1:TASK_frames
        L_FA_task_y(frame,:) =...
            (L_FA_3_task_isb(frame,:) - L_FA_1_task_isb(frame,:))...
            /norm (L_FA_3_task_isb(frame,:) - L_FA_1_task_isb(frame,:));
        L_FA_task_temp(frame,:) =...
            (L_FA_2_task_isb(frame,:) - L_FA_3_task_isb(frame,:))...
            /norm (L_FA_2_task_isb(frame,:) - L_FA_3_task_isb(frame,:));
        L_FA_task_z(frame,:) =...
            cross (L_FA_task_y(frame,:), L_FA_task_temp(frame,:))...
            /norm (cross (L_FA_task_y(frame,:), L_FA_task_temp(frame,:)));
        L_FA_task_x(frame,:) =...
            cross(L_FA_task_z(frame,:), L_FA_task_y(frame,:))...
            /norm (cross(L_FA_task_z(frame,:), L_FA_task_y(frame,:)));
    end

%     % Upper Arm - Using actual acromion marker, so do not need for SB
%     for frame=1:TASK_frames    
%         R_UA_task_y(frame,:) =...
%             (R_UA_3_task_isb(frame,:) - R_UA_1_task_isb(frame,:))...
%             /norm (R_UA_3_task_isb(frame,:) - R_UA_1_task_isb(frame,:));
%         R_UA_task_temp(frame,:) =...
%             (R_UA_2_task_isb(frame,:) - R_UA_1_task_isb(frame,:))...
%             /norm (R_UA_2_task_isb(frame,:) - R_UA_1_task_isb(frame,:));
%         R_UA_task_x(frame,:) =...
%             cross (R_UA_task_y(frame,:), R_UA_task_temp(frame,:))...
%             /norm (cross (R_UA_task_y(frame,:), R_UA_task_temp(frame,:)));
%         R_UA_task_z(frame,:) =...
%             cross(R_UA_task_y(frame,:), R_UA_task_x(frame,:))...
%             /norm (cross(R_UA_task_y(frame,:), R_UA_task_x(frame,:)));
%     end

%     for frame=1:TASK_frames - Using actual acromion marker, so do not need for SB
%         L_UA_task_y(frame,:) =...
%             (L_UA_1_task_isb(frame,:) - L_UA_3_task_isb(frame,:))...
%             /norm (L_UA_1_task_isb(frame,:) - L_UA_3_task_isb(frame,:));
%         L_UA_task_temp(frame,:) =...
%             (L_UA_2_task_isb(frame,:) - L_UA_3_task_isb(frame,:))...
%             /norm (L_UA_2_task_isb(frame,:) - L_UA_3_task_isb(frame,:));
%         L_UA_task_z(frame,:) =...
%             cross (L_UA_task_y(frame,:), L_UA_task_temp(frame,:))...
%             /norm (cross (L_UA_task_y(frame,:), L_UA_task_temp(frame,:)));
%         L_UA_task_x(frame,:) =...
%             cross(L_UA_task_z(frame,:), L_UA_task_y(frame,:))...
%             /norm (cross(L_UA_task_z(frame,:), L_UA_task_y(frame,:)));
%     end

    % Thoracic
    for frame=1:TASK_frames
        THOR_task_z(frame,:) =...
            (THOR_1_task_isb(frame,:) - THOR_4_task_isb(frame,:))...
            /norm (THOR_1_task_isb(frame,:) - THOR_4_task_isb(frame,:));
        THOR_task_temp(frame,:) =...
            (THOR_2_task_isb(frame,:) - THOR_4_task_isb(frame,:))...
            /norm (THOR_2_task_isb(frame,:) - THOR_4_task_isb(frame,:));
        THOR_task_x(frame,:) =...
            cross (THOR_task_z(frame,:), THOR_task_temp(frame,:))...
            /norm (cross (THOR_task_z(frame,:), THOR_task_temp(frame,:)));
        THOR_task_y(frame,:) =...
            cross (THOR_task_x(frame,:), THOR_task_z(frame,:))...
            /norm (cross (THOR_task_x(frame,:), THOR_task_z(frame,:)));
    end

    % Lumbar
    for frame=1:TASK_frames
        LMB_task_z(frame,:) =...
            (LMB_1_task_isb(frame,:) - LMB_3_task_isb(frame,:))...
            /norm (LMB_1_task_isb(frame,:) - LMB_3_task_isb(frame,:));
        LMB_task_temp(frame,:) =...
            (LMB_2_task_isb(frame,:) - LMB_3_task_isb(frame,:))...
            /norm (LMB_2_task_isb(frame,:) - LMB_3_task_isb(frame,:));
        LMB_task_x(frame,:) =...
            cross (LMB_task_z(frame,:), LMB_task_temp(frame,:))...
            /norm (cross (LMB_task_z(frame,:), LMB_task_temp(frame,:)));
        LMB_task_y(frame,:) =...
            cross (LMB_task_x(frame,:), LMB_task_z(frame,:))...
            /norm (cross (LMB_task_x(frame,:), LMB_task_z(frame,:)));
    end

% 5. Define rotation matrix (global to local) for task clusters
% Use for loops to iteratively go through each frame/row

    % Hand
    for frame=1:TASK_frames
        R_HND_task_global_rot (1,1) = dot (R_HND_task_x(frame,:), GLOBAL_x);
        R_HND_task_global_rot (1,2) = dot (R_HND_task_x(frame,:), GLOBAL_y);
        R_HND_task_global_rot (1,3) = dot (R_HND_task_x(frame,:), GLOBAL_z);
        R_HND_task_global_rot (2,1) = dot (R_HND_task_y(frame,:), GLOBAL_x);
        R_HND_task_global_rot (2,2) = dot (R_HND_task_y(frame,:), GLOBAL_y);
        R_HND_task_global_rot (2,3) = dot (R_HND_task_y(frame,:), GLOBAL_z);
        R_HND_task_global_rot (3,1) = dot (R_HND_task_z(frame,:), GLOBAL_x);
        R_HND_task_global_rot (3,2) = dot (R_HND_task_z(frame,:), GLOBAL_y);
        R_HND_task_global_rot (3,3) = dot (R_HND_task_z(frame,:), GLOBAL_z);
    end

    for frame=1:TASK_frames
        L_HND_task_global_rot (1,1) = dot (L_HND_task_x(frame,:), GLOBAL_x);
        L_HND_task_global_rot (1,2) = dot (L_HND_task_x(frame,:), GLOBAL_y);
        L_HND_task_global_rot (1,3) = dot (L_HND_task_x(frame,:), GLOBAL_z);
        L_HND_task_global_rot (2,1) = dot (L_HND_task_y(frame,:), GLOBAL_x);
        L_HND_task_global_rot (2,2) = dot (L_HND_task_y(frame,:), GLOBAL_y);
        L_HND_task_global_rot (2,3) = dot (L_HND_task_y(frame,:), GLOBAL_z);
        L_HND_task_global_rot (3,1) = dot (L_HND_task_z(frame,:), GLOBAL_x);
        L_HND_task_global_rot (3,2) = dot (L_HND_task_z(frame,:), GLOBAL_y);
        L_HND_task_global_rot (3,3) = dot (L_HND_task_z(frame,:), GLOBAL_z);
    end

    % Forearm
    for frame=1:TASK_frames
        R_FA_task_global_rot (1,1) = dot (R_FA_task_x(frame,:), GLOBAL_x);
        R_FA_task_global_rot (1,2) = dot (R_FA_task_x(frame,:), GLOBAL_y);
        R_FA_task_global_rot (1,3) = dot (R_FA_task_x(frame,:), GLOBAL_z);
        R_FA_task_global_rot (2,1) = dot (R_FA_task_y(frame,:), GLOBAL_x);
        R_FA_task_global_rot (2,2) = dot (R_FA_task_y(frame,:), GLOBAL_y);
        R_FA_task_global_rot (2,3) = dot (R_FA_task_y(frame,:), GLOBAL_z);
        R_FA_task_global_rot (3,1) = dot (R_FA_task_z(frame,:), GLOBAL_x);
        R_FA_task_global_rot (3,2) = dot (R_FA_task_z(frame,:), GLOBAL_y);
        R_FA_task_global_rot (3,3) = dot (R_FA_task_z(frame,:), GLOBAL_z);
    end

    for frame=1:TASK_frames
        L_FA_task_global_rot (1,1) = dot (L_FA_task_x(frame,:), GLOBAL_x);
        L_FA_task_global_rot (1,2) = dot (L_FA_task_x(frame,:), GLOBAL_y);
        L_FA_task_global_rot (1,3) = dot (L_FA_task_x(frame,:), GLOBAL_z);
        L_FA_task_global_rot (2,1) = dot (L_FA_task_y(frame,:), GLOBAL_x);
        L_FA_task_global_rot (2,2) = dot (L_FA_task_y(frame,:), GLOBAL_y);
        L_FA_task_global_rot (2,3) = dot (L_FA_task_y(frame,:), GLOBAL_z);
        L_FA_task_global_rot (3,1) = dot (L_FA_task_z(frame,:), GLOBAL_x);
        L_FA_task_global_rot (3,2) = dot (L_FA_task_z(frame,:), GLOBAL_y);
        L_FA_task_global_rot (3,3) = dot (L_FA_task_z(frame,:), GLOBAL_z);
    end

%     % Upper arm - Using actual acromion marker, so do not need for SB
%     for frame=1:TASK_frames
%         R_UA_task_global_rot (1,1) = dot (R_UA_task_x(frame,:), GLOBAL_x);
%         R_UA_task_global_rot (1,2) = dot (R_UA_task_x(frame,:), GLOBAL_y);
%         R_UA_task_global_rot (1,3) = dot (R_UA_task_x(frame,:), GLOBAL_z);
%         R_UA_task_global_rot (2,1) = dot (R_UA_task_y(frame,:), GLOBAL_x);
%         R_UA_task_global_rot (2,2) = dot (R_UA_task_y(frame,:), GLOBAL_y);
%         R_UA_task_global_rot (2,3) = dot (R_UA_task_y(frame,:), GLOBAL_z);
%         R_UA_task_global_rot (3,1) = dot (R_UA_task_z(frame,:), GLOBAL_x);
%         R_UA_task_global_rot (3,2) = dot (R_UA_task_z(frame,:), GLOBAL_y);
%         R_UA_task_global_rot (3,3) = dot (R_UA_task_z(frame,:), GLOBAL_z);
%     end
% 
%     for frame=1:TASK_frames
%         L_UA_task_global_rot (1,1) = dot (L_UA_task_x(frame,:), GLOBAL_x);
%         L_UA_task_global_rot (1,2) = dot (L_UA_task_x(frame,:), GLOBAL_y);
%         L_UA_task_global_rot (1,3) = dot (L_UA_task_x(frame,:), GLOBAL_z);
%         L_UA_task_global_rot (2,1) = dot (L_UA_task_y(frame,:), GLOBAL_x);
%         L_UA_task_global_rot (2,2) = dot (L_UA_task_y(frame,:), GLOBAL_y);
%         L_UA_task_global_rot (2,3) = dot (L_UA_task_y(frame,:), GLOBAL_z);
%         L_UA_task_global_rot (3,1) = dot (L_UA_task_z(frame,:), GLOBAL_x);
%         L_UA_task_global_rot (3,2) = dot (L_UA_task_z(frame,:), GLOBAL_y);
%         L_UA_task_global_rot (3,3) = dot (L_UA_task_z(frame,:), GLOBAL_z);
%     end

    % Thoracic
    for frame=1:TASK_frames
        THOR_task_global_rot (1,1) = dot (THOR_task_x(frame,:), GLOBAL_x);
        THOR_task_global_rot (1,2) = dot (THOR_task_x(frame,:), GLOBAL_y);
        THOR_task_global_rot (1,3) = dot (THOR_task_x(frame,:), GLOBAL_z);
        THOR_task_global_rot (2,1) = dot (THOR_task_y(frame,:), GLOBAL_x);
        THOR_task_global_rot (2,2) = dot (THOR_task_y(frame,:), GLOBAL_y);
        THOR_task_global_rot (2,3) = dot (THOR_task_y(frame,:), GLOBAL_z);
        THOR_task_global_rot (3,1) = dot (THOR_task_z(frame,:), GLOBAL_x);
        THOR_task_global_rot (3,2) = dot (THOR_task_z(frame,:), GLOBAL_y);
        THOR_task_global_rot (3,3) = dot (THOR_task_z(frame,:), GLOBAL_z);
    end

    % Lumbar
    for frame=1:TASK_frames
        LMB_task_global_rot (1,1) = dot (LMB_task_x(frame,:), GLOBAL_x);
        LMB_task_global_rot (1,2) = dot (LMB_task_x(frame,:), GLOBAL_y);
        LMB_task_global_rot (1,3) = dot (LMB_task_x(frame,:), GLOBAL_z);
        LMB_task_global_rot (2,1) = dot (LMB_task_y(frame,:), GLOBAL_x);
        LMB_task_global_rot (2,2) = dot (LMB_task_y(frame,:), GLOBAL_y);
        LMB_task_global_rot (2,3) = dot (LMB_task_y(frame,:), GLOBAL_z);
        LMB_task_global_rot (3,1) = dot (LMB_task_z(frame,:), GLOBAL_x);
        LMB_task_global_rot (3,2) = dot (LMB_task_z(frame,:), GLOBAL_y);
        LMB_task_global_rot (3,3) = dot (LMB_task_z(frame,:), GLOBAL_z);
    end

% 6. Create virtual markers from the cal & task relationships
% Select one cluster per marker and comment out unnecessary

    % Hand to recreate: DP3, MCP2, MCP5, WRS, or WUS
    R_DP3_task_virtual = (R_HND_1_task_isb' + R_HND_task_global_rot*R_DP3_vs_HND)';
    R_MCP2_task_virtual = (R_HND_1_task_isb' + R_HND_task_global_rot*R_MCP2_vs_HND)';
    R_MCP5_task_virtual = (R_HND_1_task_isb' + R_HND_task_global_rot*R_MCP5_vs_HND)';
    R_WRS_task_virtual = (R_HND_1_task_isb' + R_HND_task_global_rot*R_WRS_vs_HND)';
    R_WUS_task_virtual = (R_HND_1_task_isb' + R_HND_task_global_rot*R_WUS_vs_HND)';

    L_DP3_task_virtual = (L_HND_1_task_isb' + L_HND_task_global_rot*L_DP3_vs_HND)';
    L_MCP2_task_virtual = (L_HND_1_task_isb' + L_HND_task_global_rot*L_MCP2_vs_HND)';
    L_MCP5_task_virtual = (L_HND_1_task_isb' + L_HND_task_global_rot*L_MCP5_vs_HND)';
    L_WRS_task_virtual = (L_HND_1_task_isb' + L_HND_task_global_rot*L_WRS_vs_HND)';
    L_WUS_task_virtual = (L_HND_1_task_isb' + L_HND_task_global_rot*L_WUS_vs_HND)';

    % Forearm to recreate: LEC, or MEC, don't use to recreate WRS/WUS, doesn't work for task
    R_LEC_task_virtual = (R_FA_1_task_isb' + R_FA_task_global_rot*R_LEC_vs_FA)';
    R_MEC_task_virtual = (R_FA_1_task_isb' + R_FA_task_global_rot*R_MEC_vs_FA)';

    L_LEC_task_virtual = (L_FA_1_task_isb' + L_FA_task_global_rot*L_LEC_vs_FA)';
    L_MEC_task_virtual = (L_FA_1_task_isb' + L_FA_task_global_rot*L_MEC_vs_FA)';

    % R_WRS_task_virtual = (R_FA_1_task_isb' + R_FA_task_global_rot*R_WRS_vs_FA)';
    % R_WUS_task_virtual = (R_FA_1_task_isb' + R_FA_task_global_rot*R_WUS_vs_FA)';
    % 
    % L_WRS_task_virtual = (L_FA_1_task_isb' + L_FA_task_global_rot*L_WRS_vs_FA)';
    % L_WUS_task_virtual = (L_FA_1_task_isb' + L_FA_task_global_rot*L_WUS_vs_FA)';

    % Upper arm to recreate: LEC, MEC, or AP - Using actual acromion marker, so do not need for SB
%     R_AP_task_isb = (R_UA_1_task_isb' + R_UA_task_global_rot*R_AP_vs_UA)';
%     
%     L_AP_task_virtual = (L_UA_1_task_isb' + L_UA_task_global_rot*L_AP_vs_UA)';

%   R_LEC_task_virtual = (R_UA_1_task_isb' + R_UA_task_global_rot*R_LEC_vs_UA)';
%   R_MEC_task_virtual = (R_UA_1_task_isb' + R_UA_task_global_rot*R_MEC_vs_UA)';

%   L_LEC_task_virtual = (L_UA_1_task_isb' + L_UA_task_global_rot*L_LEC_vs_UA)';
%   L_MEC_task_virtual = (L_UA_1_task_isb' + L_UA_task_global_rot*L_MEC_vs_UA)';

    % Thoracic to recreate: SS, XP, C7, T8, or L5 (must recreate SS & XP)
    SS_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*SS_vs_THOR)';
    XP_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*XP_vs_THOR)';
    C7_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*C7_vs_THOR)';
    T8_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*T8_vs_THOR)';

    % L5_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*L5_vs_THOR)';
    % 
    % R_ASIS_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*R_ASIS_vs_THOR)';
    % L_ASIS_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*L_ASIS_vs_THOR)';
    % 
    % R_PSIS_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*R_PSIS_vs_THOR)';
    % L_PSIS_task_virtual = (THOR_1_task_isb' + THOR_task_global_rot*L_PSIS_vs_THOR)';

    % Lumbar to recreate: L5, ASIS, or PSIS (must recreate ASIS)
    L5_task_virtual  = (LMB_1_task_isb' + LMB_task_global_rot*L5_vs_LMB)';

    R_ASIS_task_virtual = (LMB_1_task_isb' + LMB_task_global_rot*R_ASIS_vs_LMB)';
    L_ASIS_task_virtual = (LMB_1_task_isb' + LMB_task_global_rot*L_ASIS_vs_LMB)';

    R_PSIS_task_virtual = (LMB_1_task_isb' + LMB_task_global_rot*R_PSIS_vs_LMB)';
    L_PSIS_task_virtual = (LMB_1_task_isb' + LMB_task_global_rot*L_PSIS_vs_LMB)';

% 7. Plot/visual check of virtual markers
figure('Name', TASK_file + "_VIRTUAL" , 'NumberTitle', 'off')
    % Hand
    scatter3(R_DP3_task_virtual(1,1), R_DP3_task_virtual(1,2), R_DP3_task_virtual(1,3),...
   'r','filled');
    hold on
    scatter3(L_DP3_task_virtual(1,1), L_DP3_task_virtual(1,2), L_DP3_task_virtual(1,3),...
    'r','filled'); 

    text(R_DP3_task_virtual(1,1),R_DP3_task_virtual(1,2),R_DP3_task_virtual(1,3),'   R_DP3');
    text(L_DP3_task_virtual(1,1),L_DP3_task_virtual(1,2),L_DP3_task_virtual(1,3),'   L_DP3');

    scatter3(R_WUS_task_virtual(1,1), R_WUS_task_virtual(1,2), R_WUS_task_virtual(1,3),...
        'r','filled'); 
    scatter3(L_WUS_task_virtual(1,1), L_WUS_task_virtual(1,2), L_WUS_task_virtual(1,3),...
        'r','filled');
    scatter3(R_WRS_task_virtual(1,1), R_WRS_task_virtual(1,2), R_WRS_task_virtual(1,3),...
        'r','filled'); 
    scatter3(L_WRS_task_virtual(1,1), L_WRS_task_virtual(1,2), L_WRS_task_virtual(1,3),...
        'r','filled'); 

    scatter3(R_MCP2_task_virtual(1,1),R_MCP2_task_virtual(1,2),R_MCP2_task_virtual(1,3),...
        'r','filled');
    scatter3(L_MCP2_task_virtual(1,1),L_MCP2_task_virtual(1,2),L_MCP2_task_virtual(1,3),...
        'r','filled');
    scatter3(R_MCP5_task_virtual(1,1),R_MCP5_task_virtual(1,2),R_MCP5_task_virtual(1,3),...
        'r','filled');
    scatter3(L_MCP5_task_virtual(1,1),L_MCP5_task_virtual(1,2),L_MCP5_task_virtual(1,3),...
        'r','filled');

    text(R_WRS_task_virtual(1,1),R_WRS_task_virtual(1,2),R_WRS_task_virtual(1,3),'   R_WRS');
    text(L_WRS_task_virtual(1,1),L_WRS_task_virtual(1,2),L_WRS_task_virtual(1,3),'   L_WRS');
    text(R_WUS_task_virtual(1,1),R_WUS_task_virtual(1,2),R_WUS_task_virtual(1,3),'   R_WUS');
    text(L_WUS_task_virtual(1,1),L_WUS_task_virtual(1,2),L_WUS_task_virtual(1,3),'   L_WUS');

    text(R_MCP2_task_virtual(1,1),R_MCP2_task_virtual(1,2),R_MCP2_task_virtual(1,3),...
        '   R_MCP2');
    text(L_MCP2_task_virtual(1,1),L_MCP2_task_virtual(1,2),L_MCP2_task_virtual(1,3),...
        '   L_MCP2');
    text(R_MCP5_task_virtual(1,1),R_MCP5_task_virtual(1,2),R_MCP5_task_virtual(1,3),...
        '   R_MCP5');
    text(L_MCP5_task_virtual(1,1),L_MCP5_task_virtual(1,2),L_MCP5_task_virtual(1,3),...
        '   L_MCP5');

    line([R_WUS_task_virtual(1,1),R_WRS_task_virtual(1,1)],...
        [R_WUS_task_virtual(1,2),R_WRS_task_virtual(1,2)],...
        [R_WUS_task_virtual(1,3),R_WRS_task_virtual(1,3)],'Color','black')
    line([R_WRS_task_virtual(1,1),R_MCP2_task_virtual(1,1)],...
        [R_WRS_task_virtual(1,2),R_MCP2_task_virtual(1,2)],...
        [R_WRS_task_virtual(1,3),R_MCP2_task_virtual(1,3)],'Color','black')
    line([R_WUS_task_virtual(1,1),R_MCP5_task_virtual(1,1)],...
        [R_WUS_task_virtual(1,2),R_MCP5_task_virtual(1,2)],...
        [R_WUS_task_virtual(1,3),R_MCP5_task_virtual(1,3)],'Color','black')
    line([R_MCP2_task_virtual(1,1),R_MCP5_task_virtual(1,1)],...
        [R_MCP2_task_virtual(1,2),R_MCP5_task_virtual(1,2)],...
        [R_MCP2_task_virtual(1,3),R_MCP5_task_virtual(1,3)],'Color','black')
    line([R_MCP2_task_virtual(1,1),R_DP3_task_virtual(1,1)],...
        [R_MCP2_task_virtual(1,2),R_DP3_task_virtual(1,2)],...
        [R_MCP2_task_virtual(1,3),R_DP3_task_virtual(1,3)],'Color','black')
    line([R_MCP5_task_virtual(1,1),R_DP3_task_virtual(1,1)],...
        [R_MCP5_task_virtual(1,2),R_DP3_task_virtual(1,2)],...
        [R_MCP5_task_virtual(1,3),R_DP3_task_virtual(1,3)],'Color','black')
    line([R_WRS_task_virtual(1,1),R_LEC_task_virtual(1,1)],...
        [R_WRS_task_virtual(1,2),R_LEC_task_virtual(1,2)],...
        [R_WRS_task_virtual(1,3),R_LEC_task_virtual(1,3)],'Color','black')

    line([L_WUS_task_virtual(1,1),L_WRS_task_virtual(1,1)],...
        [L_WUS_task_virtual(1,2),L_WRS_task_virtual(1,2)],...
        [L_WUS_task_virtual(1,3),L_WRS_task_virtual(1,3)],'Color','black')
    line([L_WRS_task_virtual(1,1),L_MCP2_task_virtual(1,1)],...
        [L_WRS_task_virtual(1,2),L_MCP2_task_virtual(1,2)],...
        [L_WRS_task_virtual(1,3),L_MCP2_task_virtual(1,3)],'Color','black')
    line([L_WUS_task_virtual(1,1),L_MCP5_task_virtual(1,1)],...
        [L_WUS_task_virtual(1,2),L_MCP5_task_virtual(1,2)],...
        [L_WUS_task_virtual(1,3),L_MCP5_task_virtual(1,3)],'Color','black')
    line([L_MCP2_task_virtual(1,1),L_MCP5_task_virtual(1,1)],...
        [L_MCP2_task_virtual(1,2),L_MCP5_task_virtual(1,2)],...
        [L_MCP2_task_virtual(1,3),L_MCP5_task_virtual(1,3)],'Color','black')
    line([L_MCP2_task_virtual(1,1),L_DP3_task_virtual(1,1)],...
        [L_MCP2_task_virtual(1,2),L_DP3_task_virtual(1,2)],...
        [L_MCP2_task_virtual(1,3),L_DP3_task_virtual(1,3)],'Color','black')
    line([L_MCP5_task_virtual(1,1),L_DP3_task_virtual(1,1)],...
        [L_MCP5_task_virtual(1,2),L_DP3_task_virtual(1,2)],...
        [L_MCP5_task_virtual(1,3),L_DP3_task_virtual(1,3)],'Color','black')
    line([L_WRS_task_virtual(1,1),L_LEC_task_virtual(1,1)],...
        [L_WRS_task_virtual(1,2),L_LEC_task_virtual(1,2)],...
        [L_WRS_task_virtual(1,3),L_LEC_task_virtual(1,3)],'Color','black')


    % Forearm
    scatter3(R_LEC_task_virtual(1,1),R_LEC_task_virtual(1,2),R_LEC_task_virtual(1,3),'r',...
        'filled');
    scatter3(L_LEC_task_virtual(1,1),L_LEC_task_virtual(1,2),L_LEC_task_virtual(1,3),'r',...
        'filled');
    scatter3(R_MEC_task_virtual(1,1),R_MEC_task_virtual(1,2),R_MEC_task_virtual(1,3),'r',...
        'filled');
    scatter3(L_MEC_task_virtual(1,1),L_MEC_task_virtual(1,2),L_MEC_task_virtual(1,3),'r',...
        'filled');

    text(R_LEC_task_virtual(1,1),R_LEC_task_virtual(1,2),R_LEC_task_virtual(1,3),'   R_LEC');
    text(L_LEC_task_virtual(1,1),L_LEC_task_virtual(1,2),L_LEC_task_virtual(1,3),'   L_LEC');
    text(R_MEC_task_virtual(1,1),R_MEC_task_virtual(1,2),R_MEC_task_virtual(1,3),'   R_MEC');
    text(L_MEC_task_virtual(1,1),L_MEC_task_virtual(1,2),L_MEC_task_virtual(1,3),'   L_MEC');

    line([R_LEC_task_virtual(1,1),R_MEC_task_virtual(1,1)],...
        [R_LEC_task_virtual(1,2),R_MEC_task_virtual(1,2)],...
        [R_LEC_task_virtual(1,3),R_MEC_task_virtual(1,3)],'Color','black');
    line([L_LEC_task_virtual(1,1),L_MEC_task_virtual(1,1)],...
        [L_LEC_task_virtual(1,2),L_MEC_task_virtual(1,2)],...
        [L_LEC_task_virtual(1,3),L_MEC_task_virtual(1,3)],'Color','black');
    line([R_LEC_task_virtual(1,1),R_AP_task_isb(1,1)],...
        [R_LEC_task_virtual(1,2),R_AP_task_isb(1,2)],...
        [R_LEC_task_virtual(1,3),R_AP_task_isb(1,3)],'Color','black');
    line([L_LEC_task_virtual(1,1),L_AP_task_isb(1,1)],...
        [L_LEC_task_virtual(1,2),L_AP_task_isb(1,2)],...
        [L_LEC_task_virtual(1,3),L_AP_task_isb(1,3)],'Color','black');

    % Arm
    scatter3(R_AP_task_isb(1,1),R_AP_task_isb(1,2),R_AP_task_isb(1,3),'r',...
        'filled');
    scatter3(L_AP_task_isb(1,1),L_AP_task_isb(1,2),L_AP_task_isb(1,3),'r',...
        'filled');

    text(R_AP_task_isb(1,1),R_AP_task_isb(1,2),R_AP_task_isb(1,3),'   R_AP');
    text(L_AP_task_isb(1,1),L_AP_task_isb(1,2),L_AP_task_isb(1,3),'   L_AP');

    % Torso
    scatter3(C7_task_virtual(1,1),C7_task_virtual(1,2),C7_task_virtual(1,3),'k','filled');

    text(C7_task_virtual(1,1),C7_task_virtual(1,2),C7_task_virtual(1,3),'   C7');

    line([R_AP_task_isb(1,1),C7_task_virtual(1,1)],...
        [R_AP_task_isb(1,2),C7_task_virtual(1,2)],...
        [R_AP_task_isb(1,3),C7_task_virtual(1,3)],'Color','black');
    line([L_AP_task_isb(1,1),C7_task_virtual(1,1)],...
        [L_AP_task_isb(1,2),C7_task_virtual(1,2)],...
        [L_AP_task_isb(1,3),C7_task_virtual(1,3)],'Color','black');

    scatter3(SS_task_virtual(1,1),SS_task_virtual(1,2),SS_task_virtual(1,3),'k','filled');

    text(SS_task_virtual(1,1),SS_task_virtual(1,2),SS_task_virtual(1,3),'   SS');

    line([C7_task_virtual(1,1),SS_task_virtual(1,1)],...
        [C7_task_virtual(1,2),SS_task_virtual(1,2)],...
        [C7_task_virtual(1,3),SS_task_virtual(1,3)],'Color','black');

    scatter3(XP_task_virtual(1,1),XP_task_virtual(1,2),XP_task_virtual(1,3),'k','filled');

    text(XP_task_virtual(1,1),XP_task_virtual(1,2),XP_task_virtual(1,3),'   XP');

    line([SS_task_virtual(1,1),XP_task_virtual(1,1)],...
        [SS_task_virtual(1,2),XP_task_virtual(1,2)],...
        [SS_task_virtual(1,3),XP_task_virtual(1,3)],'Color','black');

    scatter3(T8_task_virtual(1,1),T8_task_virtual(1,2),T8_task_virtual(1,3),'k','filled');

    text(T8_task_virtual(1,1),T8_task_virtual(1,2),T8_task_virtual(1,3),'   T8');

    line([T8_task_virtual(1,1),C7_task_virtual(1,1)],...
        [T8_task_virtual(1,2),C7_task_virtual(1,2)],...
        [T8_task_virtual(1,3),C7_task_virtual(1,3)],'Color','black');

    scatter3(L5_task_virtual(1,1),L5_task_virtual(1,2),L5_task_virtual(1,3),'k','filled')

    text(L5_task_virtual(1,1),L5_task_virtual(1,2),L5_task_virtual(1,3),'   L5');

    line([L5_task_virtual(1,1),T8_task_virtual(1,1)],...
        [L5_task_virtual(1,2),T8_task_virtual(1,2)],...
        [L5_task_virtual(1,3),T8_task_virtual(1,3)],'Color','black');

    % Pelvis
    scatter3(R_ASIS_task_virtual(1,1),R_ASIS_task_virtual(1,2),R_ASIS_task_virtual(1,3),...
        'k','filled');
    scatter3(L_ASIS_task_virtual(1,1),L_ASIS_task_virtual(1,2),L_ASIS_task_virtual(1,3),...
        'k','filled');
    scatter3(R_PSIS_task_virtual(1,1),R_PSIS_task_virtual(1,2),R_PSIS_task_virtual(1,3),...
        'k','filled');
    scatter3(L_PSIS_task_virtual(1,1),L_PSIS_task_virtual(1,2),L_PSIS_task_virtual(1,3),...
        'k','filled');

    text(R_ASIS_task_virtual(1,1),R_ASIS_task_virtual(1,2),R_ASIS_task_virtual(1,3),...
        '   R_ASIS');
    text(L_ASIS_task_virtual(1,1),L_ASIS_task_virtual(1,2),L_ASIS_task_virtual(1,3),...
        '   L_ASIS');
    text(R_PSIS_task_virtual(1,1),R_PSIS_task_virtual(1,2),R_PSIS_task_virtual(1,3),...
        '   R_PSIS');
    text(L_PSIS_task_virtual(1,1),L_PSIS_task_virtual(1,2),L_PSIS_task_virtual(1,3),...
        '   L_PSIS');

    line([R_ASIS_task_virtual(1,1),L_ASIS_task_virtual(1,1)],...
        [R_ASIS_task_virtual(1,2),L_ASIS_task_virtual(1,2)],...
        [R_ASIS_task_virtual(1,3),L_ASIS_task_virtual(1,3)],'Color','black');
    line([R_PSIS_task_virtual(1,1),L_PSIS_task_virtual(1,1)],...
        [R_ASIS_task_virtual(1,2),L_PSIS_task_virtual(1,2)],...
        [R_PSIS_task_virtual(1,3),L_PSIS_task_virtual(1,3)],'Color','black');
    line([R_ASIS_task_virtual(1,1),R_PSIS_task_virtual(1,1)],...
        [R_ASIS_task_virtual(1,2),R_PSIS_task_virtual(1,2)],...
        [R_ASIS_task_virtual(1,3),R_PSIS_task_virtual(1,3)],'Color','black');
    line([L_ASIS_task_virtual(1,1),L_PSIS_task_virtual(1,1)],...
        [L_ASIS_task_virtual(1,2),L_PSIS_task_virtual(1,2)],...
        [L_ASIS_task_virtual(1,3),L_PSIS_task_virtual(1,3)],'Color','black');

    line([R_PSIS_task_virtual(1,1),L5_task_virtual(1,1)],...
        [R_PSIS_task_virtual(1,2),L5_task_virtual(1,2)],...
        [R_PSIS_task_virtual(1,3),L5_task_virtual(1,3)],'Color','black');
    line([L_PSIS_task_virtual(1,1),L5_task_virtual(1,1)],...
        [L_PSIS_task_virtual(1,2),L5_task_virtual(1,2)],...
        [L_PSIS_task_virtual(1,3),L5_task_virtual(1,3)],'Color','black');

    xlabel ('x-axis');
    ylabel ('y-axis');
    zlabel ('z-axis');

% LOW PASS FILTER INDIVIDUAL TASK MARKERS
    % 2nd order, zero phase shift butterworth filter at 6Hz
    % command = butter(n,Wn)
    % n = 0.5*filter order 
    % Wn = cutt off/(sample rate/2)
        % sample rate = 150 Hz
        % cut off = 6 Hz
    % Wn = 6/(150/2)
    % Wn = 6/75

    [b,a] = butter (2,6/75);

    % Replace 'isb' with 'virtual' if using the recreated marker
    R_DP3_task_filt = filtfilt(b,a, R_DP3_task_isb);
    L_DP3_task_filt = filtfilt(b,a, L_DP3_task_isb);

    R_MCP2_task_filt = filtfilt(b,a, R_MCP2_task_isb);
    R_MCP5_task_filt = filtfilt(b,a, R_MCP5_task_isb);
    L_MCP2_task_filt = filtfilt(b,a, L_MCP2_task_isb);
    L_MCP5_task_filt = filtfilt(b,a, L_MCP5_task_isb);

    R_WRS_task_filt = filtfilt(b,a, R_WRS_task_isb);
    R_WUS_task_filt = filtfilt(b,a, R_WUS_task_isb);
    L_WRS_task_filt = filtfilt(b,a, L_WRS_task_isb);
    L_WUS_task_filt = filtfilt(b,a, L_WUS_task_isb);

    R_LEC_task_filt = filtfilt(b,a, R_LEC_task_isb);
    R_MEC_task_filt = filtfilt(b,a, R_MEC_task_isb);
    L_LEC_task_filt = filtfilt(b,a, L_LEC_task_isb);
    L_MEC_task_filt = filtfilt(b,a, L_MEC_task_isb);

    R_AP_task_filt = filtfilt(b,a, R_AP_task_isb);
    L_AP_task_filt = filtfilt(b,a, L_AP_task_isb);

    SS_task_filt = filtfilt(b,a, SS_task_virtual);
    XP_task_filt = filtfilt(b,a, XP_task_virtual);

    C7_task_filt = filtfilt(b,a, C7_task_isb);
    T8_task_filt = filtfilt(b,a, T8_task_isb);
    L5_task_filt = filtfilt(b,a, L5_task_isb);

    R_ASIS_task_filt = filtfilt(b,a, R_ASIS_task_virtual);
    L_ASIS_task_filt = filtfilt(b,a, L_ASIS_task_virtual);

    R_PSIS_task_filt = filtfilt(b,a, R_PSIS_task_isb);
    L_PSIS_task_filt = filtfilt(b,a, L_PSIS_task_isb);

% EXPORT TASK
% 1. Convert padded data from array to table
    % Replace 'virtual' with 'isb' if using orignal marker
    R_DP3_task_processed = array2table(R_DP3_task_filt);
    L_DP3_task_processed = array2table(L_DP3_task_filt);

    R_MCP2_task_processed = array2table(R_MCP2_task_filt);
    R_MCP5_task_processed = array2table(R_MCP5_task_filt);
    L_MCP2_task_processed = array2table(L_MCP2_task_filt);
    L_MCP5_task_processed = array2table(L_MCP5_task_filt);

    R_WRS_task_processed = array2table(R_WRS_task_filt);
    R_WUS_task_processed = array2table(R_WUS_task_filt);
    L_WRS_task_processed = array2table(L_WRS_task_filt);
    L_WUS_task_processed = array2table(L_WUS_task_filt);

    R_LEC_task_processed = array2table(R_LEC_task_filt);
    R_MEC_task_processed = array2table(R_MEC_task_filt);
    L_LEC_task_processed = array2table(L_LEC_task_filt);
    L_MEC_task_processed = array2table(L_MEC_task_filt);

    R_AP_task_processed = array2table(R_AP_task_filt);
    L_AP_task_processed = array2table(L_AP_task_filt);

    SS_task_processed = array2table(SS_task_filt);
    XP_task_processed = array2table(XP_task_filt);

    C7_task_processed = array2table(C7_task_filt);
    T8_task_processed = array2table(T8_task_filt);
    L5_task_processed = array2table(L5_task_filt);

    R_ASIS_task_processed = array2table(R_ASIS_task_filt);
    L_ASIS_task_processed = array2table(L_ASIS_task_filt);

    R_PSIS_task_processed = array2table(R_PSIS_task_filt);
    L_PSIS_task_processed = array2table(L_PSIS_task_filt);

% 2. Combine all data into 1 table
    TASK_processed = ...
        [R_DP3_task_processed, L_DP3_task_processed,...
        R_MCP2_task_processed, L_MCP2_task_processed,...
        R_MCP5_task_processed, L_MCP5_task_processed,...
        R_WRS_task_processed, L_WRS_task_processed,...
        R_WUS_task_processed, L_WUS_task_processed,...
        R_LEC_task_processed, L_LEC_task_processed,...
        R_MEC_task_processed, L_MEC_task_processed,...
        R_AP_task_processed, L_AP_task_processed,...
        SS_task_processed, XP_task_processed,...
        C7_task_processed, T8_task_processed, L5_task_processed,...
        R_ASIS_task_processed, L_ASIS_task_processed,...
        R_PSIS_task_processed, L_PSIS_task_processed];

% 3. Export as csv
    writetable(TASK_processed, TASK_dir + "Processed\" + "PROCESSED_"...
        + TASK_file);
end

%% ----------------------- AVERAGE TRIALS ----------------------------- %%
% SET UP PROCESSED FILE DIRECTORIES
TASK_processed_dir = "C:\Users\kathr\Documents\Waterloo MSc\Thesis\Data\KINEMATICS\" ...
        + subject_num + "_KIN\Processed\";

TASK_processed_list = dir (TASK_processed_dir + 'PROCESSED_' + subject_num + "*.csv");
TASK_processed_list_index = 1:length(TASK_processed_list);

% READ IN FILES
ROT_1_file = TASK_processed_list (TASK_processed_list_index(1)).name;
ROT_2_file = TASK_processed_list (TASK_processed_list_index(2)).name;
ROT_3_file = TASK_processed_list (TASK_processed_list_index(3)).name;

STAT_1_file = TASK_processed_list (TASK_processed_list_index(4)).name;
STAT_2_file = TASK_processed_list (TASK_processed_list_index(5)).name;
STAT_3_file = TASK_processed_list (TASK_processed_list_index(6)).name;

ROT_1 = readmatrix (TASK_processed_dir + ROT_1_file);
ROT_2 = readmatrix (TASK_processed_dir + ROT_2_file);
ROT_3 = readmatrix (TASK_processed_dir + ROT_3_file);

STAT_1 = readmatrix (TASK_processed_dir + STAT_1_file);
STAT_2 = readmatrix (TASK_processed_dir + STAT_2_file);
STAT_3 = readmatrix (TASK_processed_dir + STAT_3_file);

% AVERAGE TRIALS BY SETTING
ROT_AVG = ((ROT_1 + ROT_2 + ROT_3)/3);
STAT_AVG = ((STAT_1 + STAT_2 + STAT_3)/3);

% EXPORT AVERAGE TRIAL
writematrix(ROT_AVG, TASK_processed_dir + "AVG_" + subject_num + "_ROT.csv");
writematrix(STAT_AVG, TASK_processed_dir + "AVG_" + subject_num + "_STAT.csv");