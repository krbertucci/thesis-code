%TELUS MOMENT CALCULATION

clear all
clc
%% *********Change the folder for each participant
[calflnm, calflpth, cCheck]=uigetfile('*.*',...
                                  'Select Calibration Trial',...
                                  'C:\Users\jpskurt\OneDrive - University of Waterloo\Ladder Study\matlab\TELUS\S04',... %This has to be changed depending on what PC you're on
                                  'Multiselect', 'on');

data=csvread([calflpth calflnm],6,2); %changed to 6,3 since 5,2 has 'mm' not a value
Subject=calflpth(end-2:end-1); %JK: says the subject # is the last 2 characters of the folder you're in - matches up with the format we have
[filenames, pathnames] = uigetfile('*.csv', 'Select all Trial Files', 'MultiSelect','on'); %JK: only shows CSV's to select, asking to select all trial files.
[C, num_files] = size(filenames); %JK: not sure what this is doing here.. is this saying how many files were selected?

for k=1:num_files %repeat for the number of files that have been selected?
    files=char(filenames(k)); %JK: create character array?
    trialname=files(1:end-4)
    carry_data=csvread([pathnames,files],6,2); %JK: changed to 6,3 since 6,2 is  a 'subframe'

CalFrame=30; %JK: INPUT THIS FOR EACH PARTICIPANT - STAYS THE SAME FOR EVERY TRIAL 
mass=85.5; %JK: INPUT THIS FOR EACH PARTICIPANT

%cluster=input('Input cluster for Trial:'); %JK: commented out using the
%cluster as an input since I'll be changing this later in the code

ladder = input('input ladder: ');
task  = input('input task: ');
if ladder==1 %JK: green ladder
        if task==1 %carry 
        ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here
        

        elseif task==2 %JK: free raise top
        ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here; %not changing this to an input since the ladder force for raise shoulder will be 0
        
        elseif task==3 %JK: free raise bottom
        ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here
    
elseif ladder==2 
       if task==1 %carry 
        ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here

        elseif task==2 %JK: free raise top
         ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: ');
        
        elseif task==3 %JK: free raise bottom
        ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here
        end 
        
            
elseif ladder==3 %%added two more tasks here for the extra frames we're considering with the extra piece of the ladder 
        if task==1 %carry 
        ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here

        elseif task==2 %JK: free raise bottom lift
       ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here
        
        elseif task==3 %JK: free raise top lift
        ladder_force_rightShoulder=input('Shoulder force: ');  
        ladder_force_leftHand=input('L hand force: ');
        ladder_force_rightHand=input('R hand force: '); %JK: making the forces into inputs here
        
        end 
end 
    
        
%READ IN ALL CALIBRATION FILES/MARKERS
%The relationship between the tracking markers and anatomical markers will
%be defined in frame '25' of the trial 

%** YOU CAN CHOOSE WHICHEVER FRAME YOU LIKE

%--------------------------------------------------------------------------------------------------

%Input the cols for the torso cluster chosen %JK: check these are in the
%right order
C7=data(CalFrame,1:3);
L5=data(CalFrame,4:6);
SS=data(CalFrame,7:9);
XP=data(CalFrame,10:12);
Rt_Acro=data(CalFrame,19:21); 
Lft_Acro=data(CalFrame,22:24);
chest_c1_1=data(CalFrame,25:27); %commented out since we aren't using

chest_c1_2=data(CalFrame,28:30);
chest_c1_5=data(CalFrame,37:39);



T8=data(CalFrame,49:51);  %cluster marker 4 since we either had the cluster to the L or in the middle 

T8_c1_1=data(CalFrame,40:42); 
T8_c1_2=data(CalFrame,43:45);
T8_c1_3=data(CalFrame,46:48);
T8_c1_4=data(CalFrame,49:51);

L2_c1_1=data(CalFrame,52:54); 
L2_c1_2=data(CalFrame,55:57);
L2_c1_3=data(CalFrame,58:60);

Rt_LE=data(CalFrame,61:63);
Rt_ME=data(CalFrame,64:66);
Rt_hum_cl_1=data(CalFrame,67:69);
Rt_hum_cl_2=data(CalFrame,70:72);
Rt_hum_cl_3=data(CalFrame,73:75);

Rt_US=data(CalFrame,76:78);
Rt_RS=data(CalFrame,79:81);
Rt_FA_cl_1=data(CalFrame,82:84);
Rt_FA_cl_2=data(CalFrame,85:87);
Rt_FA_cl_3=data(CalFrame,88:90);

Lft_LE=data(CalFrame,97:99);
Lft_ME=data(CalFrame,100:102);
Lft_hum_cl_1=data(CalFrame,103:105);
Lft_hum_cl_2=data(CalFrame,106:108);
Lft_hum_cl_3=data(CalFrame,109:111);

Lft_FA_cl_1=data(CalFrame,118:120);
Lft_FA_cl_2=data(CalFrame,121:123);
Lft_FA_cl_3=data(CalFrame,124:126);
Lft_US=data(CalFrame,112:114);
Lft_RS=data(CalFrame,115:117);

%Rotate 3D coordinates from Global Coordinate System in 
% Lab (+X=right, +Y=forwards, +Z=up) 
% to ISB conventions (+X=forward, +Y=up, +Z=right)
xLCS = [0 0 1];     %+X Global is in +Z ISB
yLCS = [1 0 0];     %+Y Global is in +X ISB
zLCS = [0 1 0];     %+Z Global is in +Y ISB

GTL = [xLCS' yLCS' zLCS'];

%%%Rotate points to ISB conventions
chest_c1_1=(GTL*chest_c1_1')'; %JK: not using chest cluster
chest_c1_2=(GTL*chest_c1_2')'; 
chest_c1_5=(GTL*chest_c1_5')';

T8_c1_1=(GTL*T8_c1_1')';
T8_c1_2=(GTL*T8_c1_2')';
%T8_c1_3=(GTL*T8_c1_3')'; %JK: not going to use T8_3
T8_c1_4=(GTL*T8_c1_4')';

L2_c1_1=(GTL*L2_c1_1')';
L2_c1_2=(GTL*L2_c1_2')';
L2_c1_3=(GTL*L2_c1_3')';

XP=(GTL*XP')';
SS=(GTL*SS')';
C7=(GTL*C7')';
T8=(GTL*T8')';
L5=(GTL*L5')';

Rt_hum_cl_1=(GTL*Rt_hum_cl_1')';
Rt_hum_cl_2=(GTL*Rt_hum_cl_2')';
Rt_hum_cl_3=(GTL*Rt_hum_cl_3')';
Rt_LE=(GTL*Rt_LE')';
Rt_ME=(GTL*Rt_ME')';
Rt_Acro=(GTL*Rt_Acro')';

Lft_hum_cl_1=(GTL*Lft_hum_cl_1')';
Lft_hum_cl_2=(GTL*Lft_hum_cl_2')';
Lft_hum_cl_3=(GTL*Lft_hum_cl_3')';
Lft_LE=(GTL*Lft_LE')';
Lft_ME=(GTL*Lft_ME')';
Lft_Acro=(GTL*Lft_Acro')';

Rt_FA_cl_1=(GTL*Rt_FA_cl_1')';
Rt_FA_cl_2=(GTL*Rt_FA_cl_2')';
Rt_FA_cl_3=(GTL*Rt_FA_cl_3')';
Rt_US=(GTL*Rt_US')';
Rt_RS=(GTL*Rt_RS')';

Lft_FA_cl_1=(GTL*Lft_FA_cl_1')';
Lft_FA_cl_2=(GTL*Lft_FA_cl_2')';
Lft_FA_cl_3=(GTL*Lft_FA_cl_3')';
Lft_US=(GTL*Lft_US')';
Lft_RS=(GTL*Lft_RS')';

% % % r_sc1 = (GTL*r_sc1')'; r_sc2 = (GTL*r_sc2')'; r_sc3 = (GTL*r_sc3')'; 
% % % % l_sc1 = (GTL*l_sc1')'; l_sc2 = (GTL*l_sc2')'; l_sc3 = (GTL*l_sc3')'; 
% % % r_uc1 = (GTL*r_uc1')'; r_uc2 = (GTL*r_uc2')'; r_uc3 = (GTL*r_uc3')'; 
% % % % l_uc1 = (GTL*l_uc1')'; l_uc2 = (GTL*l_uc2')'; l_uc3 = (GTL*l_uc3')'; 
% % % r_le = (GTL*r_le')'; r_me = (GTL*r_me')';
% % % % l_le = (GTL*l_le')'; l_me = (GTL*l_me')';

%~~~Define the Local Coordinate System of the Clusters during calibration trial 

%Chest 
chest_z_cl=(chest_c1_2-chest_c1_5)/norm(chest_c1_2-chest_c1_5); 
chest_temp_cl=(chest_c1_1-chest_c1_5)/norm(chest_c1_1-chest_c1_5); 
chest_x_cl=cross(chest_temp_cl, chest_z_cl)/norm(cross(chest_temp_cl, chest_z_cl));
chest_y_cl=cross(chest_z_cl, chest_x_cl)/norm(cross(chest_z_cl, chest_x_cl));

%T8 (3) #subbed T8_3 with T8_1 since T8_3 missing in few trials 
T8_z_cl=(T8_c1_1-T8_c1_2)/norm(T8_c1_1-T8_c1_2); 
T8_temp_cl=(T8_c1_4-T8_c1_2)/norm(T8_c1_4-T8_c1_2); 
T8_x_cl=cross( T8_z_cl,T8_temp_cl)/norm(cross(T8_z_cl,T8_temp_cl));
T8_y_cl=cross(T8_z_cl, T8_x_cl)/norm(cross(T8_z_cl, T8_x_cl));

%L2
L2_z_cl=(L2_c1_3-L2_c1_2)/norm(L2_c1_3-L2_c1_2); 
L2_temp_cl=(L2_c1_1-L2_c1_2)/norm(L2_c1_1-L2_c1_2); 
L2_x_cl=cross(L2_temp_cl, L2_z_cl)/norm(cross(L2_temp_cl, L2_z_cl));
L2_y_cl=cross(L2_z_cl, L2_x_cl)/norm(cross(L2_z_cl, L2_x_cl));

%Rt_hum 
Rt_y_hum_cl=(Rt_hum_cl_3-Rt_hum_cl_2)/norm(Rt_hum_cl_3-Rt_hum_cl_2); 
Rt_temp_hum_cl=(Rt_hum_cl_1-Rt_hum_cl_3)/norm(Rt_hum_cl_1-Rt_hum_cl_3);
Rt_x_hum_cl=cross(Rt_y_hum_cl, Rt_temp_hum_cl)/norm(cross(Rt_y_hum_cl, Rt_temp_hum_cl));
Rt_z_hum_cl=cross(Rt_x_hum_cl,Rt_y_hum_cl)/norm(cross(Rt_x_hum_cl,Rt_y_hum_cl)); 

%Rt_FA
Rt_y_FA_cl=(Rt_FA_cl_3-Rt_FA_cl_2)/norm(Rt_FA_cl_3-Rt_FA_cl_2); 
Rt_temp_FA_cl=(Rt_FA_cl_1-Rt_FA_cl_3)/norm(Rt_FA_cl_1-Rt_FA_cl_3);
Rt_x_FA_cl=cross(Rt_y_FA_cl, Rt_temp_FA_cl)/norm(cross(Rt_y_FA_cl, Rt_temp_FA_cl));
Rt_z_FA_cl=cross(Rt_x_FA_cl,Rt_y_FA_cl)/norm(cross(Rt_x_FA_cl,Rt_y_FA_cl));  

%Lft_hum 
Lft_y_hum_cl=(Lft_hum_cl_3-Lft_hum_cl_2)/norm(Lft_hum_cl_3-Lft_hum_cl_2); 
Lft_temp_hum_cl=(Lft_hum_cl_1-Lft_hum_cl_3)/norm(Lft_hum_cl_1-Lft_hum_cl_3);
Lft_x_hum_cl=cross(Lft_y_hum_cl, Lft_temp_hum_cl)/norm(cross(Lft_y_hum_cl, Lft_temp_hum_cl));
Lft_z_hum_cl=cross(Lft_x_hum_cl,Lft_y_hum_cl)/norm(cross(Lft_x_hum_cl,Lft_y_hum_cl)); 

%Lft_FA
Lft_y_FA_cl=(Lft_FA_cl_3-Lft_FA_cl_2)/norm(Lft_FA_cl_3-Lft_FA_cl_2); 
Lft_temp_FA_cl=(Lft_FA_cl_1-Lft_FA_cl_3)/norm(Lft_FA_cl_1-Lft_FA_cl_3);
Lft_x_FA_cl=cross(Lft_y_FA_cl, Lft_temp_FA_cl)/norm(cross(Lft_y_FA_cl, Lft_temp_FA_cl));
Lft_z_FA_cl=cross(Lft_x_FA_cl,Lft_y_FA_cl)/norm(cross(Lft_x_FA_cl,Lft_y_FA_cl));


xglobal = [1 0 0];
yglobal = [0 1 0];
zglobal = [0 0 1];

%Rotation matrix to go from Global Coordinates to Rt Hum Coordinates
R_global_RtHum(1,1) = dot(Rt_x_hum_cl,xglobal);
R_global_RtHum(1,2) = dot(Rt_x_hum_cl,yglobal);
R_global_RtHum(1,3) = dot(Rt_x_hum_cl,zglobal);
R_global_RtHum(2,1) = dot(Rt_y_hum_cl,xglobal);
R_global_RtHum(2,2) = dot(Rt_y_hum_cl,yglobal);
R_global_RtHum(2,3) = dot(Rt_y_hum_cl,zglobal);
R_global_RtHum(3,1) = dot(Rt_z_hum_cl,xglobal);
R_global_RtHum(3,2) = dot(Rt_z_hum_cl,yglobal);
R_global_RtHum(3,3) = dot(Rt_z_hum_cl,zglobal);

%Rotation matrix to go from Global Coordinates to Rt FA Coordinates
R_global_RtFA(1,1) = dot(Rt_x_FA_cl,xglobal);
R_global_RtFA(1,2) = dot(Rt_x_FA_cl,yglobal);
R_global_RtFA(1,3) = dot(Rt_x_FA_cl,zglobal);
R_global_RtFA(2,1) = dot(Rt_y_FA_cl,xglobal);
R_global_RtFA(2,2) = dot(Rt_y_FA_cl,yglobal);
R_global_RtFA(2,3) = dot(Rt_y_FA_cl,zglobal);
R_global_RtFA(3,1) = dot(Rt_z_FA_cl,xglobal);
R_global_RtFA(3,2) = dot(Rt_z_FA_cl,yglobal);
R_global_RtFA(3,3) = dot(Rt_z_FA_cl,zglobal);

%Rotation matrix to go from Global Coordinates to Lft Hum Coordinates
R_global_LftHum(1,1) = dot(Lft_x_hum_cl,xglobal);
R_global_LftHum(1,2) = dot(Lft_x_hum_cl,yglobal);
R_global_LftHum(1,3) = dot(Lft_x_hum_cl,zglobal);
R_global_LftHum(2,1) = dot(Lft_y_hum_cl,xglobal);
R_global_LftHum(2,2) = dot(Lft_y_hum_cl,yglobal);
R_global_LftHum(2,3) = dot(Lft_y_hum_cl,zglobal);
R_global_LftHum(3,1) = dot(Lft_z_hum_cl,xglobal);
R_global_LftHum(3,2) = dot(Lft_z_hum_cl,yglobal);
R_global_LftHum(3,3) = dot(Lft_z_hum_cl,zglobal);

%Rotation matrix to go from Global Coordinates to Lft FA Coordinates
R_global_LftFA(1,1) = dot(Lft_x_FA_cl,xglobal);
R_global_LftFA(1,2) = dot(Lft_x_FA_cl,yglobal);
R_global_LftFA(1,3) = dot(Lft_x_FA_cl,zglobal);
R_global_LftFA(2,1) = dot(Lft_y_FA_cl,xglobal);
R_global_LftFA(2,2) = dot(Lft_y_FA_cl,yglobal);
R_global_LftFA(2,3) = dot(Lft_y_FA_cl,zglobal);
R_global_LftFA(3,1) = dot(Lft_z_FA_cl,xglobal);
R_global_LftFA(3,2) = dot(Lft_z_FA_cl,yglobal);
R_global_LftFA(3,3) = dot(Lft_z_FA_cl,zglobal);

%Rotation matrix to go from Global Coordinates to Chest Coordinates 
R_global_chest(1,1) = dot(chest_x_cl,xglobal);
R_global_chest(1,2) = dot(chest_x_cl,yglobal);
R_global_chest(1,3) = dot(chest_x_cl,zglobal);
R_global_chest(2,1) = dot(chest_y_cl,xglobal);
R_global_chest(2,2) = dot(chest_y_cl,yglobal);
R_global_chest(2,3) = dot(chest_y_cl,zglobal);
R_global_chest(3,1) = dot(chest_z_cl,xglobal);
R_global_chest(3,2) = dot(chest_z_cl,yglobal);
R_global_chest(3,3) = dot(chest_z_cl,zglobal);

%Rotation matrix to go from Global Coordinates to L2 Coordinates
R_global_L2(1,1) = dot(L2_x_cl,xglobal);
R_global_L2(1,2) = dot(L2_x_cl,yglobal);
R_global_L2(1,3) = dot(L2_x_cl,zglobal);
R_global_L2(2,1) = dot(L2_y_cl,xglobal);
R_global_L2(2,2) = dot(L2_y_cl,yglobal);
R_global_L2(2,3) = dot(L2_y_cl,zglobal);
R_global_L2(3,1) = dot(L2_z_cl,xglobal);
R_global_L2(3,2) = dot(L2_z_cl,yglobal);
R_global_L2(3,3) = dot(L2_z_cl,zglobal);

%Rotation matrix to go from Global Coordinates to T8 Coordinates
R_global_T8(1,1) = dot(T8_x_cl,xglobal);
R_global_T8(1,2) = dot(T8_x_cl,yglobal);
R_global_T8(1,3) = dot(T8_x_cl,zglobal);
R_global_T8(2,1) = dot(T8_y_cl,xglobal);
R_global_T8(2,2) = dot(T8_y_cl,yglobal);
R_global_T8(2,3) = dot(T8_y_cl,zglobal);
R_global_T8(3,1) = dot(T8_z_cl,xglobal);
R_global_T8(3,2) = dot(T8_z_cl,yglobal);
R_global_T8(3,3) = dot(T8_z_cl,zglobal);

%Find the relationship between the Rt Hum Cluster and ME, LE
RtLE_RtHum = R_global_RtHum*(Rt_LE-Rt_hum_cl_1)';
RtME_RtHum = R_global_RtHum*(Rt_ME-Rt_hum_cl_1)';
%RtAcro_RtHum = R_global_RtHum*(Rt_Acro-Rt_hum_cl_1)'; %JK: using actual
%marker for this

%Find the relationship between the Lft Hum Cluster and ME, LE
LftLE_LftHum = R_global_LftHum*(Lft_LE-Lft_hum_cl_1)';
LftME_LftHum = R_global_LftHum*(Lft_ME-Lft_hum_cl_1)';
%LftAcro_LftHum = R_global_LftHum*(Lft_Acro-Lft_hum_cl_1)';

%Find the relationship between the Rt FA Cluster and US, RS
RtUS_RtFA = R_global_RtFA*(Rt_US-Rt_FA_cl_1)';
RtRS_RtFA = R_global_RtFA*(Rt_RS-Rt_FA_cl_1)';

%Find the relationship between the Lft FA Cluster and US, RS
LftUS_LftFA = R_global_LftFA*(Lft_US-Lft_FA_cl_1)';
LftRS_LftFA = R_global_LftFA*(Lft_RS-Lft_FA_cl_1)';

%~~Find the relationship between the Chest Cluster and SS, XP, L5, T8, C7
XP_chest = R_global_chest*(XP-chest_c1_1)';
SS_chest = R_global_chest*(SS-chest_c1_1)';
C7_chest = R_global_chest*(C7-chest_c1_1)';
T8_chest = R_global_chest*(T8-chest_c1_1)';
L5_chest = R_global_chest*(L5-chest_c1_1)';

%~~Find the relationship between the L2 Cluster and L5 - JK edited to only L5
XP_L2 = R_global_L2*(XP-L2_c1_1)';
SS_L2 = R_global_L2*(SS-L2_c1_1)';
C7_L2 = R_global_L2*(C7-L2_c1_1)';
T8_L2 = R_global_L2*(T8-L2_c1_1)';
L5_L2 = R_global_L2*(L5-L2_c1_1)';

%~~Find the relationship between the T8 Cluster and SS, XP, T8, C7 *JK:
%removed L5 from this
XP_T8 = R_global_T8*(XP-T8_c1_2)';
SS_T8 = R_global_T8*(SS-T8_c1_2)';
C7_T8 = R_global_T8*(C7-T8_c1_2)';
T8_T8 = R_global_T8*(T8-T8_c1_2)';
L5_T8 = R_global_T8*(L5-T8_c1_2)';


%****************Print markers to check locations****************%
scatter3(SS(1,1),SS(1,2),SS(1,3),'m','filled')
text(SS(1,1),SS(1,2),SS(1,3),'   SS')
hold on 
scatter3(XP(1,1),XP(1,2),XP(1,3),'m','filled')
text(XP(1,1),XP(1,2),XP(1,3),'   XP')
scatter3(L5(1,1),L5(1,2),L5(1,3),'m','filled')
text(L5(1,1),L5(1,2),L5(1,3),'   L5') 
scatter3(C7(1,1),C7(1,2),C7(1,3),'m','filled')
text(C7(1,1),C7(1,2),C7(1,3),'   C7')
scatter3(T8(1,1),T8(1,2),T8(1,3),'m','filled')
text(T8(1,1),T8(1,2),T8(1,3),'   T8')

scatter3(Lft_US(1,1),Lft_US(1,2),Lft_US(1,3),'c','filled')
scatter3(Lft_RS(1,1),Lft_RS(1,2),Lft_RS(1,3),'c','filled')
scatter3(Lft_ME(1,1),Lft_ME(1,2),Lft_ME(1,3),'c','filled')
scatter3(Lft_LE(1,1),Lft_LE(1,2),Lft_LE(1,3),'c','filled')
scatter3(Lft_Acro(1,1),Lft_Acro(1,2),Lft_Acro(1,3),'c','filled')

scatter3(Lft_FA_cl_1(1,1),Lft_FA_cl_1(1,2),Lft_FA_cl_1(1,3),'c','filled')
scatter3(Lft_FA_cl_2(1,1),Lft_FA_cl_2(1,2),Lft_FA_cl_2(1,3),'c','filled')
scatter3(Lft_FA_cl_3(1,1),Lft_FA_cl_3(1,2),Lft_FA_cl_3(1,3),'c','filled')
scatter3(Lft_hum_cl_1(1,1),Lft_hum_cl_1(1,2),Lft_hum_cl_1(1,3),'c','filled')
scatter3(Lft_hum_cl_2(1,1),Lft_hum_cl_2(1,2),Lft_hum_cl_2(1,3),'c','filled')
scatter3(Lft_hum_cl_3(1,1),Lft_hum_cl_3(1,2),Lft_hum_cl_3(1,3),'c','filled')

scatter3(Rt_US(1,1),Rt_US(1,2),Rt_US(1,3),'k','filled')
scatter3(Rt_RS(1,1),Rt_RS(1,2),Rt_RS(1,3),'k','filled')
scatter3(Rt_ME(1,1),Rt_ME(1,2),Rt_ME(1,3),'k','filled')
scatter3(Rt_LE(1,1),Rt_LE(1,2),Rt_LE(1,3),'k','filled')
scatter3(Rt_Acro(1,1),Rt_Acro(1,2),Rt_Acro(1,3),'k','filled')

scatter3(Rt_FA_cl_1(1,1),Rt_FA_cl_1(1,2),Rt_FA_cl_1(1,3),'k','filled')
scatter3(Rt_FA_cl_2(1,1),Rt_FA_cl_2(1,2),Rt_FA_cl_2(1,3),'k','filled')
scatter3(Rt_FA_cl_3(1,1),Rt_FA_cl_3(1,2),Rt_FA_cl_3(1,3),'k','filled')
scatter3(Rt_hum_cl_1(1,1),Rt_hum_cl_1(1,2),Rt_hum_cl_1(1,3),'k','filled')
scatter3(Rt_hum_cl_2(1,1),Rt_hum_cl_2(1,2),Rt_hum_cl_2(1,3),'k','filled')
scatter3(Rt_hum_cl_3(1,1),Rt_hum_cl_3(1,2),Rt_hum_cl_3(1,3),'k','filled')

scatter3(T8_c1_1(1,1),T8_c1_1(1,2),T8_c1_1(1,3),'r','filled')
scatter3(T8_c1_2(1,1),T8_c1_2(1,2),T8_c1_2(1,3),'r','filled')
scatter3(T8_c1_3(1,1),T8_c1_3(1,2),T8_c1_3(1,3),'r','filled')

scatter3(L2_c1_1(1,1),L2_c1_1(1,2),L2_c1_1(1,3),'r','filled')
scatter3(L2_c1_2(1,1),L2_c1_2(1,2),L2_c1_2(1,3),'r','filled')
scatter3(L2_c1_3(1,1),L2_c1_3(1,2),L2_c1_3(1,3),'r','filled')

%scatter3(chest_c1_1(1,1),chest_c1_1(1,2),chest_c1_1(1,3),'r','filled')
%scatter3(chest_c1_2(1,1),chest_c1_2(1,2),chest_c1_2(1,3),'r','filled')
%scatter3(chest_c1_5(1,1),chest_c1_5(1,2),chest_c1_5(1,3),'r','filled')

line([XP(1,1),SS(1,1)],[XP(1,2),SS(1,2)],[XP(1,3),SS(1,3)],'Color','black')
line([C7(1,1),SS(1,1)],[C7(1,2),SS(1,2)],[C7(1,3),SS(1,3)],'Color','black')
line([XP(1,1),L5(1,1)],[XP(1,2),L5(1,2)],[XP(1,3),L5(1,3)],'Color','black')
line([XP(1,1),L5(1,1)],[XP(1,2),L5(1,2)],[XP(1,3),L5(1,3)],'Color','black')
line([C7(1,1),L5(1,1)],[C7(1,2),L5(1,2)],[C7(1,3),L5(1,3)],'Color','black')
line([Rt_Acro(1,1),C7(1,1)],[Rt_Acro(1,2),C7(1,2)],[Rt_Acro(1,3),C7(1,3)],'Color','black')
line([Rt_Acro(1,1),SS(1,1)],[Rt_Acro(1,2),SS(1,2)],[Rt_Acro(1,3),SS(1,3)],'Color','black')
line([Rt_Acro(1,1),XP(1,1)],[Rt_Acro(1,2),XP(1,2)],[Rt_Acro(1,3),XP(1,3)],'Color','black')
line([Rt_Acro(1,1),L5(1,1)],[Rt_Acro(1,2),L5(1,2)],[Rt_Acro(1,3),L5(1,3)],'Color','black')
line([Lft_Acro(1,1),C7(1,1)],[Lft_Acro(1,2),C7(1,2)],[Lft_Acro(1,3),C7(1,3)],'Color','black')
line([Lft_Acro(1,1),SS(1,1)],[Lft_Acro(1,2),SS(1,2)],[Lft_Acro(1,3),SS(1,3)],'Color','black')
line([Lft_Acro(1,1),XP(1,1)],[Lft_Acro(1,2),XP(1,2)],[Lft_Acro(1,3),XP(1,3)],'Color','black')
line([Lft_Acro(1,1),L5(1,1)],[Lft_Acro(1,2),L5(1,2)],[Lft_Acro(1,3),L5(1,3)],'Color','black')
line([Rt_Acro(1,1),Rt_ME(1,1)],[Rt_Acro(1,2),Rt_ME(1,2)],[Rt_Acro(1,3),Rt_ME(1,3)],'Color','black')
line([Rt_Acro(1,1),Rt_LE(1,1)],[Rt_Acro(1,2),Rt_LE(1,2)],[Rt_Acro(1,3),Rt_LE(1,3)],'Color','black')
line([Rt_ME(1,1),Rt_LE(1,1)],[Rt_ME(1,2),Rt_LE(1,2)],[Rt_ME(1,3),Rt_LE(1,3)],'Color','black')
line([Lft_Acro(1,1),Lft_ME(1,1)],[Lft_Acro(1,2),Lft_ME(1,2)],[Lft_Acro(1,3),Lft_ME(1,3)],'Color','black')
line([Lft_Acro(1,1),Lft_LE(1,1)],[Lft_Acro(1,2),Lft_LE(1,2)],[Lft_Acro(1,3),Lft_LE(1,3)],'Color','black')
line([Lft_ME(1,1),Lft_LE(1,1)],[Lft_ME(1,2),Lft_LE(1,2)],[Lft_ME(1,3),Lft_LE(1,3)],'Color','black')
line([Rt_RS(1,1),Rt_US(1,1)],[Rt_RS(1,2),Rt_US(1,2)],[Rt_RS(1,3),Rt_US(1,3)],'Color','black')
line([Rt_RS(1,1),Rt_LE(1,1)],[Rt_RS(1,2),Rt_LE(1,2)],[Rt_RS(1,3),Rt_LE(1,3)],'Color','black')
line([Rt_RS(1,1),Rt_ME(1,1)],[Rt_RS(1,2),Rt_ME(1,2)],[Rt_RS(1,3),Rt_ME(1,3)],'Color','black')
line([Rt_US(1,1),Rt_ME(1,1)],[Rt_US(1,2),Rt_ME(1,2)],[Rt_US(1,3),Rt_ME(1,3)],'Color','black')
line([Rt_US(1,1),Rt_LE(1,1)],[Rt_US(1,2),Rt_LE(1,2)],[Rt_US(1,3),Rt_LE(1,3)],'Color','black')
line([Lft_RS(1,1),Lft_US(1,1)],[Lft_RS(1,2),Lft_US(1,2)],[Lft_RS(1,3),Lft_US(1,3)],'Color','black')
line([Lft_RS(1,1),Lft_LE(1,1)],[Lft_RS(1,2),Lft_LE(1,2)],[Lft_RS(1,3),Lft_LE(1,3)],'Color','black')
line([Lft_RS(1,1),Lft_ME(1,1)],[Lft_RS(1,2),Lft_ME(1,2)],[Lft_RS(1,3),Lft_ME(1,3)],'Color','black')
line([Lft_US(1,1),Lft_ME(1,1)],[Lft_US(1,2),Lft_ME(1,2)],[Lft_US(1,3),Lft_ME(1,3)],'Color','black')
line([Lft_US(1,1),Lft_LE(1,1)],[Lft_US(1,2),Lft_LE(1,2)],[Lft_US(1,3),Lft_LE(1,3)],'Color','black')

pause 
hold off 
%-------------------------------------------------------------------------    
%Process trial files 
%-------------------------------------------------------------------------   
%Define clusters only **comment out 2/3 torso clusters JK: this will change
%Rotate clusters into ISB 
xglobal = [1 0 0];
yglobal = [0 1 0];
zglobal = [0 0 1];

CarryFrame=input('Input Frame for Trial:');

%if cluster == 1 %%%JK: commented out, using same clusters for all trials
   
    chest_c1_1=carry_data(CarryFrame,25:27); 
    chest_c1_2=carry_data(CarryFrame,28:30);
    chest_c1_5=carry_data(CarryFrame,37:39);

    chest_c1_1=(GTL*chest_c1_1')';
    chest_c1_2=(GTL*chest_c1_2')';
    chest_c1_5=(GTL*chest_c1_5')';

    chest_z_cl=(chest_c1_2-chest_c1_5)/norm(chest_c1_2-chest_c1_5); 
    chest_temp_cl=(chest_c1_1-chest_c1_5)/norm(chest_c1_1-chest_c1_5); 
    chest_x_cl=cross(chest_temp_cl, chest_z_cl)/norm(cross(chest_temp_cl, chest_z_cl));
    chest_y_cl=cross(chest_z_cl, chest_x_cl)/norm(cross(chest_z_cl, chest_x_cl));

    %Rotation matrix to go from Global Coordinates to Chest Coordinates
    R_global_chest(1,1) = dot(chest_x_cl,xglobal);
    R_global_chest(1,2) = dot(chest_x_cl,yglobal);
    R_global_chest(1,3) = dot(chest_x_cl,zglobal);
    R_global_chest(2,1) = dot(chest_y_cl,xglobal);
    R_global_chest(2,2) = dot(chest_y_cl,yglobal);
    R_global_chest(2,3) = dot(chest_y_cl,zglobal);
    R_global_chest(3,1) = dot(chest_z_cl,xglobal);
    R_global_chest(3,2) = dot(chest_z_cl,yglobal);
    R_global_chest(3,3) = dot(chest_z_cl,zglobal);

%elseif cluster == 2
     
    T8_c1_1=carry_data(CarryFrame,40:42); 
    T8_c1_2=carry_data(CarryFrame,43:45);
    %T8_c1_3=carry_data(CarryFrame,46:48);
    T8_c1_4=carry_data(CarryFrame,49:51);

    T8_c1_1=(GTL*T8_c1_1')';
    T8_c1_2=(GTL*T8_c1_2')';
    %T8_c1_3=(GTL*T8_c1_3')';
    T8_c1_4=(GTL*T8_c1_4')';

    T8_z_cl=(T8_c1_1-T8_c1_2)/norm(T8_c1_1-T8_c1_2); 
    T8_temp_cl=(T8_c1_4-T8_c1_2)/norm(T8_c1_4-T8_c1_2); 
    T8_x_cl=cross( T8_z_cl,T8_temp_cl)/norm(cross(T8_z_cl,T8_temp_cl));
    T8_y_cl=cross(T8_z_cl, T8_x_cl)/norm(cross(T8_z_cl, T8_x_cl));

    %Rotation matrix to go from Global Coordinates to T8 Coordinates
    R_global_T8(1,1) = dot(T8_x_cl,xglobal);
    R_global_T8(1,2) = dot(T8_x_cl,yglobal);
    R_global_T8(1,3) = dot(T8_x_cl,zglobal);
    R_global_T8(2,1) = dot(T8_y_cl,xglobal);
    R_global_T8(2,2) = dot(T8_y_cl,yglobal);
    R_global_T8(2,3) = dot(T8_y_cl,zglobal);
    R_global_T8(3,1) = dot(T8_z_cl,xglobal);
    R_global_T8(3,2) = dot(T8_z_cl,yglobal);
    R_global_T8(3,3) = dot(T8_z_cl,zglobal);

%elseif cluster == 3
     
 L2_c1_1=carry_data(CarryFrame,52:54); 
 L2_c1_2=carry_data(CarryFrame,55:57);
 L2_c1_3=carry_data(CarryFrame,58:60);

 L2_c1_1=(GTL*L2_c1_1')';
 L2_c1_2=(GTL*L2_c1_2')';
 L2_c1_3=(GTL*L2_c1_3')';

 L2_z_cl=(L2_c1_3-L2_c1_2)/norm(L2_c1_3-L2_c1_2); 
 L2_temp_cl=(L2_c1_1-L2_c1_2)/norm(L2_c1_1-L2_c1_2); 
 L2_x_cl=cross(L2_temp_cl, L2_z_cl)/norm(cross(L2_temp_cl, L2_z_cl));
 L2_y_cl=cross(L2_z_cl, L2_x_cl)/norm(cross(L2_z_cl, L2_x_cl));

    %Rotation matrix to go from Global Coordinates to L2 Coordinates
    R_global_L2(1,1) = dot(L2_x_cl,xglobal);
    R_global_L2(1,2) = dot(L2_x_cl,yglobal);
    R_global_L2(1,3) = dot(L2_x_cl,zglobal);
    R_global_L2(2,1) = dot(L2_y_cl,xglobal);
    R_global_L2(2,2) = dot(L2_y_cl,yglobal);
    R_global_L2(2,3) = dot(L2_y_cl,zglobal);
    R_global_L2(3,1) = dot(L2_z_cl,xglobal);
    R_global_L2(3,2) = dot(L2_z_cl,yglobal);
    R_global_L2(3,3) = dot(L2_z_cl,zglobal);
%end JK: commented out, using all clusters except chest for all
%participants

Rt_hum_cl_1=carry_data(CarryFrame,67:69);
Rt_hum_cl_2=carry_data(CarryFrame,70:72);
Rt_hum_cl_3=carry_data(CarryFrame,73:75);
 
Rt_hum_cl_1=(GTL*Rt_hum_cl_1')';
Rt_hum_cl_2=(GTL*Rt_hum_cl_2')';
Rt_hum_cl_3=(GTL*Rt_hum_cl_3')';

Rt_y_hum_cl=(Rt_hum_cl_3-Rt_hum_cl_2)/norm(Rt_hum_cl_3-Rt_hum_cl_2); 
Rt_temp_hum_cl=(Rt_hum_cl_1-Rt_hum_cl_3)/norm(Rt_hum_cl_1-Rt_hum_cl_3);
Rt_x_hum_cl=cross(Rt_y_hum_cl, Rt_temp_hum_cl)/norm(cross(Rt_y_hum_cl, Rt_temp_hum_cl));
Rt_z_hum_cl=cross(Rt_x_hum_cl,Rt_y_hum_cl)/norm(cross(Rt_x_hum_cl,Rt_y_hum_cl)); 

%Rotation matrix to go from Global Coordinates to Rt Hum Coordinates
R_global_RtHum(1,1) = dot(Rt_x_hum_cl,xglobal);
R_global_RtHum(1,2) = dot(Rt_x_hum_cl,yglobal);
R_global_RtHum(1,3) = dot(Rt_x_hum_cl,zglobal);
R_global_RtHum(2,1) = dot(Rt_y_hum_cl,xglobal);
R_global_RtHum(2,2) = dot(Rt_y_hum_cl,yglobal);
R_global_RtHum(2,3) = dot(Rt_y_hum_cl,zglobal);
R_global_RtHum(3,1) = dot(Rt_z_hum_cl,xglobal);
R_global_RtHum(3,2) = dot(Rt_z_hum_cl,yglobal);
R_global_RtHum(3,3) = dot(Rt_z_hum_cl,zglobal);

Lft_hum_cl_1=carry_data(CarryFrame,103:105);
Lft_hum_cl_2=carry_data(CarryFrame,106:108);
Lft_hum_cl_3=carry_data(CarryFrame,109:111);

Lft_hum_cl_1=(GTL*Lft_hum_cl_1')';
Lft_hum_cl_2=(GTL*Lft_hum_cl_2')';
Lft_hum_cl_3=(GTL*Lft_hum_cl_3')';

Lft_y_hum_cl=(Lft_hum_cl_3-Lft_hum_cl_2)/norm(Lft_hum_cl_3-Lft_hum_cl_2); 
Lft_temp_hum_cl=(Lft_hum_cl_1-Lft_hum_cl_3)/norm(Lft_hum_cl_1-Lft_hum_cl_3);
Lft_x_hum_cl=cross(Lft_y_hum_cl, Lft_temp_hum_cl)/norm(cross(Lft_y_hum_cl, Lft_temp_hum_cl));
Lft_z_hum_cl=cross(Lft_x_hum_cl,Lft_y_hum_cl)/norm(cross(Lft_x_hum_cl,Lft_y_hum_cl)); 

%Rotation matrix to go from Global Coordinates to Lft Hum Coordinates
R_global_LftHum(1,1) = dot(Lft_x_hum_cl,xglobal);
R_global_LftHum(1,2) = dot(Lft_x_hum_cl,yglobal);
R_global_LftHum(1,3) = dot(Lft_x_hum_cl,zglobal);
R_global_LftHum(2,1) = dot(Lft_y_hum_cl,xglobal);
R_global_LftHum(2,2) = dot(Lft_y_hum_cl,yglobal);
R_global_LftHum(2,3) = dot(Lft_y_hum_cl,zglobal);
R_global_LftHum(3,1) = dot(Lft_z_hum_cl,xglobal);
R_global_LftHum(3,2) = dot(Lft_z_hum_cl,yglobal);
R_global_LftHum(3,3) = dot(Lft_z_hum_cl,zglobal);

Rt_FA_cl_1=carry_data(CarryFrame,82:84);
Rt_FA_cl_2=carry_data(CarryFrame,85:87);
Rt_FA_cl_3=carry_data(CarryFrame,88:90);

Rt_FA_cl_1=(GTL*Rt_FA_cl_1')';
Rt_FA_cl_2=(GTL*Rt_FA_cl_2')';
Rt_FA_cl_3=(GTL*Rt_FA_cl_3')';

Rt_y_FA_cl=(Rt_FA_cl_3-Rt_FA_cl_2)/norm(Rt_FA_cl_3-Rt_FA_cl_2); 
Rt_temp_FA_cl=(Rt_FA_cl_1-Rt_FA_cl_3)/norm(Rt_FA_cl_1-Rt_FA_cl_3);
Rt_x_FA_cl=cross(Rt_y_FA_cl, Rt_temp_FA_cl)/norm(cross(Rt_y_FA_cl, Rt_temp_FA_cl));
Rt_z_FA_cl=cross(Rt_x_FA_cl,Rt_y_FA_cl)/norm(cross(Rt_x_FA_cl,Rt_y_FA_cl));  

%Rotation matrix to go from Global Coordinates to Rt FA Coordinates
R_global_RtFA(1,1) = dot(Rt_x_FA_cl,xglobal);
R_global_RtFA(1,2) = dot(Rt_x_FA_cl,yglobal);
R_global_RtFA(1,3) = dot(Rt_x_FA_cl,zglobal);
R_global_RtFA(2,1) = dot(Rt_y_FA_cl,xglobal);
R_global_RtFA(2,2) = dot(Rt_y_FA_cl,yglobal);
R_global_RtFA(2,3) = dot(Rt_y_FA_cl,zglobal);
R_global_RtFA(3,1) = dot(Rt_z_FA_cl,xglobal);
R_global_RtFA(3,2) = dot(Rt_z_FA_cl,yglobal);
R_global_RtFA(3,3) = dot(Rt_z_FA_cl,zglobal);

Lft_FA_cl_1=carry_data(CarryFrame,118:120);
Lft_FA_cl_2=carry_data(CarryFrame,121:123);
Lft_FA_cl_3=carry_data(CarryFrame,124:126);

Lft_FA_cl_1=(GTL*Lft_FA_cl_1')';
Lft_FA_cl_2=(GTL*Lft_FA_cl_2')';
Lft_FA_cl_3=(GTL*Lft_FA_cl_3')';

Lft_y_FA_cl=(Lft_FA_cl_3-Lft_FA_cl_2)/norm(Lft_FA_cl_3-Lft_FA_cl_2); 
Lft_temp_FA_cl=(Lft_FA_cl_1-Lft_FA_cl_3)/norm(Lft_FA_cl_1-Lft_FA_cl_3);
Lft_x_FA_cl=cross(Lft_y_FA_cl, Lft_temp_FA_cl)/norm(cross(Lft_y_FA_cl, Lft_temp_FA_cl));
Lft_z_FA_cl=cross(Lft_x_FA_cl,Lft_y_FA_cl)/norm(cross(Lft_x_FA_cl,Lft_y_FA_cl));

%Rotation matrix to go from Global Coordinates to Lft FA Coordinates
R_global_LftFA(1,1) = dot(Lft_x_FA_cl,xglobal);
R_global_LftFA(1,2) = dot(Lft_x_FA_cl,yglobal);
R_global_LftFA(1,3) = dot(Lft_x_FA_cl,zglobal);
R_global_LftFA(2,1) = dot(Lft_y_FA_cl,xglobal);
R_global_LftFA(2,2) = dot(Lft_y_FA_cl,yglobal);
R_global_LftFA(2,3) = dot(Lft_y_FA_cl,zglobal);
R_global_LftFA(3,1) = dot(Lft_z_FA_cl,xglobal);
R_global_LftFA(3,2) = dot(Lft_z_FA_cl,yglobal);
R_global_LftFA(3,3) = dot(Lft_z_FA_cl,zglobal);


% Create the virtual markers from relationships 
%% Select one cluster set and comment out the rest 
% 
%T8 cluster JK: using T8 for virtual XP, SS, C7, T8

%if cluster == 2 
%     Virtual_XP=(T8_c1_2' + R_global_T8*XP_T8)'; 
%     Virtual_SS=(T8_c1_2' + R_global_T8*SS_T8)'; 
%     Virtual_C7=(T8_c1_2' + R_global_T8*C7_T8)'; 
%     Virtual_T8=(T8_c1_2' + R_global_T8*T8_T8)'; 
%     Virtual_L5=(T8_c1_2' + R_global_T8*L5_T8)'; %JK

%elseif cluster == 3 JK: using L2 to recreate L5 marker for all
%L2 cluster   
    Virtual_XP=(L2_c1_1' + R_global_L2*XP_L2)'; 
    Virtual_SS=(L2_c1_1' + R_global_L2*SS_L2)'; 
    Virtual_C7=(L2_c1_1' + R_global_L2*C7_L2)'; 
    Virtual_T8=(L2_c1_1' + R_global_L2*T8_L2)'; 
    Virtual_L5=(L2_c1_1' + R_global_L2*L5_L2)';

%elseif cluster ==1 
% chest cluster JK: NOT USING. commented out
%     Virtual_XP=(chest_c1_1' + R_global_chest*XP_chest)'; 
%     Virtual_SS=(chest_c1_1' + R_global_chest*SS_chest)'; 
%     Virtual_C7=(chest_c1_1' + R_global_chest*C7_chest)'; 
%     Virtual_T8=(chest_c1_1' + R_global_chest*T8_chest)'; 
%     Virtual_L5=(chest_c1_1' + R_global_chest*L5_chest)';
%end JK

%Rt Hum cluster JK: NO virtual ACR. 
Virtual_RtLE=(Rt_hum_cl_1' + R_global_RtHum*RtLE_RtHum)'; 
Virtual_RtME=(Rt_hum_cl_1' + R_global_RtHum*RtME_RtHum)'; 
%R_ACR=(Rt_hum_cl_1' + R_global_RtHum*RtAcro_RtHum)';

%Lft Hum cluster JK: NO virutal ACR.
Virtual_LftLE=(Lft_hum_cl_1' + R_global_LftHum*LftLE_LftHum)'; 
Virtual_LftME=(Lft_hum_cl_1' + R_global_LftHum*LftME_LftHum)'; 
%L_ACR=(Lft_hum_cl_1' + R_global_LftHum*LftAcro_LftHum)'; 

%Rt FA cluster 
Virtual_RtRS=(Rt_FA_cl_1' + R_global_RtFA*RtRS_RtFA)'; 
Virtual_RtUS=(Rt_FA_cl_1' + R_global_RtFA*RtUS_RtFA)'; 

%JK added: Import REAL acromion markers
R_ACR=carry_data(CarryFrame,19:21);
L_ACR=carry_data(CarryFrame,22:24);

R_ACR=(GTL*R_ACR')';
L_ACR=(GTL*L_ACR')';

%Lft FA cluster 
Virtual_LftRS=(Lft_FA_cl_1' + R_global_LftFA*LftRS_LftFA)'; 
Virtual_LftUS=(Lft_FA_cl_1' + R_global_LftFA*LftUS_LftFA)'; 

%%Plot virtual markers
scatter3(Virtual_SS(1,1),Virtual_SS(1,2),Virtual_SS(1,3),'m','filled')
text(Virtual_SS(1,1),Virtual_SS(1,2),Virtual_SS(1,3),'   SS')
hold on 
scatter3(Virtual_XP(1,1),Virtual_XP(1,2),Virtual_XP(1,3),'m','filled')
text(Virtual_XP(1,1),Virtual_XP(1,2),Virtual_XP(1,3),'   XP')
scatter3(Virtual_L5(1,1),Virtual_L5(1,2),Virtual_L5(1,3),'m','filled')
text(Virtual_L5(1,1),Virtual_L5(1,2),Virtual_L5(1,3),'   L5') 
scatter3(Virtual_C7(1,1),Virtual_C7(1,2),Virtual_C7(1,3),'m','filled')
text(Virtual_C7(1,1),Virtual_C7(1,2),Virtual_C7(1,3),'   C7')
scatter3(Virtual_T8(1,1),Virtual_T8(1,2),Virtual_T8(1,3),'m','filled')
text(Virtual_T8(1,1),Virtual_T8(1,2),Virtual_T8(1,3),'   T8')

scatter3(Virtual_LftUS(1,1),Virtual_LftUS(1,2),Virtual_LftUS(1,3),'c','filled')
text(Virtual_LftUS(1,1),Virtual_LftUS(1,2),Virtual_LftUS(1,3),'   LftUS')
scatter3(Virtual_LftRS(1,1),Virtual_LftRS(1,2),Virtual_LftRS(1,3),'c','filled')
text(Virtual_LftRS(1,1),Virtual_LftRS(1,2),Virtual_LftRS(1,3),'   LftRS')
scatter3(Virtual_LftME(1,1),Virtual_LftME(1,2),Virtual_LftME(1,3),'c','filled')
text(Virtual_LftME(1,1),Virtual_LftME(1,2),Virtual_LftME(1,3),'   LftME')
scatter3(Virtual_LftLE(1,1),Virtual_LftLE(1,2),Virtual_LftLE(1,3),'c','filled')
text(Virtual_LftLE(1,1),Virtual_LftLE(1,2),Virtual_LftLE(1,3),'   LftLE')
scatter3(L_ACR(1,1),L_ACR(1,2),L_ACR(1,3),'c','filled') %JK: changed to L_ACR
text(L_ACR(1,1),L_ACR(1,2),L_ACR(1,3),'  LftAcro')

scatter3(Virtual_RtUS(1,1),Virtual_RtUS(1,2),Virtual_RtUS(1,3),'r','filled')
text(Virtual_RtUS(1,1),Virtual_RtUS(1,2),Virtual_RtUS(1,3),'   RtUS')
scatter3(Virtual_RtRS(1,1),Virtual_RtRS(1,2),Virtual_RtRS(1,3),'r','filled')
text(Virtual_RtRS(1,1),Virtual_RtRS(1,2),Virtual_RtRS(1,3),'   RtRS')
scatter3(Virtual_RtME(1,1),Virtual_RtME(1,2),Virtual_RtME(1,3),'r','filled')
text(Virtual_RtME(1,1),Virtual_RtME(1,2),Virtual_RtME(1,3),'   RtME')
scatter3(Virtual_RtLE(1,1),Virtual_RtLE(1,2),Virtual_RtLE(1,3),'r','filled')
text(Virtual_RtLE(1,1),Virtual_RtLE(1,2),Virtual_RtLE(1,3),'   RtLE')
scatter3(R_ACR(1,1),R_ACR(1,2),R_ACR(1,3),'r','filled')
text(R_ACR(1,1),R_ACR(1,2),R_ACR(1,3),'  RtAcro')

% scatter3(Rt_hum_cl_1(1,1),Rt_hum_cl_1(1,2),Rt_hum_cl_1(1,3),'r','filled')
% scatter3(Rt_hum_cl_2(1,1),Rt_hum_cl_2(1,2),Rt_hum_cl_2(1,3),'r','filled')
% scatter3(Rt_hum_cl_3(1,1),Rt_hum_cl_3(1,2),Rt_hum_cl_3(1,3),'r','filled')
% 
% scatter3(Rt_FA_cl_1(1,1),Rt_FA_cl_1(1,2),Rt_FA_cl_1(1,3),'r','filled')
% scatter3(Rt_FA_cl_2(1,1),Rt_FA_cl_2(1,2),Rt_FA_cl_2(1,3),'r','filled')
% scatter3(Rt_FA_cl_3(1,1),Rt_FA_cl_3(1,2),Rt_FA_cl_3(1,3),'r','filled')
% 
% scatter3(Lft_hum_cl_1(1,1),Lft_hum_cl_1(1,2),Lft_hum_cl_1(1,3),'c','filled')
% scatter3(Lft_hum_cl_2(1,1),Lft_hum_cl_2(1,2),Lft_hum_cl_2(1,3),'c','filled')
% scatter3(Lft_hum_cl_3(1,1),Lft_hum_cl_3(1,2),Lft_hum_cl_3(1,3),'c','filled')
% 
% scatter3(Lft_FA_cl_1(1,1),Lft_FA_cl_1(1,2),Lft_FA_cl_1(1,3),'c','filled')
% scatter3(Lft_FA_cl_2(1,1),Lft_FA_cl_2(1,2),Lft_FA_cl_2(1,3),'c','filled')
% scatter3(Lft_FA_cl_3(1,1),Lft_FA_cl_3(1,2),Lft_FA_cl_3(1,3),'c','filled')

%JK: swapped virtual acr markers with real ACR markers 
line([Virtual_XP(1,1),Virtual_SS(1,1)],[Virtual_XP(1,2),Virtual_SS(1,2)],[Virtual_XP(1,3),Virtual_SS(1,3)],'Color','black')
line([Virtual_C7(1,1),Virtual_SS(1,1)],[Virtual_C7(1,2),Virtual_SS(1,2)],[Virtual_C7(1,3),Virtual_SS(1,3)],'Color','black')
line([Virtual_XP(1,1),Virtual_L5(1,1)],[Virtual_XP(1,2),Virtual_L5(1,2)],[Virtual_XP(1,3),Virtual_L5(1,3)],'Color','black')
line([Virtual_XP(1,1),Virtual_L5(1,1)],[Virtual_XP(1,2),Virtual_L5(1,2)],[Virtual_XP(1,3),Virtual_L5(1,3)],'Color','black')
line([Virtual_C7(1,1),Virtual_L5(1,1)],[Virtual_C7(1,2),Virtual_L5(1,2)],[Virtual_C7(1,3),Virtual_L5(1,3)],'Color','black')
line([R_ACR(1,1),Virtual_C7(1,1)],[R_ACR(1,2),Virtual_C7(1,2)],[R_ACR(1,3),Virtual_C7(1,3)],'Color','black')
line([R_ACR(1,1),Virtual_SS(1,1)],[R_ACR(1,2),Virtual_SS(1,2)],[R_ACR(1,3),Virtual_SS(1,3)],'Color','black')
line([R_ACR(1,1),Virtual_XP(1,1)],[R_ACR(1,2),Virtual_XP(1,2)],[R_ACR(1,3),Virtual_XP(1,3)],'Color','black')
line([R_ACR(1,1),Virtual_L5(1,1)],[R_ACR(1,2),Virtual_L5(1,2)],[R_ACR(1,3),Virtual_L5(1,3)],'Color','black')
line([L_ACR(1,1),Virtual_C7(1,1)],[L_ACR(1,2),Virtual_C7(1,2)],[L_ACR(1,3),Virtual_C7(1,3)],'Color','black')
line([L_ACR(1,1),Virtual_SS(1,1)],[L_ACR(1,2),Virtual_SS(1,2)],[L_ACR(1,3),Virtual_SS(1,3)],'Color','black')
line([L_ACR(1,1),Virtual_XP(1,1)],[L_ACR(1,2),Virtual_XP(1,2)],[L_ACR(1,3),Virtual_XP(1,3)],'Color','black')
line([L_ACR(1,1),Virtual_L5(1,1)],[L_ACR(1,2),Virtual_L5(1,2)],[L_ACR(1,3),Virtual_L5(1,3)],'Color','black')
line([R_ACR(1,1),Virtual_RtME(1,1)],[R_ACR(1,2),Virtual_RtME(1,2)],[R_ACR(1,3),Virtual_RtME(1,3)],'Color','black')
line([R_ACR(1,1),Virtual_RtLE(1,1)],[R_ACR(1,2),Virtual_RtLE(1,2)],[R_ACR(1,3),Virtual_RtLE(1,3)],'Color','black')
line([Virtual_RtME(1,1),Virtual_RtLE(1,1)],[Virtual_RtME(1,2),Virtual_RtLE(1,2)],[Virtual_RtME(1,3),Virtual_RtLE(1,3)],'Color','black')
line([L_ACR(1,1),Virtual_LftME(1,1)],[L_ACR(1,2),Virtual_LftME(1,2)],[L_ACR(1,3),Virtual_LftME(1,3)],'Color','black')
line([L_ACR(1,1),Virtual_LftLE(1,1)],[L_ACR(1,2),Virtual_LftLE(1,2)],[L_ACR(1,3),Virtual_LftLE(1,3)],'Color','black')
line([Virtual_LftME(1,1),Virtual_LftLE(1,1)],[Virtual_LftME(1,2),Virtual_LftLE(1,2)],[Virtual_LftME(1,3),Virtual_LftLE(1,3)],'Color','black')
line([Virtual_RtRS(1,1),Virtual_RtUS(1,1)],[Virtual_RtRS(1,2),Virtual_RtUS(1,2)],[Virtual_RtRS(1,3),Virtual_RtUS(1,3)],'Color','black')
line([Virtual_RtRS(1,1),Virtual_RtLE(1,1)],[Virtual_RtRS(1,2),Virtual_RtLE(1,2)],[Virtual_RtRS(1,3),Virtual_RtLE(1,3)],'Color','black')
line([Virtual_RtRS(1,1),Virtual_RtME(1,1)],[Virtual_RtRS(1,2),Virtual_RtME(1,2)],[Virtual_RtRS(1,3),Virtual_RtME(1,3)],'Color','black')
line([Virtual_RtUS(1,1),Virtual_RtME(1,1)],[Virtual_RtUS(1,2),Virtual_RtME(1,2)],[Virtual_RtUS(1,3),Virtual_RtME(1,3)],'Color','black')
line([Virtual_RtUS(1,1),Virtual_RtLE(1,1)],[Virtual_RtUS(1,2),Virtual_RtLE(1,2)],[Virtual_RtUS(1,3),Virtual_RtLE(1,3)],'Color','black')
line([Virtual_LftRS(1,1),Virtual_LftUS(1,1)],[Virtual_LftRS(1,2),Virtual_LftUS(1,2)],[Virtual_LftRS(1,3),Virtual_LftUS(1,3)],'Color','black')
line([Virtual_LftRS(1,1),Virtual_LftLE(1,1)],[Virtual_LftRS(1,2),Virtual_LftLE(1,2)],[Virtual_LftRS(1,3),Virtual_LftLE(1,3)],'Color','black')
line([Virtual_LftRS(1,1),Virtual_LftME(1,1)],[Virtual_LftRS(1,2),Virtual_LftME(1,2)],[Virtual_LftRS(1,3),Virtual_LftME(1,3)],'Color','black')
line([Virtual_LftUS(1,1),Virtual_LftME(1,1)],[Virtual_LftUS(1,2),Virtual_LftME(1,2)],[Virtual_LftUS(1,3),Virtual_LftME(1,3)],'Color','black')
line([Virtual_LftUS(1,1),Virtual_LftLE(1,1)],[Virtual_LftUS(1,2),Virtual_LftLE(1,2)],[Virtual_LftUS(1,3),Virtual_LftLE(1,3)],'Color','black')

% line([Point_B(1,1),Point_A(1,1)],[Point_B(1,2),Point_A(1,2)],[Point_B(1,3),Point_A(1,3)],'Color','red')
% line([R_ACR(1,1),EJC_Rt(1,1)],[R_ACR(1,2),EJC_Rt(1,2)],[R_ACR(1,3),EJC_Rt(1,3)],'Color','red')
% line([L_ACR(1,1),EJC_Lft(1,1)],[L_ACR(1,2),EJC_Lft(1,2)],[L_ACR(1,3),EJC_Lft(1,3)],'Color','red')

pause 
hold off 

%Create segment dcm 
%Calculate joint centers 
WJC_Rt(1,:)=(Virtual_RtUS(1,:)+Virtual_RtRS(1,:))/2;
EJC_Rt(1,:)=(Virtual_RtLE(1,:)+Virtual_RtME(1,:))/2;
WJC_Lft(1,:)=(Virtual_LftUS(1,:)+Virtual_LftRS(1,:))/2;
EJC_Lft(1,:)=(Virtual_LftLE(1,:)+Virtual_LftME(1,:))/2;

%Forearm Right
Rt_FA_y_seg(1,:)=EJC_Rt(1,:)-Virtual_RtLE(1,:); 
segment_temp_RtFA(1,:)=Virtual_RtRS(1,:)-Virtual_RtUS(1,:); 
Rt_FA_x_seg(1,:)=cross(Rt_FA_y_seg(1,:),segment_temp_RtFA(1,:));
Rt_FA_z_seg(1,:)=cross(Rt_FA_x_seg(1,:),Rt_FA_y_seg(1,:));

Rt_FA_y(1,:)=Rt_FA_y_seg(1,:)/norm(Rt_FA_y_seg(1,:)); 
Rt_FA_x(1,:)=Rt_FA_x_seg(1,:)/norm(Rt_FA_x_seg(1,:)); 
Rt_FA_z(1,:)=Rt_FA_z_seg(1,:)/norm(Rt_FA_z_seg(1,:)); 

%Upper Arm Right 
Rt_UA_y_seg(1,:)=R_ACR(1,:)-Virtual_RtLE(1,1); %JK: shouldnt this br RACR (or SJC) to EJC? are we using ACR as the JC for shoulder?
segment_temp_RtUA(1,:)=Virtual_RtME(1,:)-Virtual_RtLE(1,:);
Rt_UA_x_seg(1,:)=cross(Rt_UA_y_seg(1,:),segment_temp_RtUA(1,:));
Rt_UA_z_seg(1,:)=cross(Rt_UA_x_seg(1,:),Rt_UA_y_seg(1,:));

Rt_UA_y(1,:)=Rt_UA_y_seg(1,:)/norm(Rt_UA_y_seg(1,:));
Rt_UA_x(1,:)=Rt_UA_x_seg(1,:)/norm(Rt_UA_x_seg(1,:)); 
Rt_UA_z(1,:)=Rt_UA_z_seg(1,:)/norm(Rt_UA_z_seg(1,:)); 

%Forearm Left
Lft_FA_y_seg(1,:)=EJC_Lft(1,:)-Virtual_LftLE(1,:); 
segment_temp_LftFA(1,:)=Virtual_LftRS(1,:)-Virtual_LftUS(1,:); 
Lft_FA_x_seg(1,:)=cross(Lft_FA_y_seg(1,:),segment_temp_LftFA(1,:));
Lft_FA_z_seg(1,:)=cross(Lft_FA_x_seg(1,:),Lft_FA_y_seg(1,:));

Lft_FA_y(1,:)=Lft_FA_y_seg(1,:)/norm(Lft_FA_y_seg(1,:)); 
Lft_FA_x(1,:)=Lft_FA_x_seg(1,:)/norm(Lft_FA_x_seg(1,:)); 
Lft_FA_z(1,:)=Lft_FA_z_seg(1,:)/norm(Lft_FA_z_seg(1,:)); 

%Upper Arm Left 
Lft_UA_y_seg(1,:)=L_ACR(1,:)-Virtual_LftLE(1,1); %JK: changed to L_ACR
segment_temp_LftUA(1,:)=Virtual_LftME(1,:)-Virtual_LftLE(1,:);
Lft_UA_x_seg(1,:)=cross(Lft_UA_y_seg(1,:),segment_temp_LftUA(1,:));
Lft_UA_z_seg(1,:)=cross(Lft_UA_x_seg(1,:),Lft_UA_y_seg(1,:));

Lft_UA_y(1,:)=Lft_UA_y_seg(1,:)/norm(Lft_UA_y_seg(1,:)); 
Lft_UA_x(1,:)=Lft_UA_x_seg(1,:)/norm(Lft_UA_x_seg(1,:)); 
Lft_UA_z(1,:)=Lft_UA_z_seg(1,:)/norm(Lft_UA_z_seg(1,:));

%torso
Point_A(1,:)=(Virtual_T8(1,:)+Virtual_XP(1,:))/2; 
Point_B(1,:)=(Virtual_C7(1,:)+Virtual_SS(1,:))/2; 
TJC(1,:)=(Point_B(1,:)+Point_A(1,:))/2; 
        
T_y_seg(1,:)=Point_B(1,:)-Point_A(1,:); 
segment_temp_T(1,:)=R_ACR(1,:)-Point_B(1,:); 
T_x_seg(1,:)=cross(T_y_seg(1,:),segment_temp_T(1,:));
T_z_seg(1,:)=cross(T_x_seg(1,:),T_y_seg(1,:));
T_y(1,:)=T_y_seg(1,:)/norm(T_y_seg(1,:)); 
T_x(1,:)=T_x_seg(1,:)/norm(T_x_seg(1,:)); 
T_z(1,:)=T_z_seg(1,:)/norm(T_z_seg(1,:));

%create direction cosine matrices  
    Rt_FA_DC_GL(1,:).dcm=[Rt_FA_x(1,:);Rt_FA_y(1,:);Rt_FA_z(1,:)]';
    Rt_UA_DC_GL(1,:).dcm=[Rt_UA_x(1,:);Rt_UA_y(1,:);Rt_UA_z(1,:)]';
    T_DC_GL(1,:).dcm=[T_x(1,:);T_y(1,:);T_z(1,:)]';
    Lft_FA_DC_GL(1,:).dcm=[Lft_FA_x(1,:);Lft_FA_y(1,:);Lft_FA_z(1,:)]';
    Lft_UA_DC_GL(1,:).dcm=[Lft_UA_x(1,:);Lft_UA_y(1,:);Lft_UA_z(1,:)]';
    Rt_FA_DC_LG(1,:).dcm=Rt_FA_DC_GL(1,:).dcm'; 
    Rt_UA_DC_LG(1,:).dcm=Rt_UA_DC_GL(1,:).dcm';
    T_DC_LG(1,:).dcm=T_DC_GL(1,:).dcm';
    Lft_FA_DC_LG(1,:).dcm=Lft_FA_DC_GL(1,:).dcm'; 
    Lft_UA_DC_LG(1,:).dcm=Lft_UA_DC_GL(1,:).dcm';
    
    uv_check_y(1,:)=sqrt(Lft_UA_y(1,1)^2+Lft_UA_y(1,2)^2+Lft_UA_y(1,3)^2); 
    uv_check_x(1,:)=sqrt(Lft_UA_x(1,1)^2+Lft_UA_x(1,2)^2+Lft_UA_x(1,3)^2);
    uv_check_z(1,:)=sqrt(Lft_UA_z(1,1)^2+Lft_UA_z(1,2)^2+Lft_UA_z(1,3)^2);
    
%Calculate realtive joint angles according to Wu 
%~~~~~~~UPPER ARM RELATIVE TO THORAX~~~~~~~%
%Calculate the rotation matrix and angles for YXY rotation sequence (Wu et al, 2005) 

%  %Calculate humerothoracic motion (humerus relative to thorax) using
%     %Y-X-Y' rotation sequence (Euler Method)  **Alan 
%     HT_beta_raw(i,1) = rad2deg(acos(RMht(2,2)));
%     HT_gamma_raw(i,1) =rad2deg(asin(RMht(1,2)/sin(deg2rad(HT_beta_raw(i,1)))));
%     HT_gamma2_raw(i,1) = rad2deg(asin(RMht(2,1)/sin(deg2rad(HT_beta_raw(i,1)))));
%     
     %Calculate humerothoracic motion (humerus relative to thorax) using
    %Y-X-Y' rotation sequence (Euler Method)
    % HT_beta_raw: Elevation of the Humerus (Euler Method)
    %HT_gamma_raw: Interal/External Rotation of the Humerus (Euler Method)
    %HT_gamma2_raw: Plane of Elevation of the Humerus (Euler Method)
    
    Rt_UA_T(1,:).dcm=T_DC_LG(1,:).dcm*Rt_UA_DC_GL(1,:).dcm;
    R_HT_beta_raw(1,1) = rad2deg(acos(Rt_UA_T(1,:).dcm(2,2)));
    R_HT_gamma_raw(1,1) = rad2deg(asin(Rt_UA_T(1,:).dcm(1,2)/sin(deg2rad(R_HT_beta_raw(1,1)))));
    R_HT_gamma2_raw(1,1) = rad2deg(asin(Rt_UA_T(1,:).dcm(2,1)/sin(deg2rad(R_HT_beta_raw(1,1)))));

%     unwrap_R_HT_beta_raw(1,1)=unwrap(R_HT_beta_raw(1,1))
%     unwrap_R_HT_beta_raw(1,1)=rad2deg(unwrap_R_HT_beta_raw(1,1))
    
    %Left 
    Lft_UA_T(1,:).dcm=T_DC_LG(1,:).dcm*Lft_UA_DC_GL(1,:).dcm;
    L_HT_beta_raw(1,1) = rad2deg(acos(Lft_UA_T(1,:).dcm(2,2)));
    L_HT_gamma_raw(1,1) = rad2deg(asin(Lft_UA_T(1,:).dcm(1,2)/sin(deg2rad(L_HT_beta_raw(1,1)))));
    L_HT_gamma2_raw(1,1) = rad2deg(asin(Lft_UA_T(1,:).dcm(2,1)/sin(deg2rad(L_HT_beta_raw(1,1)))));
    
% %Right 
%    Rt_Rot_x_UA_T(1,1)=acosd(Rt_UA_T(1,:).dcm(2,2)); %Abduction  
%    Rt_Rot_y_UA_T(1,1)=180/pi*(atan2(Rt_UA_T(1,:).dcm(2,1),-(Rt_UA_T(1,:).dcm(2,3)))); %Axial Rotation  
%    Rt_Rot_z_UA_T(1,1)=180/pi*(atan2(Rt_UA_T(1,:).dcm(1,2),Rt_UA_T(1,:).dcm(3,2))); %Ant/posterior translation 
%left
%    Lft_Rot_x_UA_T(1,1)=acosd(Lft_UA_T(1,:).dcm(2,2)); %Abduction  
%    Lft_Rot_y_UA_T(1,1)=180/pi*(atan2(Lft_UA_T(1,:).dcm(2,1),-(Lft_UA_T(1,:).dcm(2,3)))); %Axial Rotation  
%    Lft_Rot_z_UA_T(1,1)=180/pi*(atan2(Lft_UA_T(1,:).dcm(1,2),Lft_UA_T(1,:).dcm(3,2))); %Ant/posterior

%Thorax 
    Rot_x_T_GL(1,1)=asind(T_DC_GL(1,:).dcm(3,2));
    Rot_y_T_GL(1,1)=180/pi*(atan2(-(T_DC_GL(1,:).dcm(3,1)),T_DC_GL(1,:).dcm(3,3)));
    Rot_z_T_GL(1,1)=180/pi*(atan2(-(T_DC_GL(1,:).dcm(1,2)),T_DC_GL(1,:).dcm(2,2))); %Lateral bend (-ve leftward; +ve rightward)

    Rt_UA_Ang=horzcat(R_HT_beta_raw,R_HT_gamma_raw,R_HT_gamma2_raw); 
    Lft_UA_Ang=horzcat(L_HT_beta_raw, L_HT_gamma_raw, L_HT_gamma2_raw); 
    Thorax_Ang=horzcat(Rot_x_T_GL, Rot_y_T_GL, Rot_z_T_GL); 
    
%~~~~~~FOREARM RELATIVE TO UPPER ARM (ulna relative to humerus~~~~~~%Calculate the rotation matrix and angles for ZXY rotation sequence (Wu et al, 2005)
%     Rt_FA_UA(1,:).dcm=Rt_UA_DC_GL(1,:).dcm'*Rt_FA_DC_GL(1,:).dcm;
%     Rt_Rot_x_FA_UA(1,1)=asind(Rt_FA_UA(1,:).dcm(3,2))  %Axial rotation (0=full pronation; 180=full supination) 
%     Rt_Rot_z_FA_UA(1,1)=180/pi*(atan2(-(Rt_FA_UA(1,:).dcm(1,2)),Rt_FA_UA(1,:).dcm(2,2)))
%     Rt_Rot_y_FA_UA(1,1)=180/pi*(atan2(-(Rt_FA_UA(1,:).dcm(3,1)),Rt_FA_UA(1,:).dcm(3,3))) %Abduction/adduction   
% 
%     Lft_FA_UA(1,:).dcm=Lft_UA_DC_GL(1,:).dcm'*Lft_FA_DC_GL(1,:).dcm;
%     Lft_Rot_x_FA_UA(1,1)=asind(Lft_FA_UA(1,:).dcm(3,2));  %Axial rotation (0=full pronation; 180=full supination) 
%     Lft_Rot_z_FA_UA(1,1)=180/pi*(atan2(-(Lft_FA_UA(1,:).dcm(1,2)),Lft_FA_UA(1,:).dcm(2,2)));
%     Lft_Rot_y_FA_UA(1,1)=180/pi*(atan2(-(Lft_FA_UA(1,:).dcm(3,1)),Lft_FA_UA(1,:).dcm(3,3)));  %Abduction/adduction  translation 
    
   


%%Calculate Joint forces and moments 
%Convert mm to m
Virtual_XP(1,:)=Virtual_XP/1000; 
Virtual_SS(1,:)=Virtual_SS/1000; 
Virtual_C7(1,:)=Virtual_C7/1000; 
Virtual_T8(1,:)=Virtual_T8/1000; 
Virtual_L5(1,:)=Virtual_L5/1000; 

Virtual_RtLE(1,:)=Virtual_RtLE/1000; 
Virtual_RtME(1,:)=Virtual_RtME/1000;  
%R_ACR(1,:)=R_ACR/1000; 
R_ACR(1,:)=R_ACR/1000; %added JK
Virtual_RtRS(1,:)=Virtual_RtRS/1000; 
Virtual_RtUS(1,:)=Virtual_RtUS/1000;  

Virtual_LftLE(1,:)=Virtual_LftLE/1000; 
Virtual_LftME(1,:)=Virtual_LftME/1000; 
%L_ACR(1,:)=L_ACR/1000; 
L_ACR(1,:)=L_ACR/1000;
Virtual_LftRS(1,:)=Virtual_LftRS/1000; 
Virtual_LftUS(1,:)=Virtual_LftUS/1000; 

WJC_Rt(1,:)=WJC_Rt(1,:)/1000; 
EJC_Rt(1,:)=EJC_Rt(1,:)/1000; 
WJC_Lft(1,:)=WJC_Lft(1,:)/1000; 
EJC_Lft(1,:)=EJC_Lft(1,:)/1000; 
TJC(1,:)=TJC(1,:)/100;
Point_A(1,:)=Point_A(1,:)/1000; 
Point_B(1,:)=Point_B(1,:)/1000; 

longaxistorso=Point_B(1,:)-Point_A(1,:); 

lshoulder(1,1:3)=L_ACR(1,1:3) + 50*longaxistorso(1,1:3); %subbed L_ACR

Rshoulder(1,1:3)=R_ACR(1,1:3) + 50*longaxistorso(1,1:3);


%Estimate segment masses 
forearm_mass=mass*0.016+mass*0.006; %%account for forearm and hand mass u; 
mass_trunk=mass*0.584;
hand_mass=mass*0.006; 
upperarm_mass=mass*0.028; 
%Calc segment CoM

%Hand_COM(i,:)=WJC(i,:)-(0.494*(WJC(i,:)-HJC(i,:))); 
Rt_Forearm_COM(1,:)=EJC_Rt(1,:)-(0.430*(EJC_Rt(1,:)-WJC_Rt(1,:))); 
Rt_Upperarm_COM(1,:)=R_ACR(1,:)-(0.436*(R_ACR(1,:)-EJC_Rt(1,:))); 

Lft_Forearm_COM(1,:)=EJC_Lft(1,:)-(0.430*(EJC_Lft(1,:)-WJC_Lft(1,:))); 
Lft_Upperarm_COM(1,:)=L_ACR(1,:)-(0.436*(L_ACR(1,:)-EJC_Lft(1,:)));

%Torso - solve at L5 
  
%check joint centers 
scatter3(Rt_Forearm_COM(1,1),Rt_Forearm_COM(1,2),Rt_Forearm_COM(1,3),'c','filled')
text(Rt_Forearm_COM(1,1),Rt_Forearm_COM(1,2),Rt_Forearm_COM(1,3),'   Rt_4A')
hold on 
scatter3(Lft_Forearm_COM(1,1),Lft_Forearm_COM(1,2),Lft_Forearm_COM(1,3),'c','filled')
text(Lft_Forearm_COM(1,1),Lft_Forearm_COM(1,2),Lft_Forearm_COM(1,3),'   Lft_4A')

scatter3(Rt_Upperarm_COM(1,1),Rt_Upperarm_COM(1,2),Rt_Upperarm_COM(1,3),'c','filled')
text(Rt_Upperarm_COM(1,1),Rt_Upperarm_COM(1,2),Rt_Upperarm_COM(1,3),'   Rt_UA')

scatter3(Lft_Upperarm_COM(1,1),Lft_Upperarm_COM(1,2),Lft_Upperarm_COM(1,3),'c','filled')
text(Lft_Upperarm_COM(1,1),Lft_Upperarm_COM(1,2),Lft_Upperarm_COM(1,3),'   Lft_UA')

scatter3(Virtual_SS(1,1),Virtual_SS(1,2),Virtual_SS(1,3),'m','filled')
text(Virtual_SS(1,1),Virtual_SS(1,2),Virtual_SS(1,3),'   SS')

scatter3(Virtual_XP(1,1),Virtual_XP(1,2),Virtual_XP(1,3),'m','filled')
text(Virtual_XP(1,1),Virtual_XP(1,2),Virtual_XP(1,3),'   XP')
scatter3(Virtual_L5(1,1),Virtual_L5(1,2),Virtual_L5(1,3),'m','filled')
text(Virtual_L5(1,1),Virtual_L5(1,2),Virtual_L5(1,3),'   L5') 
scatter3(Virtual_C7(1,1),Virtual_C7(1,2),Virtual_C7(1,3),'m','filled')
text(Virtual_C7(1,1),Virtual_C7(1,2),Virtual_C7(1,3),'   C7')
scatter3(Virtual_T8(1,1),Virtual_T8(1,2),Virtual_T8(1,3),'m','filled')
text(Virtual_T8(1,1),Virtual_T8(1,2),Virtual_T8(1,3),'   T8')

scatter3(Virtual_LftUS(1,1),Virtual_LftUS(1,2),Virtual_LftUS(1,3),'c','filled')
text(Virtual_LftUS(1,1),Virtual_LftUS(1,2),Virtual_LftUS(1,3),'   LftUS')
scatter3(Virtual_LftRS(1,1),Virtual_LftRS(1,2),Virtual_LftRS(1,3),'c','filled')
text(Virtual_LftRS(1,1),Virtual_LftRS(1,2),Virtual_LftRS(1,3),'   LftRS')
scatter3(Virtual_LftME(1,1),Virtual_LftME(1,2),Virtual_LftME(1,3),'c','filled')
text(Virtual_LftME(1,1),Virtual_LftME(1,2),Virtual_LftME(1,3),'   LftME')
scatter3(Virtual_LftLE(1,1),Virtual_LftLE(1,2),Virtual_LftLE(1,3),'c','filled')
text(Virtual_LftLE(1,1),Virtual_LftLE(1,2),Virtual_LftLE(1,3),'   LftLE')
scatter3(L_ACR(1,1),L_ACR(1,2),L_ACR(1,3),'c','filled')
text(L_ACR(1,1),L_ACR(1,2),L_ACR(1,3),'  LftAcro')

scatter3(Virtual_RtUS(1,1),Virtual_RtUS(1,2),Virtual_RtUS(1,3),'r','filled')
text(Virtual_RtUS(1,1),Virtual_RtUS(1,2),Virtual_RtUS(1,3),'   RtUS')
scatter3(Virtual_RtRS(1,1),Virtual_RtRS(1,2),Virtual_RtRS(1,3),'r','filled')
text(Virtual_RtRS(1,1),Virtual_RtRS(1,2),Virtual_RtRS(1,3),'   RtRS')
scatter3(Virtual_RtME(1,1),Virtual_RtME(1,2),Virtual_RtME(1,3),'r','filled')
text(Virtual_RtME(1,1),Virtual_RtME(1,2),Virtual_RtME(1,3),'   RtME')
scatter3(Virtual_RtLE(1,1),Virtual_RtLE(1,2),Virtual_RtLE(1,3),'r','filled')
text(Virtual_RtLE(1,1),Virtual_RtLE(1,2),Virtual_RtLE(1,3),'   RtLE')
scatter3(R_ACR(1,1),R_ACR(1,2),R_ACR(1,3),'r','filled')
text(R_ACR(1,1),R_ACR(1,2),R_ACR(1,3),'  RtAcro')

%--------------------------------------------------------------------------------------------------

%Calculate Static Joint Forces 

g=9.81;
g_direction=[0,-1,0]; 

Rt_F_gravity_forearm_x(1,:)=(forearm_mass*g)*(dot(g_direction, Rt_FA_DC_GL(1,:).dcm(:,1))); 
Rt_F_gravity_forearm_y(1,:)=(forearm_mass*g)*(dot(g_direction, Rt_FA_DC_GL(1,:).dcm(:,2)));
Rt_F_gravity_forearm_z(1,:)=(forearm_mass*g)*(dot(g_direction, Rt_FA_DC_GL(1,:).dcm(:,3)));
Rt_F_gravity_forearm=horzcat(Rt_F_gravity_forearm_x,Rt_F_gravity_forearm_y,Rt_F_gravity_forearm_z); 
Rt_F_gravity_upperarm_x(1,:)=(upperarm_mass*g)*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,1)));
Rt_F_gravity_upperarm_y(1,:)=(upperarm_mass*g)*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,2))); 
Rt_F_gravity_upperarm_z(1,:)=(upperarm_mass*g)*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,3))); 
Rt_F_gravity_upperarm=horzcat(Rt_F_gravity_upperarm_x,Rt_F_gravity_upperarm_y,Rt_F_gravity_upperarm_z); 
Lft_F_gravity_forearm_x(1,:)=(forearm_mass*g)*(dot(g_direction, Lft_FA_DC_GL(1,:).dcm(:,1))); 
Lft_F_gravity_forearm_y(1,:)=(forearm_mass*g)*(dot(g_direction, Lft_FA_DC_GL(1,:).dcm(:,2)));
Lft_F_gravity_forearm_z(1,:)=(forearm_mass*g)*(dot(g_direction, Lft_FA_DC_GL(1,:).dcm(:,3)));
Lft_F_gravity_forearm=horzcat(Lft_F_gravity_forearm_x,Lft_F_gravity_forearm_y,Lft_F_gravity_forearm_z); 
Lft_F_gravity_upperarm_x(1,:)=(upperarm_mass*g)*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,1)));
Lft_F_gravity_upperarm_y(1,:)=(upperarm_mass*g)*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,2))); 
Lft_F_gravity_upperarm_z(1,:)=(upperarm_mass*g)*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,3)));
Lft_F_gravity_upperarm=horzcat(Lft_F_gravity_upperarm_x,Lft_F_gravity_upperarm_y,Lft_F_gravity_upperarm_z); 
F_gravity_torso_x(1,:)=(mass_trunk*g)*(dot(g_direction,  T_DC_GL(1,:).dcm(:,1))); 
F_gravity_torso_y(1,:)=(mass_trunk*g)*(dot(g_direction,  T_DC_GL(1,:).dcm(:,2))); 
F_gravity_torso_z(1,:)=(mass_trunk*g)*(dot(g_direction,  T_DC_GL(1,:).dcm(:,3))); 
F_gravity_torso=horzcat(F_gravity_torso_x,F_gravity_torso_y,F_gravity_torso_z); 

Rt_F_ladder_x(1,:)=(ladder_force_rightHand)*(dot(g_direction,Rt_FA_DC_GL(1,:).dcm(:,1))); %%removed multiplying the ladder force by g since ladder force inputs are in NEWTONS
Rt_F_ladder_y(1,:)=(ladder_force_rightHand)*(dot(g_direction,Rt_FA_DC_GL(1,:).dcm(:,2))); %%in original study must have recorded ladder force in kg, then converted to N here 
Rt_F_ladder_z(1,:)=(ladder_force_rightHand)*(dot(g_direction,Rt_FA_DC_GL(1,:).dcm(:,3)));
Rt_F_ladder=horzcat(Rt_F_ladder_x,Rt_F_ladder_y,Rt_F_ladder_z); 

Lft_F_ladder_x(1,:)=(ladder_force_leftHand)*(dot(g_direction,Lft_FA_DC_GL(1,:).dcm(:,1))); 
Lft_F_ladder_y(1,:)=(ladder_force_leftHand)*(dot(g_direction,Lft_FA_DC_GL(1,:).dcm(:,2)));
Lft_F_ladder_z(1,:)=(ladder_force_leftHand)*(dot(g_direction,Lft_FA_DC_GL(1,:).dcm(:,3)));
Lft_F_ladder=horzcat(Lft_F_ladder_x,Lft_F_ladder_y,Lft_F_ladder_z); 

RtSh_F_ladder_x(1,:)=(ladder_force_rightShoulder)*(dot(g_direction,T_DC_GL(1,:).dcm(:,1))); 
RtSh_F_ladder_y(1,:)=(ladder_force_rightShoulder)*(dot(g_direction,T_DC_GL(1,:).dcm(:,2)));
RtSh_F_ladder_z(1,:)=(ladder_force_rightShoulder)*(dot(g_direction,T_DC_GL(1,:).dcm(:,3)));
RtSh_F_ladder=horzcat(RtSh_F_ladder_x,RtSh_F_ladder_y,RtSh_F_ladder_z);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Calcualte joint forces in global 
%Calculate static forces in local 
%Shoulder static force
forearm_mass=mass*0.016+mass*0.006; %%account for forearm and hand mass u; 
mass_trunk=mass*0.584;
hand_mass=mass*0.006; 
upperarm_mass=mass*0.028; 

%Global forces 
Rt_F_elbow_global = (-1)*((ladder_force_rightHand*-1) + (forearm_mass*-g));   %changed -g to -1 for all ladder forces 
Lft_F_elbow_global = (-1)*((ladder_force_leftHand*-1) + (forearm_mass*-g));   

Rt_F_shoulder_global = (-1)*((ladder_force_rightHand*-1) + (forearm_mass*-g) + (upperarm_mass*-g) + (ladder_force_rightShoulder*-1));   
Lft_F_shoulder_global = (-1)*((ladder_force_leftHand*-1) + (forearm_mass*-g) + (upperarm_mass*-g)) ;   
Torso_F_global =(-1)*( (ladder_force_rightHand*-1) + (forearm_mass*-g) + (upperarm_mass*-g) + (ladder_force_rightShoulder*-1)+(ladder_force_leftHand*-1) + (forearm_mass*-g) + (upperarm_mass*-g) +(mass_trunk*-g)); 

%move into local systems 
Rt_F_shoulder_local_x(1,:)=Rt_F_shoulder_global*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,1)));
Rt_F_shoulder_local_y(1,:)=Rt_F_shoulder_global*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,2)));
Rt_F_shoulder_local_z(1,:)=Rt_F_shoulder_global*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,3)));

Lft_F_shoulder_local_x(1,:)=Lft_F_shoulder_global*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,1)));
Lft_F_shoulder_local_y(1,:)=Lft_F_shoulder_global*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,2)));
Lft_F_shoulder_local_z(1,:)=Lft_F_shoulder_global*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,3)));

F_torso_local_x(1,:)=Torso_F_global*(dot(g_direction,  T_DC_GL(1,:).dcm(:,1))); 
F_torso_local_y(1,:)=Torso_F_global*(dot(g_direction,  T_DC_GL(1,:).dcm(:,2))); 
F_torso_local_z(1,:)=Torso_F_global*(dot(g_direction,  T_DC_GL(1,:).dcm(:,3))); 

Rt_F_shoulder_local=horzcat(Rt_F_shoulder_local_x,Rt_F_shoulder_local_y,Rt_F_shoulder_local_z); 
Lft_F_shoulder_local=horzcat(Lft_F_shoulder_local_x,Lft_F_shoulder_local_y,Lft_F_shoulder_local_z); 
F_torso_local=horzcat(F_torso_local_x,F_torso_local_y,F_torso_local_z); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Calculate moment arms and put into local coordinate systems (GL)
R_Hand_proxma=WJC_Rt(:,1:3);
L_Hand_proxma=WJC_Lft(:,1:3);

R_Forearm_proxma=EJC_Rt(1,:)-Rt_Forearm_COM(1,:);
R_Forearm_distma=WJC_Rt(1,:)-Rt_Forearm_COM(1,:);

    R_Forearm_proxma_x_GL(1,:)=dot(R_Forearm_proxma(1,:),Rt_FA_DC_GL(1,:).dcm(:,1)); 
    R_Forearm_proxma_y_GL(1,:)=dot(R_Forearm_proxma(1,:),Rt_FA_DC_GL(1,:).dcm(:,2)); 
    R_Forearm_proxma_z_GL(1,:)=dot(R_Forearm_proxma(1,:),Rt_FA_DC_GL(1,:).dcm(:,3)); 
    
    R_Forearm_proxma_GL=horzcat(R_Forearm_proxma_x_GL,R_Forearm_proxma_y_GL,R_Forearm_proxma_z_GL); 
    
    R_Forearm_distma_x_GL(1,:)=dot(R_Forearm_distma(1,:),Rt_FA_DC_GL(1,:).dcm(:,1)); 
    R_Forearm_distma_y_GL(1,:)=dot(R_Forearm_distma(1,:),Rt_FA_DC_GL(1,:).dcm(:,2)); 
    R_Forearm_distma_z_GL(1,:)=dot(R_Forearm_distma(1,:),Rt_FA_DC_GL(1,:).dcm(:,3)); 
    
    R_Forearm_distma_GL=horzcat(R_Forearm_distma_x_GL,R_Forearm_distma_y_GL,R_Forearm_distma_z_GL); 

L_Forearm_proxma=EJC_Lft(1,:)-Lft_Forearm_COM(1,:);
L_Forearm_distma=WJC_Lft(1,:)-Lft_Forearm_COM(1,:);

    L_Forearm_proxma_x_GL(1,:)=dot(L_Forearm_proxma(1,:),Lft_FA_DC_GL(1,:).dcm(:,1)); 
    L_Forearm_proxma_y_GL(1,:)=dot(L_Forearm_proxma(1,:),Lft_FA_DC_GL(1,:).dcm(:,2)); 
    L_Forearm_proxma_z_GL(1,:)=dot(L_Forearm_proxma(1,:),Lft_FA_DC_GL(1,:).dcm(:,3)); 
    
    L_Forearm_proxma_GL=horzcat(L_Forearm_proxma_x_GL,L_Forearm_proxma_y_GL,L_Forearm_proxma_z_GL); 
    
    L_Forearm_distma_x_GL(1,:)=dot(L_Forearm_distma(1,:),Lft_FA_DC_GL(1,:).dcm(:,1)); 
    L_Forearm_distma_y_GL(1,:)=dot(L_Forearm_distma(1,:),Lft_FA_DC_GL(1,:).dcm(:,2)); 
    L_Forearm_distma_z_GL(1,:)=dot(L_Forearm_distma(1,:),Lft_FA_DC_GL(1,:).dcm(:,3)); 
    
    L_Forearm_distma_GL=horzcat(L_Forearm_distma_x_GL,L_Forearm_distma_y_GL,L_Forearm_distma_z_GL);

R_UpperArm_proxma=R_ACR-Rt_Upperarm_COM(1,:);
R_UpperArm_distma=EJC_Rt(1,:)-Rt_Upperarm_COM(1,:);

    R_UpperArm_proxma_x_GL(1,:)=dot(R_UpperArm_proxma(1,:),Rt_UA_DC_GL(1,:).dcm(:,1)); 
    R_UpperArm_proxma_y_GL(1,:)=dot(R_UpperArm_proxma(1,:),Rt_UA_DC_GL(1,:).dcm(:,2)); 
    R_UpperArm_proxma_z_GL(1,:)=dot(R_UpperArm_proxma(1,:),Rt_UA_DC_GL(1,:).dcm(:,3)); 
    
    R_UpperArm_proxma_GL=horzcat(R_UpperArm_proxma_x_GL,R_UpperArm_proxma_y_GL,R_UpperArm_proxma_z_GL); 
    
    R_UpperArm_distma_x_GL(1,:)=dot(R_UpperArm_distma(1,:),Rt_UA_DC_GL(1,:).dcm(:,1)); 
    R_UpperArm_distma_y_GL(1,:)=dot(R_UpperArm_distma(1,:),Rt_UA_DC_GL(1,:).dcm(:,2)); 
    R_UpperArm_distma_z_GL(1,:)=dot(R_UpperArm_distma(1,:),Rt_UA_DC_GL(1,:).dcm(:,3)); 
    
    R_UpperArm_distma_GL=horzcat(R_UpperArm_distma_x_GL,R_UpperArm_distma_y_GL,R_UpperArm_distma_z_GL); 

L_UpperArm_proxma=L_ACR-Lft_Upperarm_COM(1,:);
L_UpperArm_distma=EJC_Lft(1,:)-Lft_Upperarm_COM(1,:);

    L_UpperArm_proxma_x_GL(1,:)=dot(L_UpperArm_proxma(1,:),Lft_UA_DC_GL(1,:).dcm(:,1)); 
    L_UpperArm_proxma_y_GL(1,:)=dot(L_UpperArm_proxma(1,:),Lft_UA_DC_GL(1,:).dcm(:,2)); 
    L_UpperArm_proxma_z_GL(1,:)=dot(L_UpperArm_proxma(1,:),Lft_UA_DC_GL(1,:).dcm(:,3)); 
    
    L_UpperArm_proxma_GL=horzcat(L_UpperArm_proxma_x_GL,L_UpperArm_proxma_y_GL,L_UpperArm_proxma_z_GL); 
    
    L_UpperArm_distma_x_GL(1,:)=dot(L_UpperArm_distma(1,:),Lft_UA_DC_GL(1,:).dcm(:,1)); 
    L_UpperArm_distma_y_GL(1,:)=dot(L_UpperArm_distma(1,:),Lft_UA_DC_GL(1,:).dcm(:,2)); 
    L_UpperArm_distma_z_GL(1,:)=dot(L_UpperArm_distma(1,:),Lft_UA_DC_GL(1,:).dcm(:,3)); 
    
    L_UpperArm_distma_GL=horzcat(L_UpperArm_distma_x_GL,L_UpperArm_distma_y_GL,L_UpperArm_distma_z_GL); 

SH_center(1,:)=(R_ACR(1,:)+L_ACR(1,:))/2;
t_vec=SH_center-Virtual_L5; 
cgt=Virtual_SS+t_vec*-0.449;
LowBack_proxma=(cgt-Virtual_L5);

l5s1_rshoul=R_ACR(1,:)-cgt(1,:); 
l5s1_lshoul=L_ACR(1,:)-cgt(1,:); 

l5s1_rshoul_x_GL(1,:)=dot(l5s1_rshoul(1,:),T_DC_GL(1,:).dcm(:,1)); 
l5s1_rshoul_y_GL(1,:)=dot(l5s1_rshoul(1,:),T_DC_GL(1,:).dcm(:,2)); 
l5s1_rshoul_z_GL(1,:)=dot(l5s1_rshoul(1,:),T_DC_GL(1,:).dcm(:,3)); 

l5s1_rshoul_GL=horzcat(l5s1_rshoul_x_GL,l5s1_rshoul_y_GL,l5s1_rshoul_z_GL);

l5s1_lshoul_x_GL(1,:)=dot(l5s1_lshoul(1,:),T_DC_GL(1,:).dcm(:,1)); 
l5s1_lshoul_y_GL(1,:)=dot(l5s1_lshoul(1,:),T_DC_GL(1,:).dcm(:,2)); 
l5s1_lshoul_z_GL(1,:)=dot(l5s1_lshoul(1,:),T_DC_GL(1,:).dcm(:,3)); 

l5s1_lshoul_GL=horzcat(l5s1_lshoul_x_GL,l5s1_lshoul_y_GL,l5s1_lshoul_z_GL);

LowBack_proxma_x_GL(1,:)=dot(LowBack_proxma(1,:),T_DC_GL(1,:).dcm(:,1)); 
LowBack_proxma_y_GL(1,:)=dot(LowBack_proxma(1,:),T_DC_GL(1,:).dcm(:,2)); 
LowBack_proxma_z_GL(1,:)=dot(LowBack_proxma(1,:),T_DC_GL(1,:).dcm(:,3)); 

LowBack_proxma_GL=horzcat(LowBack_proxma_x_GL,LowBack_proxma_y_GL,LowBack_proxma_z_GL); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate moments in global 
g_direction=[0,-1,0];

Rt_F_elbow_global_gravity=Rt_F_elbow_global*g_direction; 
Rt_F_ladder_gravity = ladder_force_rightHand*g_direction; %removed *g here

Lft_F_elbow_global_gravity=Lft_F_elbow_global*g_direction; 
Lft_F_ladder_gravity = ladder_force_leftHand*g_direction; %removed *g

Rt_F_shoulder_global_gravity=Rt_F_shoulder_global*g_direction; 
Lft_F_shoulder_global_gravity=Lft_F_shoulder_global*g_direction;

Torso_F_global_gravity=Torso_F_global*g_direction;

R_StaticElbowTorque(1,:)=(-1)*( (cross(abs(R_Forearm_distma(1,:)),Rt_F_ladder_gravity(1,:))+(cross(abs(R_Forearm_proxma(1,:)),Rt_F_elbow_global_gravity(1,:))))); 
L_StaticElbowTorque(1,:)=(-1)*( (cross(abs(L_Forearm_distma(1,:)),Lft_F_ladder_gravity(1,:))+(cross(abs(L_Forearm_proxma(1,:)),Lft_F_elbow_global_gravity(1,:))))); 

R_StaticShoulderTorque(1,:)=(-1)*( (-R_StaticElbowTorque)+(cross(abs(R_UpperArm_distma(1,:)),Rt_F_elbow_global_gravity(1,:))+(cross(abs(R_UpperArm_proxma(1,:)),Rt_F_shoulder_global_gravity(1,:))))); 
L_StaticShoulderTorque(1,:)=(-1)*( (-L_StaticElbowTorque)+(cross(abs(L_UpperArm_distma(1,:)),Lft_F_elbow_global_gravity(1,:))+(cross(abs(L_UpperArm_proxma(1,:)),Lft_F_shoulder_global_gravity(1,:)))));

StaticTorsoTorque(1,:)=(-1)*((-R_StaticShoulderTorque)+(-L_StaticShoulderTorque)+(cross(abs(LowBack_proxma),Torso_F_global_gravity))+(cross(abs(l5s1_rshoul),Rt_F_shoulder_global_gravity))+(cross(abs(l5s1_lshoul),Lft_F_shoulder_global_gravity)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate humeral elevation angle with dot product 

%Shoulder joint center to elbow joint center and long axis of torso 

R_longaxisHum=(R_ACR-EJC_Rt)/norm(R_ACR-EJC_Rt); 
L_longaxisHum=(L_ACR-EJC_Lft)/norm(L_ACR-EJC_Lft);

longaxistorso=(Point_B(1,:)-Point_A(1,:))/norm(Point_B(1,:)-Point_A(1,:)); 

R_Hum_Elv=acosd(dot(longaxistorso,R_longaxisHum)); 
L_Hum_Elv=acosd(dot(longaxistorso,L_longaxisHum)); 

% Convert to local systems 
% move into local systems 
% Rt_M_shoulder_local_x(1,:)=R_StaticShoulderTorque*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,1)));
% Rt_M_shoulder_local_y(1,:)=R_StaticShoulderTorque*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,2)));
% Rt_M_shoulder_local_z(1,:)=R_StaticShoulderTorque*(dot(g_direction, Rt_UA_DC_GL(1,:).dcm(:,3)));
% 
% Lft_M_shoulder_local_x(1,:)=L_StaticShoulderTorque*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,1)));
% Lft_M_shoulder_local_y(1,:)=L_StaticShoulderTorque*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,2)));
% Lft_M_shoulder_local_z(1,:)=L_StaticShoulderTorque*(dot(g_direction, Lft_UA_DC_GL(1,:).dcm(:,3)));
% 
% M_torso_local_x(1,:)=StaticTorsoTorque*(dot(g_direction,  T_DC_GL(1,:).dcm(:,1))); 
% M_torso_local_y(1,:)=StaticTorsoTorque*(dot(g_direction,  T_DC_GL(1,:).dcm(:,2))); 
% M_torso_local_z(1,:)=StaticTorsoTorque*(dot(g_direction,  T_DC_GL(1,:).dcm(:,3))); 
% 
% Rt_M_shoulder_local=horzcat(Rt_M_shoulder_local_x,Rt_M_shoulder_local_y,Rt_M_shoulder_local_z); 
% Lft_M_shoulder_local=horzcat(Lft_M_shoulder_local_x,Lft_M_shoulder_local_y,Lft_M_shoulder_local_z); 
% M_torso_local=horzcat(M_torso_local_x,M_torso_local_y,M_torso_local_z); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Hum_Elevation_dot = horzcat(R_Hum_Elv,L_Hum_Elv); 
Angles=horzcat(Rt_UA_Ang, Lft_UA_Ang,Thorax_Ang);  
Forces_local = horzcat(Rt_F_shoulder_local, Lft_F_shoulder_local, F_torso_local); 
Forces_global = horzcat(Rt_F_shoulder_global, Lft_F_shoulder_global, Torso_F_global); 
Moments_global = horzcat(R_StaticShoulderTorque, L_StaticShoulderTorque, StaticTorsoTorque); 

A_F_M_dot_global = horzcat (Hum_Elevation_dot,Forces_global, Moments_global); 
A_F_M_dot_local = horzcat (Hum_Elevation_dot,Forces_local, Moments_global);
A_F_M_local = horzcat(Angles, Forces_local, Moments_global); 
A_F_M_global = horzcat(Angles, Forces_global, Moments_global); 

%%JK added:
%Sh_Elev_AnglesJK = horzcat(R_HT_beta_raw,L_HT_beta_raw);
Trunk_RotJK = horzcat(Rot_x_T_GL,Rot_y_T_GL,Rot_z_T_GL);
Forces_GlobalJK = horzcat(Rt_F_shoulder_global,Lft_F_shoulder_global,Torso_F_global);
Global_MomsJK = horzcat(R_StaticShoulderTorque, L_StaticShoulderTorque, StaticTorsoTorque);

ResultsSumm = horzcat(Hum_Elevation_dot, Trunk_RotJK, Forces_GlobalJK, Forces_local, Global_MomsJK);



%% Change file names for ladder (L1,L2,L3), task (C, R_F, R_W) and subject S#
%xlswrite([pathnames '\' trialname '_Moments_Nov25.xlsx'],Moments_global); 


%xlswrite([pathnames '\' trialname '_Ang_force_mom_local.xlsx'],A_F_M_local); 
%xlswrite([pathnames '\' trialname '_Ang_force_mom_global.xlsx'],A_F_M_global); 
%xlswrite([pathnames '\' trialname '_Angdot_force_mom_global.xlsx'],A_F_M_dot_global);
%xlswrite([pathnames '\' trialname '_Angdot_force_mom_local.xlsx'],A_F_M_dot_local);

xlswrite([pathnames '\' trialname '_TOP_ResultsSumm_Nov21_JK.xlsx'],ResultsSumm); %11 cols: this matches up with what we need to enter into the results summ excel sheet
xlswrite([pathnames '\' trialname '_TOP_Local_Forces_Nov21_JK.xlsx'],Forces_local); %9 cols: forces in x,y,z for RSh, LSh, T
xlswrite([pathnames '\' trialname '_TOP_HumElevationDot_Nov21_JK.xlsx'],Hum_Elevation_dot);

% xlswrite([pathnames '\' trialname '_top_ResultsSumm_dec16_JK.xlsx'],ResultsSumm); %11 cols: this matches up with what we need to enter into the results summ excel sheet
% xlswrite([pathnames '\' trialname '_top_Local_Forces_dec16_JK.xlsx'],Forces_local); %9 cols: forces in x,y,z for RSh, LSh, T
% xlswrite([pathnames '\' trialname '_top_HumElevationDot_dec16_JK.xlsx'],Hum_Elevation_dot);

% xlswrite([pathnames '\' trialname '_bottom_ResultsSumm_Nov21_JK.xlsx'],ResultsSumm); %11 cols: this matches up with what we need to enter into the results summ excel sheet
% xlswrite([pathnames '\' trialname '_bottom_Local_Forces_Nov21_JK.xlsx'],Forces_local); %9 cols: forces in x,y,z for RSh, LSh, T
% xlswrite([pathnames '\' trialname '_bottom_HumElevationDot_Nov21_JK.xlsx'],Hum_Elevation_dot);

% dlmwrite('S20_Joint_Angles_L1_C.txt',Joint_Angles,'\t'); 
% dlmwrite('S20_Joint_Forces_L1_C.txt',Joint_Forces,'\t'); 
% dlmwrite('S20_Joint_Torques_L1_C.txt',Joint_Torques,'\t'); 
% %  


end

end
