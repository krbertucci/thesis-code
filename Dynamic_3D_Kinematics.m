tbl_mvmnt_file_name='S01_2_014_3d';

% load raw data into tbl_mvmnt_raw
tbl_mvmnt_raw = readtable(strcat(tbl_mvmnt_file_name,'.csv'));
% IF NEED - CHOOSE FRAME INTERVAL CONTAINING RELIABLE DATA TO WORK WITHIN
tbl_mvmnt_raw = tbl_mvmnt_raw(106:256,:);
% Gait 14 Frame interval tbl_mvmnt_raw = tbl_mvmnt_raw(106:256,:);
% Gait 1    5 Frame interval tbl_mvmnt_raw = tbl_mvmnt_raw(120:266,:);
% Gait 16 Frame interval tbl_mvmnt_raw = tbl_mvmnt_raw(105:256,:);

% GAP FILL USING SPLINE INTERPOLATION
tbl_mvmnt_gapfilled=fillmissing(tbl_mvmnt_raw,'spline');

% convert tbl_mvmnt_raw into an array to be gap filled and filtered
array_mvmnt_gapfilled=table2array(tbl_mvmnt_gapfilled);
 
% LOW PASS FILTER DATA
% create temporary variable to save frame data
temp_frames=array_mvmnt_gapfilled(:,1);
% Filter using, 2nd order, Zero lag, butter worth filter
% cut off frequency of 6 Hz
% sample rate of 128 Hz (in accordance with nyquist theorem)
[b,a]=butter(2,6/64);
array_mvmnt_filt=filtfilt(b,a,array_mvmnt_gapfilled);
% Repair filtered frames using temp_frames
array_mvmnt_filt(:,1)=temp_frames;

% Convert array_mvmnt_filt back into table
tbl_mvmnt=array2table(array_mvmnt_filt,'VariableNames',tbl_mvmnt_raw.Properties.VariableNames);
   
% find number of columns and rows in tbl_mvmnt table
% number of frames is number of rows
tbl_mvmnt_size = size(tbl_mvmnt);
number_frames = tbl_mvmnt_size(1);

% set seconds per frame = time = 1/frames per second
% t_spf is time in seconds per frame
% freq_fps is the frequency in frames per second
freq_fps = 64;
t_spf = 1/freq_fps;

% unit conversion factor
length_unit_conversion_factor=0.001;

% ANKLE JOINT CENTER POSITION, VELOCITY & ACCELERATION
% Calculate xyz coordinates of ankle joint center in global coordinate
% system
tbl_mvmnt.ankle_joint_center_X=(tbl_mvmnt.iL_Foot_3_Mrk_R_MED_MALX+tbl_mvmnt.iL_Foot_3_Mrk_R_LAT_MALX)/2;
tbl_mvmnt.ankle_joint_center_Y=(tbl_mvmnt.iL_Foot_3_Mrk_R_MED_MALY+tbl_mvmnt.iL_Foot_3_Mrk_R_LAT_MALY)/2;
tbl_mvmnt.ankle_joint_center_Z=(tbl_mvmnt.iL_Foot_3_Mrk_R_MED_MALZ+tbl_mvmnt.iL_Foot_3_Mrk_R_LAT_MALZ)/2;

% calculate translational velocity of ankle joint center from position data
% in global cooridinate systsem

for frame_count=2:number_frames-1
    tbl_mvmnt.velocity_ankle_joint_center_X(frame_count)=...
        (tbl_mvmnt.ankle_joint_center_X(frame_count+1) ...
        - tbl_mvmnt.ankle_joint_center_X(frame_count-1))...
        / (2 * t_spf);
    tbl_mvmnt.velocity_ankle_joint_center_Y(frame_count)=...
        (tbl_mvmnt.ankle_joint_center_Y(frame_count+1) ...
        - tbl_mvmnt.ankle_joint_center_Y(frame_count-1))...
        / (2 * t_spf);
    tbl_mvmnt.velocity_ankle_joint_center_Z(frame_count)=...
        (tbl_mvmnt.ankle_joint_center_Z(frame_count+1) ...
        - tbl_mvmnt.ankle_joint_center_Z(frame_count-1))...
        / (2 * t_spf);
end

% calculate translational acceleration of ankle joint from position data in
% global coordinate system

for frame_count=2:number_frames-1
   tbl_mvmnt.acceleration_ankle_joint_center_X(frame_count)=...
        (tbl_mvmnt.ankle_joint_center_X(frame_count+1)...
        - 2 * tbl_mvmnt.ankle_joint_center_X(frame_count)...
        + tbl_mvmnt.ankle_joint_center_X(frame_count-1))...
        / (t_spf^2); 
    tbl_mvmnt.acceleration_ankle_joint_center_Y(frame_count)=...
        (tbl_mvmnt.ankle_joint_center_Y(frame_count+1)...
        - 2 * tbl_mvmnt.ankle_joint_center_Y(frame_count)...
        + tbl_mvmnt.ankle_joint_center_Y(frame_count-1))...
        / (t_spf^2);
    tbl_mvmnt.acceleration_ankle_joint_center_Z(frame_count)=...
        (tbl_mvmnt.ankle_joint_center_Z(frame_count+1)...
        - 2 * tbl_mvmnt.ankle_joint_center_Z(frame_count)...
        + tbl_mvmnt.ankle_joint_center_Z(frame_count-1))...
        / (t_spf^2);
end

% KNNE JOINT CENTER POSITION, VELOCITY, & ACCELERATION
% Calculate xyz coordinates of knee joint center in global coordinate
% system
% Define midpoint between lateral and medial femoral condyles along global X axis
tbl_mvmnt.knee_joint_center_X=...
    (tbl_mvmnt.iR_Thigh_R_LAT_FCONX...
    + tbl_mvmnt.iR_Thigh_R_MED_FCONX)/2;
% Define midpoint between lateral and medial femoral condyles along global Y axis
tbl_mvmnt.knee_joint_center_Y=...
    (tbl_mvmnt.iR_Thigh_R_LAT_FCONY...
    + tbl_mvmnt.iR_Thigh_R_MED_FCONY)/2;
% Define midpoint between lateral and medial femoral condyles along the global Z axis
tbl_mvmnt.knee_joint_center_Z=...
    (tbl_mvmnt.iR_Thigh_R_LAT_FCONZ...
    + tbl_mvmnt.iR_Thigh_R_MED_FCONZ)/2;

% calculate translational velocity of knee joint center from position data
% in global cooridinate systsem
% Central Difference technique using knee joint center positional data between frames

for frame_count=2:number_frames-1
    tbl_mvmnt.velocity_knee_joint_center_X(frame_count)=...
        (tbl_mvmnt.knee_joint_center_X(frame_count+1) ...
        - tbl_mvmnt.knee_joint_center_X(frame_count-1))...
        / (2 * t_spf);
     tbl_mvmnt.velocity_knee_joint_center_Y(frame_count)=...
        (tbl_mvmnt.knee_joint_center_Y(frame_count+1) ...
        - tbl_mvmnt.knee_joint_center_Y(frame_count-1))...
        / (2 * t_spf);
    tbl_mvmnt.velocity_knee_joint_center_Z(frame_count)=...
        (tbl_mvmnt.knee_joint_center_Z(frame_count+1) ...
        - tbl_mvmnt.knee_joint_center_Z(frame_count-1))...
        / (2 * t_spf);
end

% Calculate translational acceleration of knee joint from position data in
% global coordinate system

for frame_count=2:number_frames-1
   tbl_mvmnt.acceleration_knee_joint_center_X(frame_count)=...
        (tbl_mvmnt.knee_joint_center_X(frame_count+1)...
        - 2 * tbl_mvmnt.knee_joint_center_X(frame_count)...
        + tbl_mvmnt.knee_joint_center_X(frame_count-1))...
        / (t_spf^2); 
    tbl_mvmnt.acceleration_knee_joint_center_Y(frame_count)=...
        (tbl_mvmnt.knee_joint_center_Y(frame_count+1)...
        - 2 * tbl_mvmnt.knee_joint_center_Y(frame_count)...
        + tbl_mvmnt.knee_joint_center_Y(frame_count-1))...
        / (t_spf^2);
    tbl_mvmnt.acceleration_knee_joint_center_Z(frame_count)=...
        (tbl_mvmnt.knee_joint_center_Z(frame_count+1)...
        - 2 * tbl_mvmnt.knee_joint_center_Z(frame_count)...
        + tbl_mvmnt.knee_joint_center_Z(frame_count-1))...
        / (t_spf^2);
end

% HIP JOINT CENTER POSITION, VELOCITY, & ACCELERATION
% Calculate xyz coordinates of hip joint center in global coordinate
% system (Bell et al., 1989)
% Define absolute distance between XYZ coordinates of Right ASIS and Left ASIS 
% (represented as distance_RASIS_LASIS)
for frame_count=1:number_frames
    tbl_mvmnt.distance_RASIS_LASIS(frame_count)=...
        norm([tbl_mvmnt.iPelvis_R_ASISX(frame_count) tbl_mvmnt.iPelvis_R_ASISY(frame_count) tbl_mvmnt.iPelvis_R_ASISZ(frame_count)]...
        - [tbl_mvmnt.iPelvis_L_ASISX(frame_count) tbl_mvmnt.iPelvis_L_ASISY(frame_count) tbl_mvmnt.iPelvis_L_ASISZ(frame_count)]); 
end

% Define hip_joint_center_x position in global cooridinate system as the
% sum of x component of RASIS vector and (distance factor*distance_RASIS_LASIS) 
tbl_mvmnt.hip_joint_center_X=tbl_mvmnt.iPelvis_R_ASISX... 
    + 0.14* tbl_mvmnt.distance_RASIS_LASIS;

% Define hip_joint_center_y in global cooridinate system as the
% sum of y component of RASIS vector and (distance factor*distance_RASIS_LASIS)
tbl_mvmnt.hip_joint_center_Y=tbl_mvmnt.iPelvis_R_ASISY... 
    + (-0.3* tbl_mvmnt.distance_RASIS_LASIS);

% Define hip_joint_center_z in global cooridinate system as the
% sum of z component of RASIS vector and (distance factor*distance_RASIS_LASIS) center and RASIS
tbl_mvmnt.hip_joint_center_Z=tbl_mvmnt.iPelvis_R_ASISZ... 
    + (-0.22* tbl_mvmnt.distance_RASIS_LASIS);

% calculate translational velocity of hip joint center from position data
% in global cooridinate systsem

% Central Difference technique using knee joint center positional data between frames

for frame_count=2:number_frames-1
    tbl_mvmnt.velocity_hip_joint_center_X(frame_count)=...
        (tbl_mvmnt.hip_joint_center_X(frame_count+1) ...
        - tbl_mvmnt.hip_joint_center_X(frame_count-1))...
        / (2 * t_spf);
     tbl_mvmnt.velocity_hip_joint_center_Y(frame_count)=...
        (tbl_mvmnt.hip_joint_center_Y(frame_count+1) ...
        - tbl_mvmnt.hip_joint_center_Y(frame_count-1))...
        / (2 * t_spf);
    tbl_mvmnt.velocity_hip_joint_center_Z(frame_count)=...
        (tbl_mvmnt.hip_joint_center_Z(frame_count+1) ...
        - tbl_mvmnt.hip_joint_center_Z(frame_count-1))...
        / (2 * t_spf);
end

% Calculate translational acceleration of hip joint from position data in
% global coordinate system

for frame_count=2:number_frames-1
   tbl_mvmnt.acceleration_hip_joint_center_X(frame_count)=...
        (tbl_mvmnt.hip_joint_center_X(frame_count+1)...
        - 2 * tbl_mvmnt.hip_joint_center_X(frame_count)...
        + tbl_mvmnt.hip_joint_center_X(frame_count-1))...
        / (t_spf^2); 
    tbl_mvmnt.acceleration_hip_joint_center_Y(frame_count)=...
        (tbl_mvmnt.hip_joint_center_Y(frame_count+1)...
        - 2 * tbl_mvmnt.hip_joint_center_Y(frame_count)...
        + tbl_mvmnt.hip_joint_center_Y(frame_count-1))...
        / (t_spf^2);
    tbl_mvmnt.acceleration_hip_joint_center_Z(frame_count)=...
        (tbl_mvmnt.hip_joint_center_Z(frame_count+1)...
        - 2 * tbl_mvmnt.hip_joint_center_Z(frame_count)...
        + tbl_mvmnt.hip_joint_center_Z(frame_count-1))...
        / (t_spf^2);
end


% PLOTS for JOINT CENTER POSITIONS
tpf=tbl_mvmnt.Frame*t_spf;
% ANKLE JOINT CENTER POSITION, VELOCITY, & ACCELERATION 
figure('Name',"Ankle Joint Center Position, Velocity & Acceleration for " + tbl_mvmnt_file_name,'NumberTitle','off')
   subplot(3,3,1)
       plot(tpf,tbl_mvmnt.ankle_joint_center_X)
       title('Ankle Joint Position X')
       xlabel('time (s)')
       ylabel('position (mm)')
   subplot(3,3,2)   
       plot(tpf,tbl_mvmnt.ankle_joint_center_Y)
       title('Ankle Joint Position Y')
       xlabel('time (s)')
       ylabel('position (mm)')
   subplot(3,3,3)
       plot(tpf,tbl_mvmnt.ankle_joint_center_Z)
       title('Ankle Joint Position Z')
       xlabel('time (s)')
       ylabel('position (mm)')
       
   subplot(3,3,4)
       plot(tpf,tbl_mvmnt.velocity_ankle_joint_center_X)
       title('Ankle Joint Velocity X')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
   subplot(3,3,5)   
       plot(tpf,tbl_mvmnt.velocity_ankle_joint_center_Y)
       title('Ankle Joint Velocity Y')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
   subplot(3,3,6)
       plot(tpf,tbl_mvmnt.velocity_ankle_joint_center_Z)
       title('Ankle Joint Velocity Z')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
          
   subplot(3,3,7)
       plot(tpf,tbl_mvmnt.acceleration_ankle_joint_center_X)
       title('Ankle Joint Acceleration X')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
   subplot(3,3,8)   
       plot(tpf,tbl_mvmnt.acceleration_ankle_joint_center_Y)
       title('Ankle Joint Acceleration Y')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
   subplot(3,3,9)
       plot(tpf,tbl_mvmnt.acceleration_ankle_joint_center_Z)
       title('Ankle Joint Acceleration Z')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
saveas(gcf, "ankle_joint_center_p_v_a_" + tbl_mvmnt_file_name + ".png")
       
% KNEE JOINT CENTER POSITION, VELOCITY & ACCELERATION PLOTS
figure('Name',"Knee Joint Center Position, Velocity, & Acceleration for " + tbl_mvmnt_file_name,'NumberTitle','off')
   subplot(3,3,1)
       plot(tpf,tbl_mvmnt.knee_joint_center_X)
       title('Knee Joint Position X')
       xlabel('time (s)')
       ylabel('position (mm)')
   subplot(3,3,2)   
       plot(tpf,tbl_mvmnt.knee_joint_center_Y)
       title('Knee Joint Position Y')
       xlabel('time (s)')
       ylabel('position (mm)')
   subplot(3,3,3)
       plot(tpf,tbl_mvmnt.knee_joint_center_Z)
       title('Knee Joint Position Z')
       xlabel('time (s)')
       ylabel('position (mm)')
       
   subplot(3,3,4)
       plot(tpf,tbl_mvmnt.velocity_knee_joint_center_X)
       title('Knee Joint Velocity X')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
   subplot(3,3,5)   
       plot(tpf,tbl_mvmnt.velocity_knee_joint_center_Y)
       title('Knee Joint Velocity Y')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
   subplot(3,3,6)
       plot(tpf,tbl_mvmnt.velocity_knee_joint_center_Z)
       title('Knee Joint Velocity Z')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
          
   subplot(3,3,7)
       plot(tpf,tbl_mvmnt.acceleration_knee_joint_center_X)
       title('Knee Joint Acceleration X')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
   subplot(3,3,8)   
       plot(tpf,tbl_mvmnt.acceleration_knee_joint_center_Y)
       title('Knee Joint Acceleration Y')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
   subplot(3,3,9)
       plot(tpf,tbl_mvmnt.acceleration_knee_joint_center_Z)
       title('Knee Joint Acceleration Z')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
saveas(gcf, "knee_joint_center_p_v_a_" + tbl_mvmnt_file_name + ".png")
% HIP JOINT CENTER POSITION, VELOCITY, & ACCELERATION PLOTS
figure('Name',"Hip Joint Center Position, Velocity, & Acceleration for " + tbl_mvmnt_file_name,'NumberTitle','off')
   subplot(3,3,1)
       plot(tpf,tbl_mvmnt.hip_joint_center_X)
       title('Hip Joint Position X')
       xlabel('time (s)')
       ylabel('position (mm)')
   subplot(3,3,2)   
       plot(tpf,tbl_mvmnt.hip_joint_center_Y)
       title('Hip Joint Position Y')
       xlabel('time (s)')
       ylabel('position (mm)')
   subplot(3,3,3)
       plot(tpf,tbl_mvmnt.hip_joint_center_Z)
       title('Hip Joint Position Z')
       xlabel('time (s)')
       ylabel('position (mm)')
       
   subplot(3,3,4)
       plot(tpf,tbl_mvmnt.velocity_hip_joint_center_X)
       title('Hip Joint Velocity X')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
   subplot(3,3,5)   
       plot(tpf,tbl_mvmnt.velocity_hip_joint_center_Y)
       title('Hip Joint Velocity Y')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
   subplot(3,3,6)
       plot(tpf,tbl_mvmnt.velocity_hip_joint_center_Z)
       title('Hip Joint Velocity Z')
       xlabel('time (s)')
       ylabel('velocity (mm/s)')
          
   subplot(3,3,7)
       plot(tpf,tbl_mvmnt.acceleration_hip_joint_center_X)
       title('Hip Joint Acceleration X')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
   subplot(3,3,8)   
       plot(tpf,tbl_mvmnt.acceleration_hip_joint_center_Y)
       title('Hip Joint Acceleration Y')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
   subplot(3,3,9)
       plot(tpf,tbl_mvmnt.acceleration_hip_joint_center_Z)
       title('Hip Joint Acceleration Z')
       xlabel('time (s)')
       ylabel('acceleration (mm/s^2)')
saveas(gcf, "hip_joint_center_p_v_a_" + tbl_mvmnt_file_name + ".png")       

% LOCAL COORDINATE SYSTEM OF FOOT
% Set origin to heel marker/calcaneus
% Define x axis along quasi-sagittal plane
% Define temporary vector between the first metatarsal and the calcaneus
tbl_mvmnt.temp_calc_1MT_x=(tbl_mvmnt.iL_Foot_3_Mrk_R_1ST_HMTX...
    -tbl_mvmnt.iL_Foot_3_Mrk_R_HEELX);
tbl_mvmnt.temp_calc_1MT_y=(tbl_mvmnt.iL_Foot_3_Mrk_R_1ST_HMTY...
    -tbl_mvmnt.iL_Foot_3_Mrk_R_HEELY);
tbl_mvmnt.temp_calc_1MT_z=(tbl_mvmnt.iL_Foot_3_Mrk_R_1ST_HMTZ...
    -tbl_mvmnt.iL_Foot_3_Mrk_R_HEELZ);
% Define temporary vector between the fifth metatarsal and the calcaneus
tbl_mvmnt.temp_calc_5MT_x=(tbl_mvmnt.iL_Foot_3_Mrk_R_5TH_BMTX...
    - tbl_mvmnt.iL_Foot_3_Mrk_R_HEELX);
tbl_mvmnt.temp_calc_5MT_y=(tbl_mvmnt.iL_Foot_3_Mrk_R_5TH_BMTY...
    - tbl_mvmnt.iL_Foot_3_Mrk_R_HEELY);
tbl_mvmnt.temp_calc_5MT_z=(tbl_mvmnt.iL_Foot_3_Mrk_R_5TH_BMTZ...
    - tbl_mvmnt.iL_Foot_3_Mrk_R_HEELZ);


% Calculate x axis as cross product of temp_calc_1MT and temp_calc_5MT
for frame_count=1:number_frames
    temp_calc_1MT=[tbl_mvmnt.temp_calc_1MT_x(frame_count) tbl_mvmnt.temp_calc_1MT_y(frame_count) tbl_mvmnt.temp_calc_1MT_z(frame_count)];
    temp_calc_5MT=[tbl_mvmnt.temp_calc_5MT_x(frame_count) tbl_mvmnt.temp_calc_5MT_y(frame_count) tbl_mvmnt.temp_calc_5MT_z(frame_count)];
    foot_x_axis=cross(temp_calc_5MT,temp_calc_1MT);
    
    tbl_mvmnt.foot_x_axis_x(frame_count)=foot_x_axis(1);
    tbl_mvmnt.foot_x_axis_y(frame_count)=foot_x_axis(2);
    tbl_mvmnt.foot_x_axis_z(frame_count)=foot_x_axis(3);
end

% Define temporary vector between the calcaneus and second toe
tbl_mvmnt.temp_calc_2toe_x=(tbl_mvmnt.iL_Foot_3_Mrk_R_TOEX...
    - tbl_mvmnt.iL_Foot_3_Mrk_R_HEELX);
tbl_mvmnt.temp_calc_2toe_y=(tbl_mvmnt.iL_Foot_3_Mrk_R_TOEY...
    - tbl_mvmnt.iL_Foot_3_Mrk_R_HEELY);
tbl_mvmnt.temp_calc_2toe_z=(tbl_mvmnt.iL_Foot_3_Mrk_R_TOEZ...
    - tbl_mvmnt.iL_Foot_3_Mrk_R_HEELZ);

% Define z axis along quasi-transverse plane
% Calculate z axis as cross product between temp_calc_2toe and x axis
for frame_count=1:number_frames
    temp_calc_2toe=[tbl_mvmnt.temp_calc_2toe_x(frame_count) tbl_mvmnt.temp_calc_2toe_y(frame_count) tbl_mvmnt.temp_calc_2toe_z(frame_count)];
    foot_x_axis=[tbl_mvmnt.foot_x_axis_x(frame_count)  tbl_mvmnt.foot_x_axis_y(frame_count) tbl_mvmnt.foot_x_axis_z(frame_count)];
    foot_z_axis=cross(temp_calc_2toe,foot_x_axis);
    
    tbl_mvmnt.foot_z_axis_x(frame_count)=foot_z_axis(1);
    tbl_mvmnt.foot_z_axis_y(frame_count)=foot_z_axis(2);
    tbl_mvmnt.foot_z_axis_z(frame_count)=foot_z_axis(3);
end

% Define y axis along intersection of the quasi-sagittal plane, defined by
% x axis, and quasi-transverse plane, defined by z axis
% Calculate y axis as the cross product between the x and z axes
for frame_count=1:number_frames
    foot_z_axis=[tbl_mvmnt.foot_z_axis_x(frame_count) tbl_mvmnt.foot_z_axis_y(frame_count) tbl_mvmnt.foot_z_axis_z(frame_count)];
    foot_x_axis=[tbl_mvmnt.foot_x_axis_x(frame_count) tbl_mvmnt.foot_x_axis_y(frame_count) tbl_mvmnt.foot_x_axis_z(frame_count)];
    foot_y_axis=cross(foot_z_axis,foot_x_axis);
    
    tbl_mvmnt.foot_y_axis_x(frame_count)=foot_y_axis(1);
    tbl_mvmnt.foot_y_axis_y(frame_count)=foot_y_axis(2);
    tbl_mvmnt.foot_y_axis_z(frame_count)=foot_y_axis(3);
end

% Create unit vectors along defined foot axes
for frame_count=1:number_frames
    % Define foot axis vectors
    foot_x_axis=[tbl_mvmnt.foot_x_axis_x(frame_count) tbl_mvmnt.foot_x_axis_y(frame_count) tbl_mvmnt.foot_x_axis_z(frame_count)];
    foot_y_axis=[tbl_mvmnt.foot_y_axis_x(frame_count) tbl_mvmnt.foot_y_axis_y(frame_count) tbl_mvmnt.foot_y_axis_z(frame_count)];
    foot_z_axis=[tbl_mvmnt.foot_z_axis_x(frame_count) tbl_mvmnt.foot_z_axis_y(frame_count) tbl_mvmnt.foot_z_axis_z(frame_count)];
    
    % calculate vector norms for each foot axis
    foot_x_norm=norm(foot_x_axis);
    foot_y_norm=norm(foot_y_axis);
    foot_z_norm=norm(foot_z_axis);
   
    % cacluate the foot i unit vectors using the xyz components of the foot axis
    % vectors divided by the foot x vector norm
    tbl_mvmnt.foot_i_x(frame_count)=tbl_mvmnt.foot_x_axis_x(frame_count) / foot_x_norm; 
    tbl_mvmnt.foot_i_y(frame_count)=tbl_mvmnt.foot_x_axis_y(frame_count) / foot_x_norm;
    tbl_mvmnt.foot_i_z(frame_count)=tbl_mvmnt.foot_x_axis_z(frame_count) / foot_x_norm;
    
    % cacluate the foot j unit vectors using the xyz components of the foot axis
    % vectors divided by the foot y vector norm
    tbl_mvmnt.foot_j_x(frame_count)=tbl_mvmnt.foot_y_axis_x(frame_count) / foot_y_norm; 
    tbl_mvmnt.foot_j_y(frame_count)=tbl_mvmnt.foot_y_axis_y(frame_count) / foot_y_norm;
    tbl_mvmnt.foot_j_z(frame_count)=tbl_mvmnt.foot_y_axis_z(frame_count) / foot_y_norm;
    
    % cacluate the foot k unit vectors using the xyz components of the foot axis
    % vectors divided by the foot z vector norm
    tbl_mvmnt.foot_k_x(frame_count)=tbl_mvmnt.foot_z_axis_x(frame_count) / foot_z_norm; 
    tbl_mvmnt.foot_k_y(frame_count)=tbl_mvmnt.foot_z_axis_y(frame_count) / foot_z_norm;
    tbl_mvmnt.foot_k_z(frame_count)=tbl_mvmnt.foot_z_axis_z(frame_count) / foot_z_norm;
end

% LOCAL COORDINATE SYSTEM OF SHANK
% Set origin to ankle joint center
% Define shank y axis as vector between tibial condyle center and ankle
% joint center
% Define xyz coorinates of tibial condyle center
tbl_mvmnt.tibial_condyle_center_X=(tbl_mvmnt.iR_Shank_R_LAT_TCONX + tbl_mvmnt.iR_Shank_R_MED_TCONX)/2;
tbl_mvmnt.tibial_condyle_center_Y=(tbl_mvmnt.iR_Shank_R_LAT_TCONY + tbl_mvmnt.iR_Shank_R_MED_TCONY)/2;
tbl_mvmnt.tibial_condyle_center_Z=(tbl_mvmnt.iR_Shank_R_LAT_TCONZ + tbl_mvmnt.iR_Shank_R_MED_TCONZ)/2;

tbl_mvmnt.shank_y_axis_x=(tbl_mvmnt.tibial_condyle_center_X - tbl_mvmnt.ankle_joint_center_X);
tbl_mvmnt.shank_y_axis_y=(tbl_mvmnt.tibial_condyle_center_Y - tbl_mvmnt.ankle_joint_center_Y);
tbl_mvmnt.shank_y_axis_z=(tbl_mvmnt.tibial_condyle_center_Z - tbl_mvmnt.ankle_joint_center_Z);

% Define shank temporary vector between medial tibial condyle and ankle joint
% center
tbl_mvmnt.shank_temp_x=(tbl_mvmnt.iR_Shank_R_MED_TCONX - tbl_mvmnt.ankle_joint_center_X);
tbl_mvmnt.shank_temp_y=(tbl_mvmnt.iR_Shank_R_MED_TCONY - tbl_mvmnt.ankle_joint_center_Y);
tbl_mvmnt.shank_temp_z=(tbl_mvmnt.iR_Shank_R_MED_TCONZ - tbl_mvmnt.ankle_joint_center_Z);

% Define shank x axis as perpendicular to the frontal plane (plane
% containing ankle joint center, medial and lateral condyles)
% Define shank x axis as cross product of shank y axis and temporary vector
for frame_count=1:number_frames
    shank_y_axis=[tbl_mvmnt.shank_y_axis_x(frame_count) tbl_mvmnt.shank_y_axis_y(frame_count) tbl_mvmnt.shank_y_axis_z(frame_count)];
    shank_temp=[tbl_mvmnt.shank_temp_x(frame_count) tbl_mvmnt.shank_temp_y(frame_count) tbl_mvmnt.shank_temp_z(frame_count)];
    shank_x_axis=cross(shank_temp,shank_y_axis);
    
    tbl_mvmnt.shank_x_axis_x(frame_count)=shank_x_axis(1);
    tbl_mvmnt.shank_x_axis_y(frame_count)=shank_x_axis(2);
    tbl_mvmnt.shank_x_axis_z(frame_count)=shank_x_axis(3);
end

% Define shank z axis as perpendicular to the shank x and y axes
% Define shank z axis as cross product between shank y and x axes
for frame_count=1:number_frames
    shank_y_axis=[tbl_mvmnt.shank_y_axis_x(frame_count) tbl_mvmnt.shank_y_axis_y(frame_count) tbl_mvmnt.shank_y_axis_z(frame_count)];
    shank_x_axis=[tbl_mvmnt.shank_x_axis_x(frame_count) tbl_mvmnt.shank_x_axis_y(frame_count) tbl_mvmnt.shank_x_axis_z(frame_count)];
    shank_z_axis=cross(shank_x_axis,shank_y_axis);
    
    tbl_mvmnt.shank_z_axis_x(frame_count)=shank_z_axis(1);
    tbl_mvmnt.shank_z_axis_y(frame_count)=shank_z_axis(2);
    tbl_mvmnt.shank_z_axis_z(frame_count)=shank_z_axis(3);
end

% Create unit vectors along defined shank axes
for frame_count=1:number_frames
    % Define shank axis vectors
    shank_x_axis=[tbl_mvmnt.shank_x_axis_x(frame_count) tbl_mvmnt.shank_x_axis_y(frame_count) tbl_mvmnt.shank_x_axis_z(frame_count)];
    shank_y_axis=[tbl_mvmnt.shank_y_axis_x(frame_count) tbl_mvmnt.shank_y_axis_y(frame_count) tbl_mvmnt.shank_y_axis_z(frame_count)];
    shank_z_axis=[tbl_mvmnt.shank_z_axis_x(frame_count) tbl_mvmnt.shank_z_axis_y(frame_count) tbl_mvmnt.shank_z_axis_z(frame_count)];
    
    % calculate vector norms for each shank axis
    shank_x_norm=norm(shank_x_axis);
    shank_y_norm=norm(shank_y_axis);
    shank_z_norm=norm(shank_z_axis);
   
    % cacluate the shank i unit vectors using the xyz components of the shank axis
    % vectors divided by the shank x vector norm
    tbl_mvmnt.shank_i_x(frame_count)=tbl_mvmnt.shank_x_axis_x(frame_count) / shank_x_norm; 
    tbl_mvmnt.shank_i_y(frame_count)=tbl_mvmnt.shank_x_axis_y(frame_count) / shank_x_norm;
    tbl_mvmnt.shank_i_z(frame_count)=tbl_mvmnt.shank_x_axis_z(frame_count) / shank_x_norm;
    
    % cacluate the shank j unit vectors using the xyz components of the shank axis
    % vectors divided by the shank y vector norm
    tbl_mvmnt.shank_j_x(frame_count)=tbl_mvmnt.shank_y_axis_x(frame_count) / shank_y_norm; 
    tbl_mvmnt.shank_j_y(frame_count)=tbl_mvmnt.shank_y_axis_y(frame_count) / shank_y_norm;
    tbl_mvmnt.shank_j_z(frame_count)=tbl_mvmnt.shank_y_axis_z(frame_count) / shank_y_norm;
    
    % cacluate the shank k unit vectors using the xyz components of the shank axis
    % vectors divided by the shank z vector norm
    tbl_mvmnt.shank_k_x(frame_count)=tbl_mvmnt.shank_z_axis_x(frame_count) / shank_z_norm; 
    tbl_mvmnt.shank_k_y(frame_count)=tbl_mvmnt.shank_z_axis_y(frame_count) / shank_z_norm;
    tbl_mvmnt.shank_k_z(frame_count)=tbl_mvmnt.shank_z_axis_z(frame_count) / shank_z_norm;
end

% LOCAL COORDINATE SYSTEM OF THIGH
% Define the origin at the hip joint center
% Define thigh y axis vector between hip joint center and knee joint center
tbl_mvmnt.thigh_y_axis_x=(tbl_mvmnt.hip_joint_center_X - tbl_mvmnt.knee_joint_center_X);
tbl_mvmnt.thigh_y_axis_y=(tbl_mvmnt.hip_joint_center_Y - tbl_mvmnt.knee_joint_center_Y);
tbl_mvmnt.thigh_y_axis_z=(tbl_mvmnt.hip_joint_center_Z - tbl_mvmnt.knee_joint_center_Z);

% Define thigh temporary vector between knee joint center and medial
% femoral condyle
tbl_mvmnt.thigh_temp_x=(tbl_mvmnt.iR_Thigh_R_MED_FCONX - tbl_mvmnt.knee_joint_center_X);
tbl_mvmnt.thigh_temp_y=(tbl_mvmnt.iR_Thigh_R_MED_FCONY - tbl_mvmnt.knee_joint_center_Y);
tbl_mvmnt.thigh_temp_z=(tbl_mvmnt.iR_Thigh_R_MED_FCONZ - tbl_mvmnt.knee_joint_center_Z);

% Define thigh x axis as cross product of thigh temporary vector and y axis
for frame_count=1:number_frames
    thigh_y_axis=[tbl_mvmnt.thigh_y_axis_x(frame_count) tbl_mvmnt.thigh_y_axis_y(frame_count) tbl_mvmnt.thigh_y_axis_z(frame_count)];
    thigh_temp=[tbl_mvmnt.thigh_temp_x(frame_count) tbl_mvmnt.thigh_temp_y(frame_count) tbl_mvmnt.thigh_temp_z(frame_count)];
   
    thigh_x_axis=cross(thigh_temp,thigh_y_axis);
    
    tbl_mvmnt.thigh_x_axis_x(frame_count)=thigh_x_axis(1);
    tbl_mvmnt.thigh_x_axis_y(frame_count)=thigh_x_axis(2);
    tbl_mvmnt.thigh_x_axis_z(frame_count)=thigh_x_axis(3);
end

% Define thigh z axis as perpendicular to the thigh y and x axes
% Define thigh z axis as cross product between the thigh x and y axes
for frame_count=1:number_frames
    thigh_x_axis=[tbl_mvmnt.thigh_x_axis_x(frame_count) tbl_mvmnt.thigh_x_axis_y(frame_count) tbl_mvmnt.thigh_x_axis_z(frame_count)];
    thigh_y_axis=[tbl_mvmnt.thigh_y_axis_x(frame_count) tbl_mvmnt.thigh_y_axis_y(frame_count) tbl_mvmnt.thigh_y_axis_z(frame_count)];
    thigh_z_axis=cross(thigh_x_axis,thigh_y_axis);
    
    tbl_mvmnt.thigh_z_axis_x(frame_count)=thigh_z_axis(1);
    tbl_mvmnt.thigh_z_axis_y(frame_count)=thigh_z_axis(2);
    tbl_mvmnt.thigh_z_axis_z(frame_count)=thigh_z_axis(3);
end

% Create unit vectors along defined thigh axes
for frame_count=1:number_frames
    % Define thigh axes vectors
    thigh_x_axis=[tbl_mvmnt.thigh_x_axis_x(frame_count) tbl_mvmnt.thigh_x_axis_y(frame_count) tbl_mvmnt.thigh_x_axis_z(frame_count)];
    thigh_y_axis=[tbl_mvmnt.thigh_y_axis_x(frame_count) tbl_mvmnt.thigh_y_axis_y(frame_count) tbl_mvmnt.thigh_y_axis_z(frame_count)];
    thigh_z_axis=[tbl_mvmnt.thigh_z_axis_x(frame_count) tbl_mvmnt.thigh_z_axis_y(frame_count) tbl_mvmnt.thigh_z_axis_z(frame_count)];
    
    % calculate vector norms for each thigh axis
    thigh_x_norm=norm(thigh_x_axis);
    thigh_y_norm=norm(thigh_y_axis);
    thigh_z_norm=norm(thigh_z_axis);
   
    % cacluate the thigh i unit vectors using the xyz components of the thigh axis
    % vectors divided by the thigh x vector norm
    tbl_mvmnt.thigh_i_x(frame_count)=tbl_mvmnt.thigh_x_axis_x(frame_count) / thigh_x_norm; 
    tbl_mvmnt.thigh_i_y(frame_count)=tbl_mvmnt.thigh_x_axis_y(frame_count) / thigh_x_norm;
    tbl_mvmnt.thigh_i_z(frame_count)=tbl_mvmnt.thigh_x_axis_z(frame_count) / thigh_x_norm;
    
    % cacluate the thigh j unit vectors using the xyz components of the thigh axis
    % vectors divided by the thigh y vector norm
    tbl_mvmnt.thigh_j_x(frame_count)=tbl_mvmnt.thigh_y_axis_x(frame_count) / thigh_y_norm; 
    tbl_mvmnt.thigh_j_y(frame_count)=tbl_mvmnt.thigh_y_axis_y(frame_count) / thigh_y_norm;
    tbl_mvmnt.thigh_j_z(frame_count)=tbl_mvmnt.thigh_y_axis_z(frame_count) / thigh_y_norm;
    
    % cacluate the thigh k unit vectors using the xyz components of the thigh axis
    % vectors divided by the thigh z vector norm
    tbl_mvmnt.thigh_k_x(frame_count)=tbl_mvmnt.thigh_z_axis_x(frame_count) / thigh_z_norm; 
    tbl_mvmnt.thigh_k_y(frame_count)=tbl_mvmnt.thigh_z_axis_y(frame_count) / thigh_z_norm;
    tbl_mvmnt.thigh_k_z(frame_count)=tbl_mvmnt.thigh_z_axis_z(frame_count) / thigh_z_norm;
end

% FOOT SEGMENT ANGLES
% calculate foot segment euler angles using ZYX rotation matrix and direction
% cosine matrix values
tbl_mvmnt.foot_beta=asin(- tbl_mvmnt.foot_i_z);
for frame_count=1:number_frames
    tbl_mvmnt.foot_alpha(frame_count)=asin(tbl_mvmnt.foot_i_y(frame_count)/cos(tbl_mvmnt.foot_beta(frame_count)));
    tbl_mvmnt.foot_gamma(frame_count)=asin(tbl_mvmnt.foot_j_z(frame_count)/cos(tbl_mvmnt.foot_beta(frame_count)));
end

% SHANK SEGMENT ANGLES
% calculate shank segment euler angles using ZYX rotation matrix and direction
% cosine matrix values
tbl_mvmnt.shank_beta=asin(- tbl_mvmnt.shank_i_z);
for frame_count=1:number_frames
    tbl_mvmnt.shank_alpha(frame_count)=asin(tbl_mvmnt.shank_i_y(frame_count)/cos(tbl_mvmnt.shank_beta(frame_count)));
    tbl_mvmnt.shank_gamma(frame_count)=asin(tbl_mvmnt.shank_j_z(frame_count)/cos(tbl_mvmnt.shank_beta(frame_count)));  
end

% THIGH SEGMENT ANGLES
% calculate thigh segment euler angles using ZYX rotation matrix and direction
% cosine matrix values
tbl_mvmnt.thigh_beta=asin(- tbl_mvmnt.thigh_i_z);
for frame_count=1:number_frames
    tbl_mvmnt.thigh_alpha(frame_count)=asin(tbl_mvmnt.thigh_i_y(frame_count)/cos(tbl_mvmnt.thigh_beta(frame_count)));
    tbl_mvmnt.thigh_gamma(frame_count)=asin(tbl_mvmnt.thigh_j_z(frame_count)/cos(tbl_mvmnt.thigh_beta(frame_count)));
end

% FOOT RELATIVE TO SHANK INTERSEGMENTAL ANGLE
for frame_count=1:number_frames
    % Define foot to global rotation matrix
    % Foot ijk vectors form the matrix colmumns
    temp_foot_i = [tbl_mvmnt.foot_i_x(frame_count); tbl_mvmnt.foot_i_y(frame_count); tbl_mvmnt.foot_i_z(frame_count)];
    temp_foot_j = [tbl_mvmnt.foot_j_x(frame_count); tbl_mvmnt.foot_j_y(frame_count); tbl_mvmnt.foot_j_z(frame_count)];
    temp_foot_k = [tbl_mvmnt.foot_k_x(frame_count); tbl_mvmnt.foot_k_y(frame_count); tbl_mvmnt.foot_k_z(frame_count)];
    temp_foot_rotm_global=[temp_foot_i temp_foot_j temp_foot_k];
    
    % Define global to shank rotation matrix
    % Shank ijk vectors form matrix rows
    temp_shank_i = [tbl_mvmnt.shank_i_x(frame_count) tbl_mvmnt.shank_i_y(frame_count) tbl_mvmnt.shank_i_z(frame_count)];
    temp_shank_j = [tbl_mvmnt.shank_j_x(frame_count) tbl_mvmnt.shank_j_y(frame_count) tbl_mvmnt.shank_j_z(frame_count)];
    temp_shank_k = [tbl_mvmnt.shank_k_x(frame_count) tbl_mvmnt.shank_k_y(frame_count) tbl_mvmnt.shank_k_z(frame_count)];
    temp_global_rotm_shank = [temp_shank_i; temp_shank_j; temp_shank_k];
    
    % Define foot to shank rotation matrix
    % multiply global to shank matrix by foot to global matrix
    temp_foot_rotm_shank=temp_global_rotm_shank*temp_foot_rotm_global;
    
    % Calculate Intersegmental Euler Angles for foot relative to shank and
    % store in table
    tbl_mvmnt.foot_shank_beta(frame_count)=...
        asin(- temp_foot_rotm_shank(1,3));
    tbl_mvmnt.foot_shank_alpha(frame_count)=...
        asin(temp_foot_rotm_shank(1,2)/cos(tbl_mvmnt.foot_shank_beta(frame_count)));
    tbl_mvmnt.foot_shank_gamma(frame_count)=...
        asin(temp_foot_rotm_shank(2,3)/cos(tbl_mvmnt.foot_shank_beta(frame_count)));
end
   
% THIGH RELATIVE TO SHANK INTERSEGMENTAL ANGLE
for frame_count=1:number_frames
    % Define shank to global rotation matrix
    % shank ijk vectors form the matrix columns
    temp_shank_i=[tbl_mvmnt.shank_i_x(frame_count); tbl_mvmnt.shank_i_y(frame_count); tbl_mvmnt.shank_i_z(frame_count)];
    temp_shank_j=[tbl_mvmnt.shank_j_x(frame_count); tbl_mvmnt.shank_j_y(frame_count); tbl_mvmnt.shank_j_z(frame_count)];
    temp_shank_k=[tbl_mvmnt.shank_k_x(frame_count); tbl_mvmnt.shank_k_y(frame_count); tbl_mvmnt.shank_k_z(frame_count)];
    temp_shank_rotm_global=[temp_shank_i temp_shank_j temp_shank_k];
    
    % Define global to thigh rotation matrix
    % thigh ijk vectors form matrix rows
    temp_thigh_i=[tbl_mvmnt.thigh_i_x(frame_count) tbl_mvmnt.thigh_i_y(frame_count) tbl_mvmnt.thigh_i_z(frame_count)];
    temp_thigh_j=[tbl_mvmnt.thigh_j_x(frame_count) tbl_mvmnt.thigh_j_y(frame_count) tbl_mvmnt.thigh_j_z(frame_count)];
    temp_thigh_k=[tbl_mvmnt.thigh_k_x(frame_count) tbl_mvmnt.thigh_k_y(frame_count) tbl_mvmnt.thigh_k_z(frame_count)];
    temp_global_rotm_thigh=[temp_thigh_i; temp_thigh_j; temp_thigh_k];
    
    % Define shank to thigh rotation matrix
    % multiply global to thigh matrix by shank to global matrix
    temp_shank_rotm_thigh=temp_global_rotm_thigh*temp_shank_rotm_global;
    
    % Calculate Intersegmental Euler Angles for thigh relative to shank and
    % store in table
    tbl_mvmnt.thigh_shank_beta(frame_count)=...
        asin(- temp_shank_rotm_thigh(1,3));
    tbl_mvmnt.thigh_shank_alpha(frame_count)=...
        asin(temp_shank_rotm_thigh(1,2)/cos(tbl_mvmnt.thigh_shank_beta(frame_count)));
    tbl_mvmnt.thigh_shank_gamma(frame_count)=...
        asin(temp_shank_rotm_thigh(2,3)/cos(tbl_mvmnt.thigh_shank_beta(frame_count)));
end

% DEMONSTRATE DIFFERNCE IN TIBIOFEMORAL (KNEE ANGLES) WITH
% ORIGINAL EULER SEQUENCE Rzyx = [Rx][Ry][Rz]
% DIFFERENT EULER SEQUENCE Rxyz = [Rz][Rx][Ry]
% SHANK RELATIVE TO THIGH INTERSEGMENTAL ANGLE
for frame_count=1:number_frames
    % Define shank to global rotation matrix
    % shank ijk vectors form the matrix columns
    temp_shank_i=[tbl_mvmnt.shank_i_x(frame_count); tbl_mvmnt.shank_i_y(frame_count); tbl_mvmnt.shank_i_z(frame_count)];
    temp_shank_j=[tbl_mvmnt.shank_j_x(frame_count); tbl_mvmnt.shank_j_y(frame_count); tbl_mvmnt.shank_j_z(frame_count)];
    temp_shank_k=[tbl_mvmnt.shank_k_x(frame_count); tbl_mvmnt.shank_k_y(frame_count); tbl_mvmnt.shank_k_z(frame_count)];
    temp_shank_rotm_global=[temp_shank_i temp_shank_j temp_shank_k];
    
    % Define globa to thigh rotation matrix
    % thigh ijk vectors form matrix rows
    temp_thigh_i=[tbl_mvmnt.thigh_i_x(frame_count) tbl_mvmnt.thigh_i_y(frame_count) tbl_mvmnt.thigh_i_z(frame_count)];
    temp_thigh_j=[tbl_mvmnt.thigh_j_x(frame_count) tbl_mvmnt.thigh_j_y(frame_count) tbl_mvmnt.thigh_j_z(frame_count)];
    temp_thigh_k=[tbl_mvmnt.thigh_k_x(frame_count) tbl_mvmnt.thigh_k_y(frame_count) tbl_mvmnt.thigh_k_z(frame_count)];
    temp_global_rotm_thigh=[temp_thigh_i; temp_thigh_j; temp_thigh_k];
    
    % Define shank to thigh rotation matrix
    % multiply global to thigh matrix by shank to global matrix
    temp_shank_rotm_thigh=temp_global_rotm_thigh*temp_shank_rotm_global;
    
    % Calculate Intersegmental Euler Angles for thigh relative to shank and
    % store in table
    tbl_mvmnt.thigh_shank_gamma_Rxyz(frame_count)=...
        asin( -temp_shank_rotm_thigh(3,2));
    tbl_mvmnt.thigh_shank_alpha_Rxyz(frame_count)=...
        asin(temp_shank_rotm_thigh(1,2)/cos(tbl_mvmnt.thigh_shank_gamma_Rxyz(frame_count)));
    tbl_mvmnt.thigh_shank_beta_Rxyz(frame_count)=...
        asin(temp_shank_rotm_thigh(3,1)/cos(tbl_mvmnt.thigh_shank_gamma_Rxyz(frame_count)));
end

% NORMALIZE X AXIS TO PERCENT GAIT
% Find gait start frame
frame_count = 2;
while frame_count < number_frames
    if tbl_mvmnt.velocity_ankle_joint_center_Y(frame_count) < 0 ...
        && tbl_mvmnt.velocity_ankle_joint_center_Y(frame_count + 1) > 0
        break
    end
    frame_count = frame_count + 1;
end
start_frame = frame_count;
% save start frame to table to be used in question 5
tbl_mvmnt.start_frame(1) = frame_count;

%Find gait end frame
frame_count = frame_count + 1;
while frame_count < number_frames
    if tbl_mvmnt.velocity_ankle_joint_center_Y(frame_count) < 0 ...
        && tbl_mvmnt.velocity_ankle_joint_center_Y(frame_count + 1) > 0
        break
    end
    frame_count = frame_count + 1;
end
end_frame = frame_count;
% save end frame to table to be used in question 5
tbl_mvmnt.end_frame(1) = frame_count;

% Set x axis for plots as percent_gait
% Creating a vector with same number of elements as the number of frames
% corresponding to one gait cycle
% Creating a range between each element so that spans from 0 - 100
percent_gait = linspace(0,100,(end_frame - start_frame) + 1);

% PLOTS for SEGMENT ANGLES
% Plot frames corresponding to one gait cycle (from start_frame to
% end_frame)
figure('Name',"Segment Angles vs. time for " + tbl_mvmnt_file_name,'NumberTitle','off')
   subplot(3,3,1)
       plot(percent_gait,rad2deg(tbl_mvmnt.foot_gamma(start_frame:end_frame)))
       title('Foot Gamma')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,3,2)   
       plot(percent_gait,rad2deg(tbl_mvmnt.foot_beta(start_frame:end_frame)))
       title('Foot Beta')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,3,3)
       plot(percent_gait,rad2deg(tbl_mvmnt.foot_alpha(start_frame:end_frame)))
       title('Foot Alpha')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
       
   subplot(3,3,4)
       plot(percent_gait,rad2deg(tbl_mvmnt.shank_gamma(start_frame:end_frame)))
       title('Shank Gamma')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,3,5)   
       plot(percent_gait,rad2deg(tbl_mvmnt.shank_beta(start_frame:end_frame)))
       title('Shank Beta')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,3,6)
       plot(percent_gait,rad2deg(tbl_mvmnt.shank_alpha(start_frame:end_frame)))
       title('Shank Alpha')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')       
  
   subplot(3,3,7)
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_gamma(start_frame:end_frame)))
       title('Thigh Gamma')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,3,8)   
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_beta(start_frame:end_frame)))
       title('Thigh Beta')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,3,9)
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_alpha(start_frame:end_frame)))
       title('Thigh Alpha')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
saveas(gcf, "thigh_shank_foot_seg_angles_" + tbl_mvmnt_file_name + ".png")

% PLOT FOOT RELATIVE TO SHANK INTERSEGMENTAL ANGLES
% Plot frames corresponding to one gait cycle (from start_frame to
% end_frame)
figure('Name',"Foot Relative to Shank Intersegment Angles vs. time for " + tbl_mvmnt_file_name,'NumberTitle','off')
   subplot(3,1,1)
       plot(percent_gait,rad2deg(tbl_mvmnt.foot_shank_gamma(start_frame:end_frame)))
       title('Intersegmental Rotation (Gamma)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,1,2)   
       plot(percent_gait,rad2deg(tbl_mvmnt.foot_shank_beta(start_frame:end_frame)))
       title('Intersegmental Inversion/Eversion (Beta)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,1,3)
       plot(percent_gait,rad2deg(tbl_mvmnt.foot_shank_alpha(start_frame:end_frame)))
       title('Intersegmental Dorsi/Plantar flexion (Alpha)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
saveas(gcf, "foot_vs_shank_int_seg_angles_" + tbl_mvmnt_file_name + ".png")
       
% PLOT SHANK RELATIVE TO THIGH INTERSEGMENTAL ANGLES
% Plot frames corresponding to one gait cycle (from start_frame to
% end_frame)
figure('Name',"Shank Relative to Thigh Intersegment Angles for " + tbl_mvmnt_file_name,'NumberTitle','off')
   subplot(3,1,1)
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_gamma(start_frame:end_frame)))
       title('Intersegmental Rotation (Gamma)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,1,2)   
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_beta(start_frame:end_frame)))
       title('Intersegmental Varus/Valgus (Beta)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,1,3)
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_alpha(start_frame:end_frame)))
       title('Intersegmental Flexion/Extension (Alpha)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
saveas(gcf, "shank_vs_thigh_int_seg_angle_" + tbl_mvmnt_file_name + ".png")
       
% PLOT TWO EULER CONVENTIONS TOGETHER FOR COMPARISON
% Plot frames corresponding to one gait cycle (from start_frame to
% end_frame)
figure('Name',"Tibiofemoral angles with differing Euler Conventions for " + tbl_mvmnt_file_name ,'NumberTitle','off')
   subplot(3,1,1)
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_gamma(start_frame:end_frame)))
       hold on
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_gamma_Rxyz(start_frame:end_frame)))
       hold off
       legend('Rzyx','Rxyz')
       title('Intersegmental Rotation (Gamma)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)') 
   subplot(3,1,2)   
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_beta(start_frame:end_frame)))
       hold on
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_beta_Rxyz(start_frame:end_frame)))
       hold off
       legend('Rzyx','Rxyz')
       title('Intersegmental Varus/Valgus (Beta)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
   subplot(3,1,3)
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_alpha(start_frame:end_frame)))
       hold on
       plot(percent_gait,rad2deg(tbl_mvmnt.thigh_shank_alpha_Rxyz(start_frame:end_frame)))
       hold off
       legend('Rzyx','Rxyz')
       title('Intersegmental Flexion/Extension (Alpha)')
       xlabel('% gait')
       ylabel('Euler Angle (deg)')
saveas(gcf, "Rxyz_vs_Rzyx_" + tbl_mvmnt_file_name + ".png")
       
% Export new table to excel
writetable(tbl_mvmnt,strcat(tbl_mvmnt_file_name,'-processed.csv'));


