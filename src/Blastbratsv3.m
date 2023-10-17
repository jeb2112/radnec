% Blastbrats2
% Executes commands for loading MR data, gating, detecting, segmenting and
% storing data for BraTS paper
% Modified to segment vessels

% Prepare variables
disp('Blastbrats');
global CC_labeled;
global s;
global t;
global filename;
I_ones = zeros(240,240,sliceend);
p_stack = [];

% Load gold standard manual ROIs and create manual mask
manualmask = niftiread(inputpath + "/" + "BraTS2021_" + casename + "_seg" );
manualmasket = manualmask>3; %enhancing tumor 
manualmasknc = manualmask==1; %necrotic core
manualmasket = double(manualmasket);
manualmasknc = double(manualmasknc);
manualmasktc = manualmasknc + manualmasket; %tumor core

% Load registered Nifti data
t2flair = niftiread(inputpath + "/" + "BraTS2021_" + casename + "_flair_bias.nii" ); 
t1mprage = niftiread(inputpath + "/" + "BraTS2021_" + casename + "_t1ce_bias.nii" );
t1mprage_template = niftiread(outputpath + "/" + "t1mprage_template.nii" );

% Change precision
t2flair = double(t2flair);
t1mprage = double(t1mprage);

% Rescales images to values 0 to 1 and applies brain mask
t2flairstack = rescale(t2flair);
t1mpragestack = rescale(t1mprage);

% Creates a matrix of voxels for normal brain slice
colorimage_normal = cat(3,t1mpragestack(:,:,normalslice),t2flairstack(:,:,normalslice));    

% Gating Routine
Redchannel_normal = colorimage_normal(:,:,1);
Greenchannel_normal = colorimage_normal(:,:,2);

% kmeans to calculate statistics for brain voxels
greens = [Greenchannel_normal(:)];
reds = [Redchannel_normal(:)];
X = cat(2,greens,reds);
rng(1);
[idx,C] = kmeans(X,2);

% Calculate stats for brain cluster
stdRed = std(X(idx==2,2));
stdGreen = std(X(idx==2,1));
meanRed = mean(X(idx==2,2));
meanGreen = mean(X(idx==2,1));

braincluster_size = sld4; %sld4.Value;

redDiff = braincluster_size*stdRed; 
greenDiff = braincluster_size*stdGreen; 

% Define brain gate
h = images.roi.Ellipse(gca,'Center',[meanGreen meanRed],'Semiaxes',[greenDiff redDiff],'Color','y','LineWidth',1,'LabelAlpha',0.5,'InteractionsAllowed','none');
yv = h.Vertices(:,1);
xv = h.Vertices(:,2);

% Compute ROIs based on Gates

metmaskstack = [];
fusionstack = [];
newfusionstack = [];

    greengate = meanGreen+(sld2.Value)*stdGreen;
    yv2 = [greengate 1 1 greengate];
    greengate_count = (greengate-meanGreen)/stdGreen;
    
    redgate = meanRed+(sld3.Value)*stdRed;
    xv2 = [redgate redgate 1 1]; 
    redgate_count = (redgate-meanRed)/stdRed;
   
    %Creates a matrix of voxels for brain slice
    for slice = slicestart:sliceend;  
    
        colorimage = cat(3,t1mpragestack(:,:,slice),t2flairstack(:,:,slice));    

        % Gating Routine
        Redchannel = colorimage(:,:,1);
        Greenchannel = colorimage(:,:,2);
        
        % Applying gate to brain
        gate = inpolygon(Greenchannel,Redchannel,yv2,xv2);
        brain = inpolygon(Greenchannel,Redchannel,yv,xv);
        metmask = gate-brain; 
        metmask = max(metmask,0);

        %se = strel('line',2,0); 
        %metmask = imerode(metmask,se); % erosion added to get rid of non
        %specific small non target voxels removed for ROC paper

        metmaskstack = cat(3,metmaskstack,metmask);
        fusion = imfuse(metmask,t1mprage_template(:,:,slice),'blend');
        fusionstack = cat(3,fusionstack,fusion);
    end

% Calculate connected objects 
CC_labeled = bwlabeln(metmaskstack(:,:,slicestart:sliceend),26); % beta edit on this line
stats = regionprops3(CC_labeled,"Volume","PrincipalAxisLength");
S = regionprops3(CC_labeled,'Centroid');
BB = regionprops3(CC_labeled, 'BoundingBox');

% Display Volume
f1 = figure(1);
s = sliceViewer(fusionstack,"ScaleFactors",[2,2,1]);
movegui(f1,[1200 300]); 

% Ask user input to proceed
group = 'Updates';
pref =  'Conversion';
title = '';
quest = {'Proceed with ROI selection?'};
pbtns = {'Yes','No'};

[pval,tf] = uigetpref(group,pref,title,quest,pbtns)

switch pval
case 'yes'

% Allow user to select the ROI based on Gates

h = drawpoint;

ypos = round(h.Position(2)/2); % divide by ScaleFactor
xpos = round(h.Position(1)/2); % divide by ScaleFactor

objectnumber = CC_labeled(ypos,xpos,s.SliceNumber);

objectmask = ismember(CC_labeled,objectnumber);
objectmask = double(objectmask);

thisBB = BB.BoundingBox(objectnumber,:,:);

% Creates filled in contour of objectmask 

% Creates filled in contour of objectmask 
se = strel('disk',10); 
close_object = imclose(objectmask,se); 
se2 = strel('square',2); %added to imdilate
objectmask_filled = imdilate(close_object,se2);
objectmask_filled = imfill(objectmask_filled);

% Calculates the centre slice of the ROI and z bounds of ROI

centreslice = S.Centroid(objectnumber,3);
centreslice = round(centreslice);
rectZ = thisBB(6);
            
rectslicestart = centreslice - round(rectZ/2);
if rectslicestart < slicestart
   rectslicestart = slicestart;
end
            
rectsliceend = centreslice + round(rectZ/2);
if rectsliceend > sliceend
   rectsliceend = sliceend;
end
            

% Generate overlay of ROI and display

for slice = slicestart:sliceend;  
        newfusion = imfuse(objectmask_filled(:,:,slice),t1mpragestack(:,:,slice),'blend');
        newfusionstack = cat(3,newfusionstack,newfusion);
end

close(f1);
f2 = figure(2);
t = sliceViewer(newfusionstack,"ScaleFactors",[2,2,1]);
movegui(f2,[1200 300]);

% Calculate accuracy statistics for et

groundtruth = manualmasket;
          
segmentation = objectmask;

sums = groundtruth + segmentation;
subs = groundtruth - segmentation;
          
  TP = length(find(sums == 2));
  FP = length(find(subs == -1));
  TN = length(find(sums == 0));
  FN = length(find(subs == 1));
  
  specificity_et = TN/(TN+FP);
  sensitivity_et = TP/(TP+FN);
  dicecoefficient_et = dice(groundtruth,segmentation); 

% Calculate accuracy statistics for tc

groundtruth = manualmasktc;
          
segmentation = objectmask_filled;

sums = groundtruth + segmentation;
subs = groundtruth - segmentation;
          
  TP = length(find(sums == 2));
  FP = length(find(subs == -1));
  TN = length(find(sums == 0));
  FN = length(find(subs == 1));
  
  specificity_tc = TN/(TN+FP);
  sensitivity_tc = TP/(TP+FN);
  dicecoefficient_tc = dice(groundtruth,segmentation); 

% Ask user input to save ROI
group = 'Updates';
pref =  'Conversion';
title = '';
quest = {'Save current ROI?'};
pbtns = {'Yes','No'};

[pval,tf] = uigetpref(group,pref,title,quest,pbtns)

switch pval
    case 'yes'

        % Add time to clock
        cumulative_elapsed_time = cumulative_elapsed_time + toc;

        % Load cumulative t1ce mask
        load(outputpath + "/t1ce.mat" );
        cumulative_t1ce = cumulative_t1ce + objectmask;

        % Calculate DSC for cumulative mask
        similarity = dice(cumulative_t1ce,groundtruth);

        % Derive centre slice image

        centreimage = t1mpragestack(:,:,centreslice);

         % Update the t1mprage_template to mark saved ROI
        t1mprage_template = t1mprage_template - objectmask;

        % Generate line contour of the auto at the centre slice

        autocontour = objectmask_filled(:,:,centreslice);
        autoboundaries = bwboundaries(autocontour);
        b = autoboundaries{1};

        % Generate line contour of manual ROI at the centre slice

        manualcontour = manualmasktc(:,:,centreslice);
        manualboundaries = bwboundaries(manualcontour);
        b2 = manualboundaries{1};

        % Calculate volumes
        manualmask_et_volume = nnz(manualmasket);
        manualmask_tc_volume = nnz(manualmasktc);
        objectmask_filled_volume = nnz(objectmask_filled);

        % Save ROI data
        filename = "t1ce_" + casename;
        save(outputpath + "/" + filename + ".mat",'greengate_count','redgate_count','objectmask','objectmask_filled','manualmasket','manualmasktc','centreimage','specificity_et','sensitivity_et','dicecoefficient_et','specificity_tc','sensitivity_tc','dicecoefficient_tc','b','b2','manualmask_et_volume','manualmask_tc_volume','objectmask_filled_volume','cumulative_elapsed_time');
        niftiwrite(objectmask,outputpath + "/" + filename + ".nii");
        niftiwrite(objectmask_filled,outputpath + "/" + filename + "_filled" + ".nii");
        niftiwrite(manualmasket,outputpath + "/" + filename + "_manualmask_et" + ".nii")
        niftiwrite(manualmasktc,outputpath + "/" + filename + "_manualmask_tc" + ".nii");
        niftiwrite(t1mprage_template,outputpath + "/" + 't1mprage_template.nii');
        close(f2);
        pval = [];
    case 'no'
        close(f2);
        p_stack = [];
end
    
case 'no'
    close(f1);
    p_stack = [];
end



