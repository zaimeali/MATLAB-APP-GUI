

%%
% options = trainingOptions('sgdm', ...
%   'MiniBatchSize', 32, ...
%   'InitialLearnRate', 1e-3, ...
%   'MaxEpochs', 10);
optionsStage1 = trainingOptions('sgdm', ...
    'MiniBatchSize', 2, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);
% Options for step 2
optionsStage2 = trainingOptions('sgdm', ...
    'MiniBatchSize', 2, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);
% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MiniBatchSize', 2, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);
% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MiniBatchSize', 2, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);
options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%% Network
% Load a pretrained ResNet-50.
net = resnet50;
lgraph = layerGraph(net);

% Remove the last 3 layers. 
layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
    };
lgraph = removeLayers(lgraph, layersToRemove);

% Specify the number of classes the network should classify.
numClasses = 3;
numClassesPlusBackground = numClasses + 1;

% Define new classification layers.
newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];

% Add new object classification layers.
lgraph = addLayers(lgraph, newLayers);

%%

% Connect the new layers to the network. 
lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');

% Define the number of outputs of the fully connected layer.
numOutputs = 4 * numClasses;

% Create the box regression layers.
boxRegressionLayers = [
    fullyConnectedLayer(numOutputs,'Name','rcnnBoxFC')
    rcnnBoxRegressionLayer('Name','rcnnBoxDeltas')
    ];

% Add the layers to the network.
lgraph = addLayers(lgraph, boxRegressionLayers);

% Connect the regression layers to the layer named 'avg_pool'.
lgraph = connectLayers(lgraph,'avg_pool','rcnnBoxFC');

% Select a feature extraction layer.
featureExtractionLayer = 'activation_40_relu';

% Disconnect the layers attached to the selected feature extraction layer.
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch2a');
lgraph = disconnectLayers(lgraph, featureExtractionLayer,'res5a_branch1');

% Add ROI max pooling layer.
outputSize = [14 14];
roiPool = roiMaxPooling2dLayer(outputSize,'Name','roiPool');
lgraph = addLayers(lgraph, roiPool);

% Connect feature extraction layer to ROI max pooling layer.
lgraph = connectLayers(lgraph, featureExtractionLayer,'roiPool/in');

% Connect the output of ROI max pool to the disconnected layers from above.
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch2a');
lgraph = connectLayers(lgraph, 'roiPool','res5a_branch1');

%%
% Define anchor boxes.
anchorBoxes = [
    16 16
    32 16
    16 32
    ];

% Create the region proposal layer.
proposalLayer = regionProposalLayer(anchorBoxes,'Name','regionProposal');

lgraph = addLayers(lgraph, proposalLayer);

%%
% Number of anchor boxes.
numAnchors = size(anchorBoxes,1);

% Number of feature maps in coming out of the feature extraction layer. 
numFilters = 1024;

rpnLayers = [
    convolution2dLayer(3, numFilters,'padding',[1 1],'Name','rpnConv3x3')
    reluLayer('Name','rpnRelu')
    ];

lgraph = addLayers(lgraph, rpnLayers);

% Connect to RPN to feature extraction layer.
lgraph = connectLayers(lgraph, featureExtractionLayer, 'rpnConv3x3');

%%
% Add RPN classification layers.
rpnClsLayers = [
    convolution2dLayer(1, numAnchors*2,'Name', 'rpnConv1x1ClsScores')
    rpnSoftmaxLayer('Name', 'rpnSoftmax')
    rpnClassificationLayer('Name','rpnClassification')
    ];
lgraph = addLayers(lgraph, rpnClsLayers);

% Connect the classification layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1ClsScores');

%%
% Add RPN regression layers.
rpnRegLayers = [
    convolution2dLayer(1, numAnchors*4, 'Name', 'rpnConv1x1BoxDeltas')
    rcnnBoxRegressionLayer('Name', 'rpnBoxDeltas');
    ];

lgraph = addLayers(lgraph, rpnRegLayers);

% Connect the regression layers to the RPN network.
lgraph = connectLayers(lgraph, 'rpnRelu', 'rpnConv1x1BoxDeltas');

%%
% Connect region proposal network.
lgraph = connectLayers(lgraph, 'rpnConv1x1ClsScores', 'regionProposal/scores');
lgraph = connectLayers(lgraph, 'rpnConv1x1BoxDeltas', 'regionProposal/boxDeltas');

% Connect region proposal layer to roi pooling.
lgraph = connectLayers(lgraph, 'regionProposal', 'roiPool/roi');

%%
fasterRcnn = trainFasterRCNNObjectDetector(gTruth, ...
                                        lgraph, ...
                                        options, ...
                                        'NegativeOverlapRange', [0 0.2],...
                                        'PositiveOverlapRange', [0.7 1], ...
                                        'SmallestImageDimension', 400);

                                    
%%
save fasterRcnn.mat

%%
img = imread('img/test-1.jpg');
% img = imread('img/test.jpg');
GS = rgb2gray(img);  
GS = imresize(GS, [400 450]);
% Code 1
% [bbox, score, label] = detect(rcnn, GS, 'Threshold', 0.80); 
% Code 2
[bbox, score, label] = detect(fasterRcnn, GS, 'Threshold', 0.80); 
% [score, idx] = max(score); 
% bbox = bbox(idx, :); 
% annotation = sprintf('%s: (Confidence = %f)', label(idx), score); 
% I = insertObjectAnnotation(GS, 'rectangle', bbox, annotation); 
% [selectedBbox, selectedScore] = selectStrongestBbox(bbox,score,'OverlapThreshold',0.80, 'RatioType', 'Min'); 
% detectedImg = insertShape(GS, 'Rectangle', bbox);
% I = insertObjectAnnotation(GS,'rectangle',bbox, score);
% val = score > 0.55;
% [score, idx] = find(val); 
% bbox = bbox(idx, :); 
% annotation = sprintf('%s: (Confidence = %f)', label(idx), score); 
% [selectedBbox, selectedScore] = selectStrongestBbox(bbox, score, 'OverlapThreshold', 0.1);
[selectedBbox , selectedScore, selectedLabels] = selectStrongestBboxMulticlass(bbox, score , label, 'OverlapThreshold', 0.1);
% [selectedBbox , selectedScore, labels] = selectStrongestBboxMulticlass(bbox, score , label, 'RatioType', 'Min', 'OverlapThreshold', 0.1);
% [selectedBbox , selectedScore, labels] = selectStrongestBboxMulticlass(bbox, score , label, 'RatioType', 'Min', 'OverlapThreshold', 0.65);
annotations = string(selectedLabels) + ": " + string(selectedScore);
% I = insertObjectAnnotation(GS, 'rectangle', selectedBbox, selectedScore); 
I = insertObjectAnnotation(GS, 'rectangle', selectedBbox, cellstr(annotations)); 
% [selectedBbox,selectedScore] = selectStrongestBbox(bbox,score,'OverlapThreshold',0.85);
% I = insertObjectAnnotation(GS,'rectangle',selectedBbox,selectedScore);
figure
imshow(I)

%%
% img = imread('lap2/image_1.jpg');
% 
% [bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);
% 
% %%
% [score, idx] = max(score);
% 
% bbox = bbox(idx, :);
% annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
% 
% detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);
% 
% figure
% imshow(detectedImg)

%% Video Read
% j = [];
% for i=1:10
%     j(i) = i * 10;
% end

video_filereader = vision.VideoFileReader(fullfile('result.avi'));

ion = [];
index = 1;

while ~isDone(video_freader)
    I = step(video_freader);
   
        % run the detector
    [bbox, score, label] = detect(fasterRcnn, I, 'Threshold', 0.80);
    % save this in a first loop
    % then use other loop to detect
    
    [selectedBbox , selectedScore, selectedLabels] = selectStrongestBboxMulticlass(bbox, score, ...
                                                            label, 'OverlapThreshold', 0.1);
   
        % Insert bounding box detected
%     [scores, idx] = max(selectedScorev);
%     bboxs = selectedBboxv(idx, :);
    confidence = sprintf('%s: %f', selectedLabels, selectedScore); 
%     annotations = string(selectedLabels) + ": " + string(selectedScore);
    I = insertObjectAnnotation(I, 'rectangle', selectedBbox, cellstr(confidence)); 
        %     detected = insertObjectAnnotation(GS, 'rectangle', selectedBboxv, confidence, ...
        %                       'FontSize', 14, 'LineWidth', 3, ...
        %                       'Color', 'green', 'Textcolor', 'black', ...
        %                       'TextBoxOpacity', 0.7);
                 
        % show new frame
    fprintf('%f: %s\n', index, I);
       
    index = index+1;   
end

release(video_freader);

%%
obj = VideoReader('result.avi');
nFrames = obj.Duration * obj.FrameRate;
fprintf('%f\n', nFrames);


for i=1:nFrames
    frame = read(obj, nFrames);
    
    [bbox, score, label] = detect(fasterRcnn, frame, 'Threshold', 0.80, ...
                                    'ExecutionEnvironment', 'gpu');
    
    fprintf('%.1f\n', i);
end



%% Add functions to path 
video_freader = vision.VideoFileReader(fullfile('result.avi'));
video_player = vision.VideoPlayer;
% 
while ~isDone(video_freader)
    I = step(video_freader);
   
        % run the detector
    [bboxv, scorev, labelv] = detect(fasterRcnn, I, 'Threshold', 0.80);
    % save this in a first loop
    % then use other loop to detect
    
%     [selectedBboxv , selectedScorev, selectedLabelsv] = selectStrongestBboxMulticlass(bboxv, scorev, labelv, 'OverlapThreshold', 0.1);
   
        % Insert bounding box detected
%     [scores, idx] = max(selectedScorev);
%     bboxs = selectedBboxv(idx, :);
%     confidence = sprintf('%s: %f', label(idx), scores); 
% %     annotationsv = string(selectedLabelsv) + ": " + string(selectedScorev);
%     I = insertObjectAnnotation(I, 'rectangle', bboxv, scorev); 
        %     detected = insertObjectAnnotation(GS, 'rectangle', selectedBboxv, confidence, ...
        %                       'FontSize', 14, 'LineWidth', 3, ...
        %                       'Color', 'green', 'Textcolor', 'black', ...
        %                       'TextBoxOpacity', 0.7);
                 
        % show new frame
    step(video_player, I);
    pause(0.05);
end

release(video_freader);
release(video_player);


%% Video Gray Scale
% path = 'Video/IMG_1522.mp4';
% 
% implay(path);
% 
% obj = VideoReader(path);
% nFrames = obj.NumberOfFrames;
% vidHeight = obj.Height;
% vidWidth = obj.Width;
% 
% mov(1:nFrames) = struct('cdata', zeros(vidHeight, vidWidth, 1, 'uint8'), 'colormap', []);
% 
% for k = 1:nFrames
%     mov(k).cdata = rgb2gray(read(obj, k));
% end
% 
% implay(mov)

%% Video Player
vidReader = VideoReader('Video/IMG_1522.mp4');
vidPlayer = vision.DeployableVideoPlayer;
i = 1;
results = struct('Boxes',[],'Scores',[]);
%% Video Player
while(hasFrame(vidReader))    
    % GET DATA
    I = readFrame(vidReader);    
    I = rgb2gray(I);  
    GS = imresize(I, [400 450]);
    
    [bbox, score, label] = detect(fasterRcnn, GS, 'Threshold', 0.80); 
    
    [selectedBbox , selectedScore, selectedLabels] = selectStrongestBboxMulticlass(bbox, score , label, 'OverlapThreshold', 0.1);
    annotations = string(selectedLabels) + ": " + string(selectedScore);
    I = insertObjectAnnotation(GS, 'rectangle', selectedBbox, cellstr(annotations));
    step(vidPlayer,I);
    i = i+1;   
end
results = struct2table(results);
release(vidPlayer);


%% Evaluation   => detect(fasterRcnn, GS, 'Threshold', 0.80)
detectionResults = detect(fasterRcnn, gTruth);
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, gTruth);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))


%% Video PreProcessing
video_path = 'Video/';
video_freader = vision.VideoFileReader(fullfile(video_path, 'IMG_1522.mp4'));
video_player = vision.VideoPlayer;
video_detected = vision.VideoFileWriter('result.avi', ...
                 'FrameRate', video_freader.info.VideoFrameRate);

while ~isDone(video_freader)
    frame = step(video_freader);
    GS = rgb2gray(frame);  
    frame = imresize(GS, [400 450]);
    
    step(video_player, frame);
    step(video_detected, frame);
    pause(0.05);
end

release(video_freader);
release(video_player);
release(video_detected);

%% Video Read
vidReader = VideoReader('result.avi');
vidPlayer = vision.DeployableVideoPlayer;

while(hasFrame(vidReader))    
    % GET DATA
    GS = readFrame(vidReader);    
    
%     [bbox, score, label] = detect(fasterRcnn, GS, 'Threshold', 0.80); 
    
%     [selectedBbox , selectedScore, selectedLabels] = selectStrongestBboxMulticlass(bbox, score , label, 'OverlapThreshold', 0.1);
%     annotations = string(selectedLabels) + ": " + string(selectedScore);
%     I = insertObjectAnnotation(GS, 'rectangle', selectedBbox, cellstr(annotations));
%     I = insertObjectAnnotation(GS, 'rectangle', bbox, score);
    step(vidPlayer,GS);
end
release(vidPlayer);

%%

implay('result.avi');


%%
% obj = vision.VideoFileReader(fullfile('result.avi'));
obj = VideoReader('result.avi');
nFrames = obj.Duration * obj.FrameRate;
GetFrame = [];
for j=1:nFrames
    if mod(j, 15)==0
        GetFrame=[GetFrame, j];
    end
end

vidWidth = obj.Width;
vidHeight = obj.Height;
% mov = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
%     'colormap',[]);
% k = 1;
CurFrame = 1;

vidPlayer = vision.DeployableVideoPlayer;
while hasFrame(obj)
    I = readFrame(obj);
%     I = step(obj);
    if ismember(CurFrame, GetFrame)
        [bbox, score, label] = detect(fasterRcnn, I, 'Threshold', 0.90, 'ExecutionEnvironment', 'gpu'); 
        [selectedBbox , selectedScore, selectedLabels] = selectStrongestBboxMulticlass(bbox, score , label, 'RatioType', 'Min', 'OverlapThreshold', 0);
        annotations = string(selectedLabels) + ": " + string(selectedScore);
        GS = insertObjectAnnotation(I, 'rectangle', selectedBbox, cellstr(annotations));
        step(vidPlayer,GS);
%         mov(k).cdata = I;
%         k = k+1;
    end
    CurFrame = CurFrame+1;
end

release(obj);
release(vidPlayer);
%%
% implay(mov);

%%



%%
CurFrame = 0;
while hasFrame(obj)
    CurImage = readFrame(obj);
    CurFrame = CurFrame+1;
    if ismember(CurFrame, GetFrame)
        step(CurImage);
    end
end

%%
while hasFrame(obj)
    frame = readFrame(obj, GetFrame);
    
end