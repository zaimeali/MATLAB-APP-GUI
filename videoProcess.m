%% Video Process
obj = VideoReader('test.avi');
nFrames = obj.Duration * obj.FrameRate;
GetFrame = [];
for j=1:nFrames
    if mod(j, 5)==0
        GetFrame=[GetFrame, j];
    end
end

vidWidth = obj.Width;
vidHeight = obj.Height;
CurFrame = 1;

vidPlayer = vision.DeployableVideoPlayer;
while hasFrame(obj)
    I = readFrame(obj);
    if ismember(CurFrame, GetFrame)
        [bbox, score, label] = detect(fasterRcnn, I, 'Threshold', 0.90, 'ExecutionEnvironment', 'gpu'); 
        [selectedBbox , selectedScore, selectedLabels] = selectStrongestBboxMulticlass(bbox, score , label, 'RatioType', 'Min', 'OverlapThreshold', 0);
        annotations = string(selectedLabels) + ": " + string(selectedScore);
        GS = insertObjectAnnotation(I, 'rectangle', selectedBbox, cellstr(annotations));
        step(vidPlayer,GS);
    end
    CurFrame = CurFrame+1;
end

release(obj);
release(vidPlayer);