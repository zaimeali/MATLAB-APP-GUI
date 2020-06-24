%% Video PreProcess
video_path = 'Video/';
video_freader = vision.VideoFileReader(fullfile(video_path, 'video25.mp4'));
video_player = vision.VideoPlayer;
video_detected = vision.VideoFileWriter('test.avi', ...
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