%% Data
path='imageFolder'
imds= imageDatastore(path, 'IncludeSubfolders',1, 'LabelSource', 'foldernames');
tbl=countEachLabel(imds)

%%
rng(0)

imds=splitEachLabel(imds, 716);
imds.ReadFcn= @readFunctionResNet50;
%
countEachLabel(imds)
[trainingDS, testDS] = splitEachLabel(imds, 573, 143, 'randomize');
%
trainingDS.Labels=categorical(trainingDS.Labels);
testDS.Labels=categorical(testDS.Labels);


%%

miniBatchSize=20;%
numImages=numel(trainingDS.Files);
maxEpochs=18;%
lr=0.0005;
opts=trainingOptions('adam', ...
    'InitialLearnRate', lr,...
    'MaxEpochs', maxEpochs,...
    'MiniBatchSize', miniBatchSize,...
    'LearnRateSchedule', 'piecewise',...
    'LearnRateDropPeriod',6,...
    'LearnRateDropFactor',0.2,...
    'Plots', 'training-progress');

net=trainNetwork(trainingDS, lgraph_1,opts);
%save('trainedNetIn.mat','net')
%save('testDS.mat','testDS')



%%
%
tic
[labels,err_test]=classify(net, testDS, 'MiniBatchSize', 20);
toc

confMat=confusionmat(testDS.Labels, labels);
confMat=bsxfun(@rdivide,confMat,sum(confMat,2));

mean(diag(confMat))
    
