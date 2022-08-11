function trainNN(x,y,neuron,trainfce)

t = y;

if trainfce == 1
    trainFcn = 'trainlm';  % Levenberg-Marquardt Backpropagation.
elseif trainfce == 2
    trainFcn = 'trainbr'; % Bayesian Regularizaton.
elseif trainfce == 3
    trainFcn = 'trainscg'; % Scaled Conjugate Gradient
end

hiddenLayerSize = neuron;
net = fitnet(hiddenLayerSize,trainFcn);

net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'mse';  % Mean Squared Error

%psst
net.trainParam.showWindow = false;

% trenovani
[net,tr] = train(net,x,t);

% testovani
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

% validace etc.
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);


% generovani nn fce
genFunction(net,'neuronka','MatrixOnly','yes','ShowLinks','no');
end
