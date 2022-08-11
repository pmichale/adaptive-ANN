%% inputs
disp('Two-layer feed-forward network, sigmoid activation.');
%vyber fce
disp('1. Rastrigin');
disp('2. Schwefel');
vyber = input('Vyber funkci: ');
%pocet bodu
disp('Doporuceny pocet bodu: 10000');
N = input('Zadejte pocet bodu: ');
%pocet neuronu
disp('Doporuceny pocet neuronu: 100');
neuron = input('Zadejte pocet neuronu: ');
% pocet iteraci
disp('Doporuceny pocet iteraci: 10');
iters = input('Zadejte pocet iteraci: ');
%vyber trenovaci fce pro ANN
disp('Vyber trenovacich funkci: ');
disp('1. Levenberg-Marquardt Backpropagation');
disp('2. Bayesian Regularizaton');
disp('3. Scaled Conjugate Gradient');
disp('Doporucena trenovaci funkce: 1');
trainfce = input('Vyberte trenovaci funkci: ');
% trainfce = 1; % Levenberg-Marquardt Backpropagation.
% trainfce = 2; % Bayesian Regularizaton.
% trainfce = 3; % Scaled Conjugate Gradient

%vyber fce
if vyber == 1
    fce = @rastr;
    interval = 5.12;
elseif vyber == 2
    fce = @schwef;
    interval = 500;
else
    disp('Nebyl spravne zadan nazve funkce. Program ukoncen.');
    return
end
tic
%% program

%generovani bodu
x1 = -interval + (interval+interval).*rand(1,N);
x2 = -interval + (interval+interval).*rand(1,N);
xx = [x1;x2];
x = xx;
y = zeros(1,N);

% ziskani fcnich hodnot
for i = 1:N
    xx(3,i) = fce(x(:,i));
    y(i) = xx(3,i);
end

%prealokace
results_iter_ = zeros(4,iters);
x_only_new = zeros(2,iters);
y_only_new = zeros(1,iters);

for it = 1:iters
    iter_strg = strcat('Iterace:',{' '},num2str(it),{' '},'z',{' '},...
        num2str(iters));
    disp(iter_strg);
    disp('Trenovani site...')
%trenovani NN
trainNN(x,y,neuron,trainfce);
disp('Trenovani dokonceno')
vystup = neuronka(x);
disp('Hledani minima...')
%hledani minima
problem = createOptimProblem('fmincon',...
    'objective',@(x)neuronka([x(1);x(2)]),...
    'lb',[-interval,-interval],'ub',[interval,interval],...
    'x0',[-interval,interval],'options',...
    optimoptions(@fmincon,'Algorithm','sqp','Display','off'));
gs = GlobalSearch('Display','iter');
% gs = GlobalSearch();
rng(14,'twister');
[x_min,y_min] = run(gs,problem);

% ziskani vysledku z NN
vystup_NN = neuronka(x);
errarr = [x;y;vystup_NN];
chyba = errarr(3,:)-errarr(4,:);
meanerr = mean(chyba);

%ziskani nove fcni hodnoty
x_new = x;
x_new(1,N+it) = x_min(1);
x_new(2,N+it) = x_min(2);

x_only_new(1,it) = x_min(1);
x_only_new(2,it) = x_min(2);

novahodnota = fce([x_min(1);x_min(2)]);

y_new = y;
y_new(1,N+it) = novahodnota;
y_only_new(1,it) = novahodnota;



xx_new = xx;
xx_new(1,N+it) = x_min(1);
xx_new(2,N+it) = x_min(2);
xx_new(3,N+it) = novahodnota;

x = x_new;
y = y_new;

results_iter_(:,it) = [x_min(1);x_min(2);y_min;meanerr];
end
disp('Plotting...');
hold on
plot3(x1,x2,y(1,1:N),'.b')
plot3(x_only_new(1,:),x_only_new(2,:),y_only_new(1,:),'.r')
hold off
toc
resultstab = array2table(results_iter_,'RowNames',{'X1','X2','F*opt','mean_error'});
resultstab
disp(strcat('Pocet iteraci:',{' '},num2str(size(results_iter_,2))));