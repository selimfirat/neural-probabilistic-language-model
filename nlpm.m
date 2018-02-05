clear variables;
m = matfile("data.mat");

%%
D=8;
P=64;
lr = 0.005; % learning_rate
ew = experiment(m, D, P, lr, 25) % will be done for 50 epochs for when cross entropy decreases, the rest will be omitted
%%
D=32;
P=256;
lr = 0.005; % learning_rate
ew = experiment(m, D, P, lr, 50) % will be done for 50 epochs for when cross entropy decreases, the rest will be omitted
%%
D=16;
P=128;
lr = 0.005; % learning_rate
[ew, hw, ow] = experiment(m, D, P, lr, 35); % will be done for 50 epochs for when cross entropy decreases, the rest will be omitted
%% Part b
[eX] = nlin_dim(ew); % reduces dimensionality (t-SNE)

wx = m.words;

x = eX(:, 1);
y = eX(:, 2);
scatter(x, y)
dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
pt = text(x+dx, y+dy, wx.');
%% Part c
rng('shuffle')
r = randi([1 46500], 5, 1);
test_data = cat(3, full(onehotencode(double(m.testx(1, :))).'), full(onehotencode(double(m.testx(2, :))).'), full(onehotencode(double(m.testx(3, :))).'), full(onehotencode(double(m.testd(1, :))).'));
for a = r.'
    x1 = test_data(a, :, 1).';
    x2 = test_data(a, :, 2).';
    x3 = test_data(a, :, 3).';
    
    [y, ys, v, vs, e] = forward(x1, x2, x3, ew, hw, ow);
    
    [val1, word1] = max(x1);
    [val2, word2] = max(x2);
    [val2, word3] = max(x3);
    
    disp(['Trigram: ' m.words(1, word1) m.words(1, word2) m.words(1, word3)])
    [s, ix] = sort(y,'descend');
    disp(['Candidates:' m.words(1, ix(1)) m.words(1, ix(2)) m.words(1, ix(3)) m.words(1, ix(4)) m.words(1, ix(5))])
end
%% Part a
f1 = matfile("8_64.mat");
f2 = matfile("16_128.mat");
f3 = matfile("32_256.mat");
figure()
plot(f1.cross_ent)
hold on;
plot(f2.cross_ent)
hold on;
plot(f3.cross_ent)
legend("(D, P) = (8, 64)", "(D, P) = (16, 128)", "(D, P) = (32, 256)")
xlabel("Epoch")
ylabel("Cross Entropy Error")
title("Cross Entropy Error on Epochs Plot")

%%
function [ew, hw, ow] = experiment(m, D, P, lr, epochs)
    sigma = 0.05;
    mu = 0;
    tc = 372500;
    vc = 46500;
    mb_size = 200; % size of mini batches
    mr = 0.86; % momentum rate

    % init weights
    rng default;
    ew = normrnd(mu, sigma, 250, D);
    hw = normrnd(mu, sigma, 3*D + 1, P);
    ow = normrnd(mu, sigma, P + 1, 250);
    %epochs = 5;

    train_data = cat(3, full(onehotencode(double(m.trainx(1, :))).'), full(onehotencode(double(m.trainx(2, :))).'), full(onehotencode(double(m.trainx(3, :))).'), full(onehotencode(double(m.traind(1, :))).'));
    validation_data = cat(3, full(onehotencode(double(m.valx(1, :))).'), full(onehotencode(double(m.valx(2, :))).'), full(onehotencode(double(m.valx(3, :))).'), full(onehotencode(double(m.vald(1, :))).'));

    cross_ent = zeros(epochs, 1);
    val_acc = zeros(epochs, 1);
    for i = 1:epochs
        tt = randperm(tc);

        % temporarily store delta weights for mini batches
        delta_ow = zeros(size(ow));
        delta_hw = zeros(size(hw));
        delta_ew = zeros(size(ew));

        % temporarily store delta weights for momentum
        m_delta_ow = zeros(size(ow));
        m_delta_hw = zeros(size(hw));
        m_delta_ew = zeros(size(ew));

        for j = 1:length(tt) % t = tt
           t = tt(1,j);

           x1 = train_data(t, :, 1).';
           x2 = train_data(t, :, 2).';
           x3 = train_data(t, :, 3).';
           d = train_data(t, :, 4).';

           [y, ys, v, vs, e] = forward(x1,x2,x3, ew, hw, ow);
           % [val, argmax] = max(y);

           err = d - y;

           ds = dsoftmax(y); % arrayfun(@dsoftmax, y);
           lg_y = ds .* err; % local gradient

           delta_ow = delta_ow + (lr * lg_y * [vs; 1].').';

           dl = dlogistic(v); % arrayfun(@dlogistic, v);
           lg_h = dl .* ow(1:P, :) * lg_y;

           delta_hw = delta_hw + (lr * lg_h * [e; 1].').';

           lg_e = hw * lg_h;
           lg_e = lg_e(1:D);

           delta_ew = delta_ew + (lr * lg_e * (x1 + x2 + x3).').';      

           if mod(j, 200) == 0 || j == tc
               % delta_ow = delta_ow ./ mb_size;
               % delta_hw = delta_hw ./ mb_size;
               % delta_ew = delta_ew ./ mb_size;

               ow = ow + delta_ow + mr*m_delta_ow;
               hw = hw + delta_hw + mr*m_delta_hw;
               ew = ew + delta_ew + mr*m_delta_ew;

               m_delta_ow = delta_ow;
               m_delta_hw = delta_hw;
               m_delta_ew = delta_ew;

               delta_ow = zeros(size(ow));
               delta_hw = zeros(size(hw));
               delta_ew = zeros(size(ew));
           end
        end

        s = 0;
        for v = 1:vc
            x1 = validation_data(v, :, 1).';
            x2 = validation_data(v, :, 2).';
            x3 = validation_data(v, :, 3).';
            d = validation_data(v, :, 4).';
            [y,] = forward(x1,x2,x3, ew, hw, ow);
            [val, argmax] = max(y);
            [dval, dargmax] = max(d);
            if argmax == dargmax
                s = s + 1;
            end
            cross_ent(i) = cross_ent(i) - log(y(dargmax));
        end
        save([int2str(D) '_' int2str(P) '.mat'], 'val_acc', 'cross_ent')
        val_acc(i) = s/vc;
        disp(i)
        disp(val_acc(i))
        disp(cross_ent(i))
    end
end
%%
function p = softmax(logit)
    n = exp(logit);
    d = sum(n);
    
    p = n ./ d;
end

function dp = dsoftmax(y)
    dp = y .* (1 - y);
end

function a = logistic(x)
    a = 1.0 ./ (1.0 + exp(-x));
end

function da = dlogistic(y)
    da = y .* (1 - y);
end

function [y, ys, v, vs, e] = forward(x1, x2, x3, ew, hw, ow)
    % x: 250x1
    % ew: 250xD
    % e: 3Dx1
    %x1 = (x1 - 125) / 250;
    %x2 = (x2 - 125) / 250;
    %x3 = (x3 - 125) / 250;
    e = [ew.' * x1; ew.' * x2; ew.' * x3];
    % v: Px1
    % vs: Px2
    % hw: 3DxP
    % hb: Px1
    vs = hw.' * [e; 1];
    v = logistic(vs);
    
    % ow: Px250
    % ob: 250x1
    % ys: 250x1
    % y: 250x1
    ys = ow.' * [v; 1];
    y = softmax(ys);
end

function v = onehotencode(ind)
  vecs = length(ind);
  v = sparse(ind,1:vecs,ones(1,vecs));
end