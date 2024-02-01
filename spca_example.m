clear

% create some data
w1true = [ones(5,1);-ones(5,1);zeros(90,1)];
w1true = w1true/norm(w1true);
w2true = [ones(20,1);-ones(20,1);zeros(60,1)];
w2true = w2true/norm(w2true);

Z = randn(100,2); 
Z = orth(Z);

X = normrnd(Z(:,1)*w1true' + Z(:,2)*w2true',0.05);
Y = sum(Z,2)>0;

%save('spca_example','X','Y','w1true','w2true')

% find optimal sparsity parameters
optC = tunespca(X,'K',2);
% calculate the first two principal components
[coeff,score] = spca(X,optC,'K',2);

% plot coefficients against the ground truth
figure
    subplot(2,2,1);bar(w1true);title('COEFF_1 true')
    subplot(2,2,2);bar(w2true);title('COEFF_2 true')
    subplot(2,2,3);bar(coeff(:,1));title('COEFF_1')
    subplot(2,2,4);bar(coeff(:,2));title('COEFF_2')
% plot the scores
figure
    gscatter(score(:,1),score(:,2),Y,[],'os',10,'filled')
    legend off
    xlabel('First principal component')
    ylabel('Second principal component')

% calculate and plot a regularization path for the first component
[coeff,score,~,~,c] = spca(X,(0:0.01:0.5)');
figure
    subplot(2,1,1)
    p1=plot(c,squeeze(coeff(w1true~=0,1,:))','r');
    hold on; p2=plot(c,squeeze(coeff(w1true==0,1,:))','k');
    legend([p1(1);p2(1)],{'True variable','Noise variable'},...
        'Location','southeast')
    xlabel('c');title('Coefficient')
    subplot(2,1,2)
    scatter(c,squeeze(score(Y==1,1,:))',15,[0,0.4470,0.7410],'o')
    hold on
    scatter(c,squeeze(score(Y==0,1,:))',15,[0.8500,0.3250,0.0980],'s')
    xlabel('c');title('Score')

