function [u,v] = PMDfromInit(X,c,vInit,param)

maxIter = param.maxIter;
eps = param.eps;

v = projectL2(vInit,1);
u = projectL2(X*v,1);

obj = u'*X*v;
obj = 0;
iter = 0;
improvement = 42;

while improvement>eps && iter<maxIter
    u = projectL2(X*v,1);
    v = projectL1L2((u'*X)',c);
    objNew = u'*X*v;
    improvement = (objNew-obj)/abs(obj);
    obj = objNew;
    iter = iter + 1;
end

if iter==maxIter
    warning('PMDfromInit reached maximum number of iterations')
end

    function xProj = projectL1L2(x,c)
        convergenceCrit = 1e-5;
        maxI = 100;
        nMax = sum(max(abs(x))==abs(x));
        
        if sqrt(nMax)>=c % c<=1 or pathological duplicate max values
            [~,i] = max(abs(x));
            xProj = zeros(numel(x),1);
            xProj(i) = c*sign(x(i));
        else
            xProj = projectL2(x,1);
        end
        
        cont = norm(xProj,1) > c;
        deltaRange = [0 max(abs(x))];
        iter2 = 0;
        
        while cont
            delta = mean(deltaRange);
            xProj = projectL2(softThresh(x,delta),1);
            diff = norm(xProj,1) - c;
            if diff>0
                deltaRange(1) = delta;
            else
                deltaRange(2) = delta;
            end
            cont = abs(diff) > convergenceCrit && iter2<=maxI;
            iter2 = iter2+1;
        end
    end

end