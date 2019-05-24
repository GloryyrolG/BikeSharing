function [] = fit()
    data = csvread('/Users/apple/Desktop/test/task2-1/201903.csv');
    trip = sort(data(data <= 1800))';

%     rayleigh(trip);

%     weibull(trip);

    logn(trip);
    
%     logn_fit(trip);

    logn2(trip);
% 
%     logn2_fit(trip);
    
    % nlinfit
%     modelFun = @(p, x) 1 / (sqrt(2 * pi) * p(2)) * exp(-(x - p(1)) .^ 2 ./ p(2) ^ 2);
%     startingVals = [5.5 0.5];
%     coefEsts = nlinfit(t, p, modelFun, startingVals);
%     xgrid = linspace(0, log(2000), 100);
%     line(xgrid, modelFun(coefEsts, xgrid), 'Color','r');
end

function rayleigh(trip)
%     Rayleigh
    figure;
    trip = trip / 180;
    binWidth = 0.5;
    binCtrs = 0.3: binWidth: 11;
    counts = hist(trip, binCtrs);
    n = size(trip, 2);
    prob = counts / (n * binWidth);
    bar(binCtrs, prob, 'hist');
    xlabel('Tripduration(s)');
    ylabel('f_i / n \Delta / Probability');
    xlim([0.05 10.25]);
    title('Rayleigh Distribution');
    
    [paramEsts, pCI] = raylfit(trip) % out
    xgrid = linspace(0.05, 11, 100);
    pdfEst = raylpdf(xgrid, paramEsts(1));
    line(xgrid, pdfEst, 'Linewidth', 3, 'Color', 'g');
    
    cdf = [trip' raylcdf(trip', paramEsts(1))];
    [h p ks cv] = kstest(trip', 'CDF', cdf) % out
end

function weibull(trip)
%     Weibull
    figure;
    trip = trip / 180;
    binWidth = 0.125;
    binCtrs = 0.3: binWidth: 11;
    counts = hist(trip, binCtrs);
    n = size(trip, 2);
    prob = counts / (n * binWidth);
    bar(binCtrs, prob, 'hist');
    xlabel('Tripduration(s)');
    ylabel('f_i / n \Delta / Probability');
    xlim([0.05 10.25]);
    title('Weibull Distribution');
    
    [paramEsts, pCI] = wblfit(trip) % out
    xgrid = linspace(0.05, 11, 100);
    pdfEst = wblpdf(xgrid, paramEsts(1), paramEsts(2));
    line(xgrid, pdfEst, 'Linewidth', 3, 'Color', 'g');
    
    cdf = [trip' wblcdf(trip', paramEsts(1), paramEsts(2))];
    [h p ks cv] = kstest(trip', 'CDF', cdf) % out
end

function logn(trip)
%     Log-normal2
    figure;
    trip = trip / 180;
    binWidth = 0.05;
    binCtrs = 1 / 3: binWidth: 10;
    counts = hist(trip, binCtrs);
    n = size(trip, 2);
    prob = counts / (n * binWidth);
    bar(binCtrs, prob, 'hist');
    xlabel('Tripduration(s)');
    ylabel('f_i / n \Delta / Probability');
    xlim([0.30 10.03]);
    title('Log-Normal Distribution');

    [paramEsts, pCI] = lognfit(trip) % out
    xgrid = linspace(0.30, 10.03, 100);
    pdfEst = lognpdf(xgrid, paramEsts(1), paramEsts(2));
    line(xgrid, pdfEst, 'Linewidth', 3, 'Color', 'r');
    fprintf('mu0 = %.6f, mu = %.6f, simga = %.6f\n', paramEsts(1), exp(paramEsts(1)) * 180, paramEsts(2));

    cdf = [trip' logncdf(trip', paramEsts(1), paramEsts(2))];
    [h p ks cv] = kstest(trip', 'CDF', cdf) % out
end

function logn_fit(trip)
%     Log-normal2
    figure;
    trip = trip / 180;
    binWidth = 0.05;
    binCtrs = 1 / 3: binWidth: 10;
    counts = hist(trip, binCtrs);
    n = size(trip, 2);
    prob = counts / (n * binWidth);
    bar(binCtrs, prob, 'hist');
    xlabel('Tripduration / 180(s)');
    ylabel('f_i / n \Delta / Probability');
    xlim([0.30 10.03]);
    title('Log-Normal Distribution, \mu = 0.4883, \sigma^2 = 0.5415^2');

    modelFun = @(p,x) p(2) ./ (sqrt(2 * pi) * p(4) .* (p(1) * x)) ...
        .* exp(-((log(p(1) * x) - p(3)) / p(4)) .^ 2 / 2);
    startingVals = [1 1 0.40 0.5];
    coefEsts = nlinfit(binCtrs, prob, modelFun, startingVals);
    xgrid = linspace(0.30, 10.03, 100);
    line(xgrid, modelFun(coefEsts, xgrid), 'Linewidth', 3, 'Color', 'y');
    mu = coefEsts(3) - log(coefEsts(1));
    sigma = coefEsts(4);
    fprintf('mu0 = %.6f, mu = %.6f, simga = %.6f, a = %.6f, b = %.6f\n', mu, exp(mu) * 180, sigma, ...
        coefEsts(1), coefEsts(2));

    cdf = [trip' logncdf(trip', mu, sigma)];
    [h p ks cv] = kstest(trip', 'CDF', cdf) % out
end

function logn2(trip)
%     Log-normal2
    figure;
    trip = log(trip);
    binWidth = 0.05;
    binCtrs = 4.1: binWidth: 7.5;
    counts = hist(trip, binCtrs);
    n = size(trip, 2);
    prob = counts / (n * binWidth);
    bar(binCtrs, prob, 'hist');
    xlabel('Tripduration(s)');
    ylabel('f_i / n \Delta / Probability');
    xlim([4.075 7.525]);
    title('Normal Distribution after Log');
    
    [mu, sigma, muCI, simgaCI] = normfit(trip, 0.05) % out
    xgrid = linspace(4.075, 7.525, 100);
    pdfEst = normpdf(xgrid, mu, sigma);
    line(xgrid, pdfEst, 'Linewidth', 3, 'Color', 'r');
    fprintf('mu0 = %.6f, mu = %.6f, sigma_log = %.6f\n', mu, exp(mu), sigma);
    
    [h, p, ksstat, cv] = kstest((trip - mu) / sigma) % outs
end

function logn2_fit(trip)
%     Log-normal2 nlinfit
    figure;
    trip = log(trip);
    binWidth = 0.02;
    binCtrs = 4.1: binWidth: 7.5;
    counts = hist(trip, binCtrs);
    n = size(trip, 2);
    prob = counts / (n * binWidth);
    bar(binCtrs, prob, 'hist');
    xlabel('ln Tripduration(s)');
    ylabel('f_i / n \Delta / Probability');
    xlim([4.075 7.525]);
    title('Normal Distribution after Log, \mu = 5.6905, \sigma = 0.5727^2');
    
    modelFun = @(p,x) p(2) / (sqrt(2 * pi) * p(4)) ...
        .* exp(-(((p(1) * x) - p(3)) / p(4)) .^ 2 / 2);
    startingVals = [1 1 5.6 0.4];
    coefEsts = nlinfit(binCtrs, prob, modelFun, startingVals);
    xgrid = linspace(4.075, 7.525, 100);
    line(xgrid, modelFun(coefEsts, xgrid), 'Linewidth', 3, 'Color', 'y');
    
    mu = coefEsts(3) / coefEsts(1);
    sigma = coefEsts(4) / coefEsts(1);
    fprintf('mu0 = %.6f, mu = %.6f, sigma_log = %.6f, cy = %.6f\n', mu, exp(mu), sigma, coefEsts(2));
    
    [h, p, ksstat, cv] = kstest((trip - mu) / sigma) % outs
end
