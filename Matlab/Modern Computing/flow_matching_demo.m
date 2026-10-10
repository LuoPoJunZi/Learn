function diagnostics = flow_matching_demo()
% FLOW_MATCHING_DEMO Regress a Gaussian velocity field and sample with ode45.
% Training pairs are independent. Sampling receives new noise, never targets.

rng(17);
trainingCount = 4000;
sampleCount = 300;
targetMean = [2, -1];
targetStd = [0.6, 1.3];
source = randn(trainingCount, 2);
target = randn(trainingCount, 2) .* targetStd + targetMean;
velocityTargets = target - source;
timeGrid = linspace(0, 1, 61)';
coefficientRows = zeros(numel(timeGrid), 6);
trainingLoss = zeros(size(timeGrid));

for index = 1:numel(timeGrid)
    time = timeGrid(index);
    positions = (1 - time) * source + time * target;
    features = [positions, ones(trainingCount, 1)];
    coefficients = features \ velocityTargets;
    coefficientRows(index, :) = coefficients(:)';
    residual = features * coefficients - velocityTargets;
    trainingLoss(index) = mean(residual(:).^2);
end

initialNoise = randn(sampleCount, 2);
outputTimes = linspace(0, 1, 21);
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
learnedField = @(time, state) learned_velocity( ...
    time, state, timeGrid, coefficientRows, sampleCount);
[times, states] = ode45(learnedField, outputTimes, initialNoise(:), options);
generated = reshape(states(end, :)', sampleCount, 2);

analyticField = @(time, state) gaussian_velocity( ...
    time, state, sampleCount, targetMean, targetStd);
[~, analyticStates] = ode45( ...
    analyticField, [0, 1], initialNoise(:), options);
analyticGenerated = reshape(analyticStates(end, :)', sampleCount, 2);
exactEndpoint = initialNoise .* targetStd + targetMean;

diagnostics.analyticEndpointError = max( ...
    abs(analyticGenerated(:) - exactEndpoint(:)));
diagnostics.learnedEndpointRMSE = sqrt( ...
    mean((generated(:) - exactEndpoint(:)).^2));
diagnostics.generatedMean = mean(generated, 1);
diagnostics.generatedStd = std(generated, 0, 1);
diagnostics.meanTrainingLoss = mean(trainingLoss);
assert(all(isfinite(generated(:))), 'Generated samples must be finite.');
assert(diagnostics.analyticEndpointError < 1e-4, ...
    'The analytic ODE must match its closed-form endpoint.');

fprintf('Target mean:       [%.3f, %.3f]\n', targetMean);
fprintf('Generated mean:    [%.3f, %.3f]\n', diagnostics.generatedMean);
fprintf('Target std:        [%.3f, %.3f]\n', targetStd);
fprintf('Generated std:     [%.3f, %.3f]\n', diagnostics.generatedStd);
fprintf('Mean training MSE: %.6f\n', diagnostics.meanTrainingLoss);
fprintf('Analytic ODE endpoint error: %.3e\n', ...
    diagnostics.analyticEndpointError);
fprintf('Learned endpoint RMSE:      %.6f\n', ...
    diagnostics.learnedEndpointRMSE);

figure('Name', 'Flow matching with a linear velocity model');
subplot(1, 2, 1);
scatter(initialNoise(:, 1), initialNoise(:, 2), 12, ...
    [0.25, 0.50, 0.75], 'filled');
hold on;
scatter(target(1:sampleCount, 1), target(1:sampleCount, 2), ...
    12, [0.70, 0.30, 0.30], 'filled');
scatter(generated(:, 1), generated(:, 2), 18, ...
    [0.20, 0.60, 0.35]);
hold off;
axis equal;
grid on;
xlabel('x_1');
ylabel('x_2');
title('Noise, target data, and generated samples');
legend('New noise', 'Target data', 'Generated', 'Location', 'best');

subplot(1, 2, 2);
hold on;
for index = 1:12
    plot(states(:, index), states(:, sampleCount + index), ...
        'LineWidth', 1.2);
end
hold off;
axis equal;
grid on;
xlabel('x_1');
ylabel('x_2');
title(sprintf('Learned ODE trajectories (%d times)', numel(times)));
end

function derivative = learned_velocity(time, state, timeGrid, rows, count)
coefficients = reshape(interp1(timeGrid, rows, time, 'linear'), 3, 2);
positions = reshape(state, count, 2);
velocity = [positions, ones(count, 1)] * coefficients;
derivative = velocity(:);
end

function derivative = gaussian_velocity(time, state, count, targetMean, targetStd)
positions = reshape(state, count, 2);
variance = (1 - time)^2 + time^2 * targetStd.^2;
slope = (time * targetStd.^2 - (1 - time)) ./ variance;
velocity = targetMean + (positions - time * targetMean) .* slope;
derivative = velocity(:);
end
