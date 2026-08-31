% FOURIER_NEURAL_OPERATOR_DEMO Learn a simple operator in Fourier space.

rng(19);

gridSize = 128;
numTrainingSamples = 64;
maxInputMode = 12;
retainedModes = 8;
diffusivity = 0.08;
deltaTime = 0.4;

x = (0:gridSize - 1) * (2 * pi / gridSize);
waveNumbers = [0:gridSize / 2 - 1, -gridSize / 2:-1];

% The heat equation advances every Fourier mode by this exact multiplier.
trueMultiplier = exp(-diffusivity * (waveNumbers .^ 2) * deltaTime);

trainingInputs = zeros(numTrainingSamples, gridSize);
for sampleIndex = 1:numTrainingSamples
    signal = zeros(1, gridSize);
    for modeIndex = 1:maxInputMode
        cosineWeight = randn() / modeIndex;
        sineWeight = randn() / modeIndex;
        signal = signal ...
            + cosineWeight * cos(modeIndex * x) ...
            + sineWeight * sin(modeIndex * x);
    end
    trainingInputs(sampleIndex, :) = signal;
end

trainingSpectra = fft(trainingInputs, [], 2);
targetSpectra = trainingSpectra .* trueMultiplier;
trainingTargets = real(ifft(targetSpectra, [], 2));

% Fit one complex multiplier per frequency using least squares.
spectralNumerator = sum(conj(trainingSpectra) .* targetSpectra, 1);
spectralDenominator = sum(abs(trainingSpectra) .^ 2, 1);
learnedMultiplier = zeros(1, gridSize);
observedModes = spectralDenominator > eps;
learnedMultiplier(observedModes) = ...
    spectralNumerator(observedModes) ./ spectralDenominator(observedModes);

% A small FNO keeps only a limited number of low-frequency modes.
retainedMask = abs(waveNumbers) <= retainedModes;
learnedMultiplier(~retainedMask) = 0;

testInput = ...
    1.1 * sin(x) ...
    - 0.5 * cos(3 * x) ...
    + 0.35 * sin(7 * x) ...
    + 0.2 * cos(10 * x);
testSpectrum = fft(testInput);
testTarget = real(ifft(testSpectrum .* trueMultiplier));
testPrediction = real(ifft(testSpectrum .* learnedMultiplier));

relativeError = norm(testPrediction - testTarget) / norm(testTarget);
trainingPrediction = real(ifft(trainingSpectra .* learnedMultiplier, [], 2));
trainingRelativeError = ...
    norm(trainingPrediction - trainingTargets, 'fro') ...
    / norm(trainingTargets, 'fro');

fprintf('Training samples: %d\n', numTrainingSamples);
fprintf('Grid points: %d\n', gridSize);
fprintf('Retained Fourier modes: %d on each side\n', retainedModes);
fprintf('Training relative error: %.6f\n', trainingRelativeError);
fprintf('Test relative error: %.6f\n', relativeError);

nonnegativeIndices = 1:gridSize / 2 + 1;
displayModeCount = 16;
displayIndices = 1:displayModeCount + 1;

figure('Name', 'Fourier neural operator intuition');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot(x, testInput, 'Color', [0.25, 0.3, 0.35], 'LineWidth', 1.2);
hold on;
plot(x, testTarget, 'Color', [0.0, 0.45, 0.7], 'LineWidth', 1.8);
plot(x, testPrediction, '--', 'Color', [0.85, 0.25, 0.2], 'LineWidth', 1.6);
hold off;
grid on;
xlabel('x');
ylabel('u(x)');
title('Input and one-step operator output');
legend('Input', 'Exact target', 'Learned spectral operator', ...
    'Location', 'best');

nexttile;
stem(waveNumbers(displayIndices), abs(trueMultiplier(displayIndices)), ...
    'Color', [0.0, 0.45, 0.7], 'LineWidth', 1.2);
hold on;
stem(waveNumbers(displayIndices), abs(learnedMultiplier(displayIndices)), ...
    'Color', [0.85, 0.25, 0.2], 'LineStyle', '--');
hold off;
grid on;
xlabel('Fourier mode');
ylabel('Multiplier magnitude');
title('Exact and learned mode multipliers');
legend('Exact', 'Learned and truncated', 'Location', 'best');

nexttile;
plot(x, testPrediction - testTarget, ...
    'Color', [0.55, 0.15, 0.55], 'LineWidth', 1.4);
grid on;
xlabel('x');
ylabel('Prediction error');
title(sprintf('Pointwise error, relative norm %.4f', relativeError));

nexttile;
semilogy( ...
    abs(waveNumbers(nonnegativeIndices)), ...
    abs(testSpectrum(nonnegativeIndices)) + eps, ...
    'Color', [0.1, 0.55, 0.35], ...
    'LineWidth', 1.5);
xline(retainedModes, '--', 'Retained-mode boundary');
grid on;
xlabel('Absolute Fourier mode');
ylabel('Test input magnitude');
title('High modes are deliberately truncated');

sgtitle('Learning a one-step diffusion operator in Fourier space');
