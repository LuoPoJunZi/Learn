% 展示两个互相冲突的目标，以及目标达成法得到的权衡曲线。
% 需要 Optimization Toolbox 中的 fgoalattain。

sample_points = linspace(0, 1, 100);
sample_objectives = simple_multi_objective(sample_points');

figure('Name', 'Objective functions');
plot(sample_points, sample_objectives(:, 1), 'LineWidth', 2);
hold on;
plot(sample_points, sample_objectives(:, 2), 'LineWidth', 2);
plot([0, 0], [0, 8], 'g--');
plot([1, 1], [0, 8], 'g--');
plot([0, 1], [1, 6], 'k.', 'MarkerSize', 15);
text(-0.25, 1.5, 'Minimum(f_1(x))');
text(0.75, 5.5, 'Minimum(f_2(x))');
hold off;

legend('f_1(x)', 'f_2(x)', 'Location', 'best');
xlabel({'x'; 'Trade-off region between the green lines'});
ylabel('Objective value');
grid on;

[~, minimum_f1] = fminbnd(@(x) pick_objective(x, 1), -1, 2);
[~, minimum_f2] = fminbnd(@(x) pick_objective(x, 2), -1, 2);
goal = [minimum_f1; minimum_f2];

number_of_weights = 31;
tradeoff_parameters = linspace(0, 1, number_of_weights);
pareto_values = zeros(number_of_weights, 2);
objective_function = @simple_multi_objective;
initial_guess = 0.5;
options = optimoptions('fgoalattain', 'Display', 'off');

for index = 1:number_of_weights
    ratio = tradeoff_parameters(index);
    weight = max([ratio; 1 - ratio], 1e-6);
    [~, objective_at_solution] = fgoalattain( ...
        objective_function, initial_guess, goal, weight, ...
        [], [], [], [], [], [], [], options);
    pareto_values(index, :) = objective_at_solution(:).';
end

[~, order] = sort(pareto_values(:, 1));
sorted_values = pareto_values(order, :);

figure('Name', 'Objective trade-off');
plot(sorted_values(:, 1), sorted_values(:, 2), 'ko-', ...
    'MarkerSize', 4, 'LineWidth', 1.2);
xlabel('f_1');
ylabel('f_2');
title('Goal-attainment trade-off curve');
grid on;

function objectives = simple_multi_objective(decision)
    first_objective = sqrt(1 + decision.^2);
    second_objective = 4 + 2 * sqrt(1 + (decision - 1).^2);
    objectives = [first_objective(:), second_objective(:)];

    if isscalar(decision)
        objectives = objectives.';
    end
end

function value = pick_objective(decision, objective_index)
    objectives = simple_multi_objective(decision);
    value = objectives(objective_index);
end
