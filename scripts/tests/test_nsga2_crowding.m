function test_nsga2_crowding()
% Check NSGA-II crowding edge cases without training, plotting or file writes.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    previousPath = path;
    restorePath = onCleanup(@() path(previousPath)); %#ok<NASGU>
    addpath(fullfile(root, 'Matlab', 'Multi-Objective Optimization', 'NSGA-II'));

    pop = repmat(struct('Cost', [1; 1], 'CrowdingDistance', []), 1, 3);
    result = CalcCrowdingDistance(pop, {[], 1:3});
    assert(all([result.CrowdingDistance] == 0));

    pop(1).Cost = [0; 1];
    pop(2).Cost = [1; 1];
    pop(3).Cost = [2; 1];
    result = CalcCrowdingDistance(pop, {1:3});
    assert(isinf(result(1).CrowdingDistance));
    assert(result(2).CrowdingDistance == 1);
    assert(isinf(result(3).CrowdingDistance));

    result = CalcCrowdingDistance(pop, {1, 2:3});
    assert(all(isinf([result.CrowdingDistance])));
    fprintf('NSGA-II crowding: 3 edge-case groups passed.\n');
end
