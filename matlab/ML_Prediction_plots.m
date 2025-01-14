% Load the required data
load('C:\Git\MasterThesis\matlab\nodes_x_with_id_ML.mat', 'nodes_x_with_id_ML');
load('C:\Git\MasterThesis\matlab\nodes_y_with_id_ML.mat', 'nodes_y_with_id_ML');
load('C:\Git\MasterThesis\matlab\weights_with_id_ML.mat', 'weights_with_id_ML');
load('C:\Git\MasterThesis\matlab\levelSet.mat', 'levelSet');

% Ensure data consistency
if ~isequal(length(nodes_x_with_id_ML), length(nodes_y_with_id_ML), length(weights_with_id_ML))
    error('Mismatch in the number of entries between nodes_x_with_id_ML, nodes_y_with_id_ML, and weights_with_id_ML.');
end

% Number of plots
numPlots = 5;

% Seed the random number generator
rng(1);  % Replace 42 with any fixed integer for a different sequence

% Extract IDs from nodes_x_with_id_ML
allIDs = cellfun(@(x) x{1}, nodes_x_with_id_ML, 'UniformOutput', false);  % Extract IDs from the first element

% Randomly select 5 unique IDs
randomIndices = randperm(length(allIDs), numPlots);

% Define folder to save plots
outputFolder = 'C:\Git\MasterThesis\matlab\plots';  % Updated folder path
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);  % Create folder if it doesn't exist
end

% Define the range of x and y for the level set function
x = linspace(-1, 1, 100);  % 100 points from -1 to 1
y = linspace(-1, 1, 100);  % 100 points from -1 to 1
[X, Y] = meshgrid(x, y);

% Loop through the selected IDs and create plots
for i = 1:numPlots
    % Get the current random index and ID
    currentIndex = randomIndices(i);
    currentID = allIDs{currentIndex};
    
    % Extract corresponding data for nodes and weights
    currentNodesX = nodes_x_with_id_ML{currentIndex}{2};  % Second element contains x-coordinates
    currentNodesY = nodes_y_with_id_ML{currentIndex}{2};  % Second element contains y-coordinates
    currentWeights = weights_with_id_ML{currentIndex}{2}; % Second element contains weights
    
    % Extract corresponding level set function by matching IDs
    levelSetStruct = levelSet{cellfun(@(x) strcmp(x.ID, currentID), levelSet)};
    expr = levelSetStruct.Expression;  % Symbolic level set function

    % Evaluate the level set function
    Z = zeros(size(X));  % Initialize Z matrix for level set function
    for row = 1:size(X, 1)
        for col = 1:size(X, 2)
            Z(row, col) = double(subs(expr, {'x', 'y'}, {X(row, col), Y(row, col)}));
        end
    end
    
    % Create the combined plot
    figure;
    
    % Plot the level set function's 0-level contour
    contour(X, Y, Z, [0, 0], 'LineWidth', 2, 'LineColor', 'b');  % Plot the 0-level curve (interface)
    hold on;
    
    % Plot the scatter of nodes with weights
    scatter(currentNodesX, currentNodesY, 20, currentWeights, 'filled', 'MarkerEdgeColor', 'k');
    colorbar;
    colormap(jet);
    
    % Customize the plot
    title(['prediction plot for ID: ' currentID]);
    xlabel('x');
    ylabel('y');
    axis equal;
    xlim([-1, 1]);
    ylim([-1, 1]);
    grid on;
    legend({'Level Set Contour', 'Nodes (weights as color)'}, 'Location', 'best');
    
    % Save the combined plot
    plotFileName = fullfile(outputFolder, ['combined_levelset_nodes_ID_' currentID '.png']);
    saveas(gcf, plotFileName);
end
