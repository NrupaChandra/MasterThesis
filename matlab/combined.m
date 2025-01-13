% Load the required data
load('C:\Git\MasterThesis\matlab\nodes_x_with_id.mat', 'nodes_x_with_id');
load('C:\Git\MasterThesis\matlab\nodes_y_with_id.mat', 'nodes_y_with_id');
load('C:\Git\MasterThesis\matlab\weights_with_id.mat', 'weights_with_id');
load('C:\Git\MasterThesis\matlab\nodes_x_with_id_ML.mat', 'nodes_x_with_id_ML');
load('C:\Git\MasterThesis\matlab\nodes_y_with_id_ML.mat', 'nodes_y_with_id_ML');
load('C:\Git\MasterThesis\matlab\weights_with_id_ML.mat', 'weights_with_id_ML');
load('C:\Git\MasterThesis\matlab\levelSet.mat', 'levelSet');

% Ensure data consistency for actual data
if ~isequal(length(nodes_x_with_id), length(nodes_y_with_id), length(weights_with_id))
    error('Mismatch in the number of entries between nodes_x_with_id, nodes_y_with_id, and weights_with_id.');
end

% Ensure data consistency for machine learning data
if ~isequal(length(nodes_x_with_id_ML), length(nodes_y_with_id_ML), length(weights_with_id_ML))
    error('Mismatch in the number of entries between nodes_x_with_id_ML, nodes_y_with_id_ML, and weights_with_id_ML.');
end

% Number of plots
numPlots = 5;

% Seed the random number generator
rng(1);  % Replace 1 with any fixed integer for a different sequence

% Extract IDs from nodes_x_with_id and nodes_x_with_id_ML
allIDs = cellfun(@(x) x{1}, nodes_x_with_id, 'UniformOutput', false);  % Extract IDs from the first element
allIDs_ML = cellfun(@(x) x{1}, nodes_x_with_id_ML, 'UniformOutput', false);  % Extract IDs from the first element

% Ensure that only IDs that exist in both datasets are considered
commonIDs = intersect(allIDs, allIDs_ML);

if length(commonIDs) < numPlots
    error('Not enough matching IDs between actual data and machine learning data.');
end

% Randomly select 5 matching IDs
randomIndices = randperm(length(commonIDs), numPlots);

% Define folder to save plots
outputFolder = 'C:\Git\MasterThesis\matlab\combined_plots';  % Updated folder path for combined plots
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);  % Create folder if it doesn't exist
end

% Define the range of x and y for the level set function
x = linspace(-1, 1, 100);  % 100 points from -1 to 1
y = linspace(-1, 1, 100);  % 100 points from -1 to 1
[X, Y] = meshgrid(x, y);

% Loop through the selected matching IDs and create plots
for i = 1:numPlots
    % Get the current matching ID
    currentID = commonIDs{randomIndices(i)};
    
    % Find index of the currentID in actual data
    actualIndex = find(strcmp(currentID, allIDs));
    
    % Find index of the currentID in machine learning data
    mlIndex = find(strcmp(currentID, allIDs_ML));
    
    % Extract corresponding data for nodes and weights from actual data
    currentNodesX = nodes_x_with_id{actualIndex}{2};  % Second element contains x-coordinates
    currentNodesY = nodes_y_with_id{actualIndex}{2};  % Second element contains y-coordinates
    currentWeights = weights_with_id{actualIndex}{2}; % Second element contains weights
    
    % Extract corresponding data for nodes and weights from machine learning data
    currentNodesX_ML = nodes_x_with_id_ML{mlIndex}{2};  % Second element contains x-coordinates
    currentNodesY_ML = nodes_y_with_id_ML{mlIndex}{2};  % Second element contains y-coordinates
    currentWeights_ML = weights_with_id_ML{mlIndex}{2}; % Second element contains weights
    
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
    
    % Create the plot for actual data and machine learning data combined
    figure;
    
    % Plot the level set function's 0-level contour
    contour(X, Y, Z, [0, 0], 'LineWidth', 2, 'LineColor', 'b');  % Plot the 0-level curve (interface)
    hold on;
    
    % Plot the scatter of nodes with weights from actual data
    scatter(currentNodesX, currentNodesY, 20, currentWeights, 'filled', 'MarkerEdgeColor', 'k');
    colorbar;
    colormap(jet);
    
    % Plot the scatter of nodes with weights from machine learning data
    scatter(currentNodesX_ML, currentNodesY_ML, 20, currentWeights_ML, 'filled', 'MarkerEdgeColor', 'r');
    colorbar;
    colormap(jet);
    
    % Customize the plot
    title(['Combined plot for ID: ' currentID]);
    xlabel('x');
    ylabel('y');
    axis equal;
    xlim([-1, 1]);
    ylim([-1, 1]);
    grid on;
    legend({'Level Set Contour', 'Actual Nodes (weights as color)', 'ML Nodes (weights as color)'}, 'Location', 'best');
    
    % Save the combined plot
    plotFileName = fullfile(outputFolder, ['combined_levelset_nodes_ID_' currentID '.png']);
    saveas(gcf, plotFileName);
end
