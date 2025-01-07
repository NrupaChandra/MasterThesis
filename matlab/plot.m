% Load the levelSet data
load('C:\Git\MasterThesis\matlab\levelSet.mat', 'levelSet');

% Define the range of x and y
x = linspace(-1, 1, 100);  % 100 points from -1 to 1
y = linspace(-1, 1, 100);  % 100 points from -1 to 1

% Create a meshgrid for plotting
[X, Y] = meshgrid(x, y);

% Randomly select 5 indices from the levelSet matrix
numPlots = 5;
randomIndices = randperm(numel(levelSet), numPlots);

% Loop over the randomly selected level set expressions
for i = 1:numPlots
    % Get the randomly selected level set expression
    expr = levelSet{randomIndices(i)};
    
    % Evaluate the expression for each point on the grid
    Z = zeros(size(X));  % Initialize the Z matrix (value of expression)
    
    for row = 1:numel(X)
        % Extract x and y values from the grid
        xi = X(row);
        yi = Y(row);
        
        % Calculate the value of the expression at this (xi, yi)
        Z(row) = double(subs(expr, {'x', 'y'}, {xi, yi}));  % Evaluate symbolically
    end
    
    % Create a new figure for each plot
    figure;
    
    % Plot the level set expression (contour plot)
    contour(X, Y, Z, [0, 0], 'LineWidth', 2);  % Plot the 0-level curve (interphase)
    
    % Customize the plot
    title(['Level Set Expression ' num2str(randomIndices(i))]);
    xlabel('x');
    ylabel('y');
    axis equal;
    xlim([-1 1]);
    ylim([-1 1]);
    grid on;
end