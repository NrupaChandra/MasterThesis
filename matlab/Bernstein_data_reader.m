% Define folder to save plots
outputFolder = 'C:\Git\MasterThesis\matlab';  % Change this to your desired folder path
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);  % Create folder if it doesn't exist
end

% Read the level set function data
levelSetFile = 'C:\Git\MasterThesis\matlab\Bernstein_p1_data.txt';  
levelSetData = readtable(levelSetFile, 'Delimiter', ';');

% Read the nodes and weights data
nodesWeightsFile = 'C:\Git\MasterThesis\matlab\Bernstein_p1_output.txt';  
nodesWeightsData = readtable(nodesWeightsFile, 'Delimiter', ';');

% Extract relevant columns from the table
exp_x = levelSetData.exp_x;   % x-expressions
exp_y = levelSetData.exp_y;   % y-expressions
coeff = levelSetData.coeff;   % coefficients

% Convert string data in columns to numerical matrices
n = size(levelSetData, 1);  % Number of rows in the data

% Initialize numerical matrices
exp_x_num = zeros(n, 4);  % Assuming each row has 4 numbers
exp_y_num = zeros(n, 4);  % Assuming each row has 4 numbers
coeff_num = zeros(n, 4);  % Assuming each row has 4 numbers

% Loop through each row and convert strings to numerical arrays with double precision
for i = 1:n
    exp_x_num(i, :) = sscanf(exp_x{i}, '%f,%f,%f,%f');  % Convert x-expressions
    exp_y_num(i, :) = sscanf(exp_y{i}, '%f,%f,%f,%f');  % Convert y-expressions
    coeff_num(i, :) = sscanf(coeff{i}, '%f,%f,%f,%f');  % Convert coefficients
end

% Initialize the levelSet matrix (n x 1 cell array to store expressions)
levelSet = cell(n, 1);

% Loop over each row to construct the level set expressions
for i = 1:n
    % Initialize the expression for the current row
    expr = 0;
    
    % Loop over the 4 terms (since exp_x_num, exp_y_num, coeff_num are all 4 columns)
    for j = 1:4
        % Construct the term coeff(i) * x^exp_x(i) * y^exp_y(i)
        term = coeff_num(i, j) * str2sym(['x^', num2str(exp_x_num(i, j))]) * str2sym(['y^', num2str(exp_y_num(i, j))]);
        % Add the term to the expression
        expr = expr + term;
    end
    
    % Store the expression in the levelSet matrix
    levelSet{i} = expr;
end

% Save the levelSet matrix to a .mat file
save('C:\Git\MasterThesis\matlab\levelSet.mat', 'levelSet');
