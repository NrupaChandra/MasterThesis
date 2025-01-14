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

% Read the nodes and weights data from ML
nodesWeightsFile_ML = 'C:\Git\MasterThesis\matlab\dnn_predictions_p1_output.txt';  
nodesWeightsData_ML = readtable(nodesWeightsFile_ML, 'Delimiter', ';');

% Initialize the levelSet matrix (n x 1 cell array to store expressions)
structuredData_ML = table();

% Number of rows in the data
numRows_ML = height(nodesWeightsData_ML);

% Initialize cell arrays to store the updated vectors
nodes_x_with_id_ML = cell(numRows_ML, 1);
nodes_y_with_id_ML = cell(numRows_ML, 1);
weights_with_id_ML = cell(numRows_ML, 1);

% Loop through each row and process the data
for i = 1:numRows_ML
    % Convert the comma-separated strings to numeric arrays
    nodes_x_ML = sscanf(nodesWeightsData_ML.nodes_x{i}, '%f,')';  % Convert nodes_x
    nodes_y_ML = sscanf(nodesWeightsData_ML.nodes_y{i}, '%f,')';  % Convert nodes_y
    weights_ML = sscanf(nodesWeightsData_ML.weights{i}, '%f,')';  % Convert weights
    
    % Get the ID for this row
    id_ML = nodesWeightsData_ML.id(i);
    
    % Concatenate the ID with the nodes_x, nodes_y, and weights vectors
    nodes_x_with_id_ML{i} = [id_ML, nodes_x_ML];  % Add the ID as the first element of nodes_x
    nodes_y_with_id_ML{i} = [id_ML, nodes_y_ML];  % Add the ID as the first element of nodes_y
    weights_with_id_ML{i} = [id_ML, weights_ML];  % Add the ID as the first element of weights
end

% Add the columns with ID included in the vectors to the new table
structuredData_ML.nodes_x_with_id_ML = nodes_x_with_id_ML;
structuredData_ML.nodes_y_with_id_ML = nodes_y_with_id_ML;
structuredData_ML.weights_with_id_ML = weights_with_id_ML;

% Extract relevant columns from the table
exp_x = levelSetData.exp_x;   % x-expressions
exp_y = levelSetData.exp_y;   % y-expressions
coeff = levelSetData.coeff;   % coefficients
id = levelSetData.id;  

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
    
     % Store the expression and the ID in the levelSet matrix
    levelSet{i} = struct('ID', id(i), 'Expression', expr);
end

% Initialize an empty table to store the new structured data
structuredData = table();

% Number of rows in the data
numRows = height(nodesWeightsData);

% Initialize cell arrays to store the updated vectors
nodes_x_with_id = cell(numRows, 1);
nodes_y_with_id = cell(numRows, 1);
weights_with_id = cell(numRows, 1);

% Loop through each row and process the data
for i = 1:numRows
    % Convert the comma-separated strings to numeric arrays
    nodes_x = sscanf(nodesWeightsData.nodes_x{i}, '%f,')';  % Convert nodes_x
    nodes_y = sscanf(nodesWeightsData.nodes_y{i}, '%f,')';  % Convert nodes_y
    weights = sscanf(nodesWeightsData.weights{i}, '%f,')';  % Convert weights
    
    % Get the ID for this row
    id = nodesWeightsData.id(i);
    
    % Concatenate the ID with the nodes_x, nodes_y, and weights vectors
    nodes_x_with_id{i} = [id, nodes_x];  % Add the ID as the first element of nodes_x
    nodes_y_with_id{i} = [id, nodes_y];  % Add the ID as the first element of nodes_y
    weights_with_id{i} = [id, weights];  % Add the ID as the first element of weights
end

% Add the columns with ID included in the vectors to the new table
structuredData.nodes_x_with_id = nodes_x_with_id;
structuredData.nodes_y_with_id = nodes_y_with_id;
structuredData.weights_with_id = weights_with_id;

% Save the tables

save('C:\Git\MasterThesis\matlab\nodes_x_with_id_ML.mat', 'nodes_x_with_id_ML');
save('C:\Git\MasterThesis\matlab\nodes_y_with_id_ML.mat', 'nodes_y_with_id_ML');
save('C:\Git\MasterThesis\matlab\weights_with_id_ML.mat', 'weights_with_id_ML');
save('C:\Git\MasterThesis\matlab\nodes_x_with_id.mat', 'nodes_x_with_id');
save('C:\Git\MasterThesis\matlab\nodes_y_with_id.mat', 'nodes_y_with_id');
save('C:\Git\MasterThesis\matlab\weights_with_id.mat', 'weights_with_id');
save('C:\Git\MasterThesis\matlab\levelSet.mat', 'levelSet');

