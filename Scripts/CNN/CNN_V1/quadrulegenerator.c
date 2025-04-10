
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Algoimwrapper.h"
#include <time.h>

int QUAD_ORDER = 16;

size_t getline(char **lineptr, size_t *n, FILE *stream) {
    if (*lineptr == NULL || *n == 0) {
        *n = 128;  // Start with a buffer of size 128
        *lineptr = malloc(*n);
        if (*lineptr == NULL) {
            return -1;
        }
    }

    size_t i = 0;
    int c;
    while ((c = fgetc(stream)) != EOF) {
        (*lineptr)[i++] = (char)c;
        if (c == '\n') {
            break;
        }

        if (i >= *n) {
            *n *= 2;  // Double the buffer size if needed
            *lineptr = realloc(*lineptr, *n);
            if (*lineptr == NULL) {
                return -1;
            }
        }
    }

    if (i == 0 && c == EOF) {
        return -1;  // No data or end of file
    }

    (*lineptr)[i] = '\0';  // Null-terminate the string
    return i;
}

// Function to count the number of comma-separated values in a string
int count_comma_separated_values(const char *str) {
    int count = 0;
    const char *ptr = str;

    // Count the number of commas
    while (*ptr) {
        if (*ptr == ',') {
            count++;
        }
        ptr++;
    }

    // The number of values is one more than the number of commas
    return count + 1;
}

// Function to parse comma-separated double values dynamically
double* parse_comma_separated_doubles(const char *str, int *count) {
    *count = count_comma_separated_values(str);  // Count how many values there are
    double *values = malloc(*count * sizeof(double));  // Dynamically allocate memory for the values

    if (values == NULL) {
        perror("Unable to allocate memory");
        exit(1);  // Exit if memory allocation fails
    }

    // Parse the values
    char *str_copy = strdup(str);  // Create a copy to tokenize
    char *token = strtok(str_copy, ",");
    int index = 0;

    while (token != NULL) {
        values[index++] = atof(token);  // Convert string to double
        token = strtok(NULL, ",");
    }

    free(str_copy);  // Free the copy
    return values;
}
// Function to parse comma-separated int values
int* parse_comma_separated_integers(const char *str, int *count) {
    *count = count_comma_separated_values(str);  // Count how many values there are
    int *values = malloc(*count * sizeof(double));  // Dynamically allocate memory for the values

    if (values == NULL) {
        perror("Unable to allocate memory");
        exit(1);  // Exit if memory allocation fails
    }

    char *str_copy = strdup(str);  // Create a copy of the string to modify
    char *token = strtok(str_copy, ",");
    int index = 0;
    while (token != NULL) {
        values[index++] = atof(token);  // Convert string to double
        token = strtok(NULL, ",");
    }

    free(str_copy);  // Free the duplicated string
    return values;
}

// Function to replace a substring in the filename
void replace_substring_in_filename(char *filename, const char *old_substr, const char *new_substr) {
    char *pos = strstr(filename, old_substr);
    if (pos) {
        // Allocate enough space for the new filename
        char new_filename[256];  // Adjust size as needed
        size_t prefix_len = pos - filename;  // Length of the part before the substring
        size_t suffix_len = strlen(filename) - (prefix_len + strlen(old_substr));  // Length of the part after the substring

        // Copy the part before the substring
        strncpy(new_filename, filename, prefix_len);
        new_filename[prefix_len] = '\0';

        // Append the new substring
        strcat(new_filename, new_substr);

        // Append the part after the substring
        strcat(new_filename, pos + strlen(old_substr));

        // Copy the new filename back to the original variable
        strcpy(filename, new_filename);
    }
}

void write_header_if_needed(const char *filename) {
    FILE *file = fopen(filename, "r");  // Try to open the file for reading

    if (file == NULL) {
        // File doesn't exist, so create it and write the header
        file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file for writing");
            exit(1);
        }
        // Write the header to the new file
        fprintf(file, "number;id;nodes_x;nodes_y;weights\n");
        printf("File %s created and header written.\n", filename);
    } else {
        // The file already exists, just close it
        fclose(file);
        printf("File %s already exists. No header written.\n", filename);
    }
}

// Function to find the index of an ID in the output array
int find_id_in_output(const char *id, char **ids_out, int num_out_ids) {
    for (int i = 0; i < num_out_ids; i++) {
        if (strcmp(id, ids_out[i]) == 0) {
            return i;  // Return the index if the ID matches
        }
    }
    return -1;  // Return -1 if the ID is not found
}

char *double_array_to_csv(const double *array, int size, int precision) {
    if (size == 0) {
        // Handle empty array case by returning an empty string
        char *empty_string = malloc(1);
        if (empty_string) {
            empty_string[0] = '\0';
        }
        return empty_string;
    }

    // Estimate buffer size: (digits per double + 1 for comma) * size
    int estimated_size = size * (32 + precision + 1); // 32 accounts for float formatting
    char *output = malloc(estimated_size);
    if (!output) {
        perror("Failed to allocate memory");
        return NULL;
    }

    // Write the doubles into the output string
    int pos = 0;
    for (int i = 0; i < size; i++) {
        // Append the current double to the output string
        int written = snprintf(output + pos, estimated_size - pos, "%.*f", precision, array[i]);
        if (written < 0 || written >= estimated_size - pos) {
            perror("Failed to write to string buffer");
            free(output);
            return NULL;
        }
        pos += written;

        // Append a comma except after the last value
        if (i < size - 1) {
            if (pos < estimated_size - 1) {
                output[pos++] = ',';
            } else {
                fprintf(stderr, "Output buffer overflow\n");
                free(output);
                return NULL;
            }
        }
    }

    // Null-terminate the string
    if (pos < estimated_size) {
        output[pos] = '\0';
    } else {
        output[estimated_size - 1] = '\0';
    }

    return output;
}

void print_progress_bar(int progress, int total) {
    int bar_width = 50; // Width of the progress bar
    float percentage = (float)progress / total;
    int filled = (int)(percentage * bar_width);

    // Print progress bar
    printf("\r["); // Start the bar
    for (int i = 0; i < bar_width; i++) {
        if (i < filled)
            printf("#");
        else
            printf(" ");
    }
    printf("] %.1f%%", percentage * 100); // Percentage
    fflush(stdout); // Force the output to be displayed
}

int main(int argc, char *argv[]) {
    //Nrupa part
    clock_t start_time = clock(); //start clock
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <QuadOrder> <input_filename> <io_switch>\n", argv[0]);
        return 1;
    }
    QUAD_ORDER = atoi(argv[1]);
    char *filename = argv[2];
    // If argv[3] is "1" (or any nonzero value) then I/O is enabled;

    // if it is "0" then I/O is turned off.
    int io_enabled = (strcmp(argv[3], "0") == 0 || strcmp(argv[3], "false") == 0) ? 0 : 1;
    //till here

    // Check if the user has provided a file name as an argument
    if (argc < 2) {
        fprintf(stderr, "Usage: %s (optional <QuadOrder>) <filename>\n", argv[0]);
        return 1;  // Exit with error code 1 if file name is not provided
    }

    // The file name is provided as the first argument (argv[1])
    char *filename;

	if (argc == 3){
		QUAD_ORDER = atoi(argv[1]);
		filename = argv[2];
	}else if(argc == 2){
		filename = argv[1];
	} else{
		fprintf(stderr, "Too many arguments\n");
		fprintf(stderr, "Usage: %s (optional <QuadOrder>) <filename>\n", argv[0]);
		return 1;
	}

    // Open the file
    FILE *file_input = fopen(filename, "r");

    if (file_input == NULL) {
        perror("Error opening file");
        return 1;  // Exit with error code if file can't be opened
    }

    char *line = NULL;  // To hold each line
    size_t len = 0;     // Size of the allocated buffer
    size_t read;

    // Skip the header line
    if ((read = getline(&line, &len, file_input)) == -1) {
        fprintf(stderr, "Error reading header\n");
        fclose(file_input);
        return 1;
    }

    char **ids_in;
    int *numbers_in;
    int *in_out_map; // maps from line number in input line number in output.

    char **ids_out;
    int *numbers_out;

    int num_in_ids = 0;
    int num_out_ids = 0;

    int pass = 1;
    readfile_in:
    int index = 0;


    // Read the file line by line (data lines)
    while ((read = getline(&line, &len, file_input)) != -1) {
        // Remove the newline character at the end of the line (if any)
        line[strcspn(line, "\n")] = '\0';

        // Split the line by ';'
        char *number = strtok(line, ";");
        char *id = strtok(NULL, ";");
        char *exp_x_str = strtok(NULL, ";");
        char *exp_y_str = strtok(NULL, ";");
        char *coeff_str = strtok(NULL, ";");

        // Check if all fields are found
        if (number && id && exp_x_str && exp_y_str && coeff_str) {
            // Convert the number to int
            int number_val = atoi(number);  

            /*
            // Variables to hold the number of values parsed
            int exp_x_count = 0, exp_y_count = 0, coeff_count = 0;

            // Parse the comma-separated lists into arrays of doubles
            int *exp_x_values = parse_comma_separated_integers(exp_x_str, &exp_x_count);
            int *exp_y_values = parse_comma_separated_integers(exp_y_str, &exp_y_count);
            double *coeff_values = parse_comma_separated_doubles(coeff_str, &coeff_count);
            
            // Print the parsed data
            printf("Number: %d, ID: %s\n", number_val, id);
            printf("exp_x: ");
            for (int i = 0; i < exp_x_count; i++) {
                printf("%d ", exp_x_values[i]);
            }
            printf("\n");

            printf("exp_y: ");
            for (int i = 0; i < exp_y_count; i++) {
                printf("%d ", exp_y_values[i]);
            }
            printf("\n");

            printf("coeff: ");
            for (int i = 0; i < coeff_count; i++) {
                printf("%.5f ", coeff_values[i]);
            }
            printf("\n");
            */
           if(pass == 2){
                ids_in[index] = malloc(strlen(id) + 1);
                strcpy(ids_in[index], id);
                numbers_in[index] = number_val;
           }
           index++;
        } else {
            printf("Error parsing line: %s\n", line);
        }
    }
    if(pass == 1){
        printf("Completed first pass on input file\n");
        printf("Found %d valid data sets\n", index);
        pass = 2;
        num_in_ids = index;
        ids_in = malloc(num_in_ids * sizeof(char *));
        numbers_in = malloc(num_in_ids * sizeof(int));
        in_out_map = malloc(num_in_ids * sizeof(int));
        fseek(file_input, 0, SEEK_SET);
        (read = getline(&line, &len, file_input));
        goto readfile_in;
    } else if(pass == 2){
        printf("Completed second pass on input file\n");
        printf("Extracted ids\n");
    }
    fclose(file_input);  // Close the file when done
	char quadname[9];
	sprintf(quadname, "output_%d", QUAD_ORDER);
    char *filename_out = malloc((strlen(filename)+2+3)*sizeof(char));
	
    // Make a copy of the original filename to modify
    strcpy(filename_out, filename);

    // Replace "data" with "output" in the filename
    replace_substring_in_filename(filename_out, "data", quadname);

    printf("Writing quadrature rules to file: %s\n", filename_out);

    // Write header or create the modified file
    //Nrupa's part
    if (io_enabled)
    {
    write_header_if_needed(filename_out);
    }
    else {
        
        printf("I/O is disabled. Running in performance test mode.\n");
    }
    // read existent quad rules from output file
    // Open the file
    FILE *file_output = fopen(filename_out, "r");
    (read = getline(&line, &len, file_output));

    pass = 1;

    readfile_out:
    index = 0;

    // Read the file line by line (data lines)
    while ((read = getline(&line, &len, file_output)) != -1) {
        // Remove the newline character at the end of the line (if any)
        line[strcspn(line, "\n")] = '\0';

        // Split the line by ';'
        char *number = strtok(line, ";");
        char *id = strtok(NULL, ";");
        char *nodes_x_str = strtok(NULL, ";");
        char *nodes_y_str = strtok(NULL, ";");
        char *weights_str = strtok(NULL, ";");

        // Check if all fields are found
        if (number && id && nodes_x_str && nodes_y_str && weights_str) {
            // Convert the number to int
            int number_val = atoi(number);  

            /*
            // Variables to hold the number of values parsed
            int nodes_x_count = 0, nodes_y_count = 0, weights_count = 0;
            
            // Parse the comma-separated lists into arrays of doubles
            int *nodes_x_values = parse_comma_separated_integers(nodes_x_str, &nodes_x_count);
            int *nodes_y_values = parse_comma_separated_integers(nodes_y_str, &nodes_y_count);
            double *weights_values = parse_comma_separated_doubles(weights_str, &weights_count);
            
            // Print the parsed data
            printf("Number: %d, ID: %s\n", number_val, id);
            printf("nodes_x: ");
            for (int i = 0; i < nodes_x_count; i++) {
                printf("%d ", nodes_x_values[i]);
            }
            printf("\n");

            printf("nodes_y: ");
            for (int i = 0; i < nodes_y_count; i++) {
                printf("%d ", nodes_y_values[i]);
            }
            printf("\n");

            printf("weights: ");
            for (int i = 0; i < weights_count; i++) {
                printf("%.5f ", weights_values[i]);
            }
            printf("\n");
            */
           if(pass == 2){
                ids_out[index] = malloc(strlen(id) + 1);
                strcpy(ids_out[index], id);
                numbers_out[index] = number_val;
           }
           index++;
        } else {
            printf("Error parsing line: %s\n", line);
        }
    }
    if(pass == 1){
        printf("Completed first pass on output file\n");
        printf("Found %d valid data sets\n", index);
        pass = 2;
        num_out_ids = index;
        ids_out = malloc(num_out_ids * sizeof(char *));
        numbers_out = malloc(num_out_ids * sizeof(int));
        fseek(file_input, 0, SEEK_SET);
        (read = getline(&line, &len, file_output));
        goto readfile_out;
    } else if(pass == 2){
        printf("Completed second pass on output file\n");
        printf("Extracted ids\n");
    }
    fclose(file_output);  // Close the file when done

    // Loop over all indices in the input file. 
    printf("num_in_ids %d\n",num_in_ids);
    printf("num_out_ids %d\n",num_out_ids);

    for (int i = 0; i < num_in_ids; i++) {
        in_out_map[i] = -1;
    }

    int num_in_out = 0;
    // Loop through all IDs in ids_in and check if they exist in ids_out
    for (int i = 0; i < num_in_ids; i++) {
        int idx_in_output = find_id_in_output(ids_in[i], ids_out, num_out_ids);
        if (idx_in_output != -1) {
            // If the ID exists in ids_out, store the index in in_out_map
            in_out_map[i] = idx_in_output;
            // printf("ID %s found at position %d in input and position %d in output.\n", ids_in[i], i, idx_in_output);
            num_in_out++;
        } else {
            // If the ID does not exist in ids_out, print a message
            // printf("ID %s found at position %d in input but not found in output.\n", ids_in[i], i);
        }
    }

    printf("Found %d datasets in output\n", num_in_out);
    printf("Generating quadrature rules for %d datasets\n", num_in_ids-num_in_out);

    int num_generated = 0;    
    quadType int_type = Volume;
    Poly poly;

    file_output = fopen(filename_out, "a");
    file_input = fopen(filename, "r");
    (read = getline(&line, &len, file_input)); // skip header
    for (int i = 0; i < num_in_ids; i++) {
        read = getline(&line, &len, file_input);
        if(in_out_map[i] == -1){
            // Remove the newline character at the end of the line (if any)
            line[strcspn(line, "\n")] = '\0';

            // Split the line by ';'
            char *number = strtok(line, ";");
            char *id = strtok(NULL, ";");
            char *exp_x_str = strtok(NULL, ";");
            char *exp_y_str = strtok(NULL, ";");
            char *coeff_str = strtok(NULL, ";");

            // Check if all fields are found
            if (number && id && exp_x_str && exp_y_str && coeff_str) {
                // Convert the number to int
                int number_val = atoi(number);  

                // Variables to hold the number of values parsed
                int exp_x_count = 0, exp_y_count = 0, coeff_count = 0;

                // Parse the comma-separated lists into arrays of doubles
                int *exp_x_values = parse_comma_separated_integers(exp_x_str, &exp_x_count);
                int *exp_y_values = parse_comma_separated_integers(exp_y_str, &exp_y_count);
                double *coeff_values = parse_comma_separated_doubles(coeff_str, &coeff_count);
                
                int* exponents = NULL;
                double* coefficients = NULL;
                poly.dimension = 2;

                exponents = (int*)malloc(2 * exp_x_count * sizeof(int));
                if (exponents == NULL) {
                    printf("Memory allocation failed\n");
                    return 1;
                }
                coefficients = (double*)malloc(exp_x_count * sizeof(double));
                if (coefficients == NULL) {
                    printf("Memory allocation failed\n");
                    free(exponents);
                    return 1;
                }
                for(int j = 0; j < exp_x_count; j++){
                    exponents[2 * j] = exp_x_values[j];
                    exponents[2 * j + 1] = exp_y_values[j];
                    coefficients[j] = -coeff_values[j];
                }

                free(exp_x_values);
                free(exp_y_values);
                free(coeff_values);

                poly.exp = exponents;
                poly.coef = coefficients;
                poly.size = exp_x_count;

                int int_points = exp_x_count;

                // Redirect stdout to null to not clutter the output
                freopen("NUL", "w", stdout); // On Windows, use "NUL"; on Unix-like systems, use "/dev/null"
                if (!stdout) {
                    perror("Failed to redirect stdout");
                    return 1;
                }             

                QuadScheme algoimscheme = call_quad_multi_poly(poly, int_points, QUAD_ORDER, int_type);

                
                // Restore the original stdout
                freopen("CON","w",stdout);

                double* nodes_x = malloc(algoimscheme.size * sizeof(double));
                if (nodes_x == NULL) {
                    printf("Memory allocation failed\n");
                    free(exponents);
                    return 1;
                }
                double* nodes_y = malloc(algoimscheme.size * sizeof(double));
                if (nodes_y == NULL) {
                    printf("Memory allocation failed\n");
                    free(exponents);
                    return 1;
                }
                double* weights = malloc(algoimscheme.size * sizeof(double));
                if (weights == NULL) {
                    printf("Memory allocation failed\n");
                    free(exponents);
                    return 1;
                }

                for(int j = 0; j < algoimscheme.size; j++){
                    nodes_x[j] = algoimscheme.nodes[2 * j];
                    nodes_y[j] = algoimscheme.nodes[2 * j + 1];
                    weights[j] = algoimscheme.weights[j];
                }
                
                fprintf(file_output, "%s;%s", number, id);
                char* nodes_x_str = double_array_to_csv(nodes_x, algoimscheme.size, 16);
                fprintf(file_output, ";%s", nodes_x_str);
                char* nodes_y_str = double_array_to_csv(nodes_y, algoimscheme.size, 16);
                fprintf(file_output, ";%s", nodes_y_str);
                char* weights_str  = double_array_to_csv(weights, algoimscheme.size, 16);
                fprintf(file_output, ";%s", weights_str);
                fprintf(file_output, "\n");
                fflush(file_output);                

                in_out_map[i] = num_generated;
                num_generated++;
                num_in_out++;                

                free(exponents);
                free(coefficients);
                free(nodes_x);
                free(nodes_y);
                free(weights);
                free(nodes_x_str);
                free(nodes_y_str);
                free(weights_str);

                free(algoimscheme.nodes);
                free(algoimscheme.weights);
            } else {
                printf("Error parsing line: %s\n", line);
            }
        }
        if(i % 100 == 0){
            print_progress_bar(i, num_in_ids);
        }
    }
    fprintf(file_output, "\n");
    printf("Generated %d quadrature rules\n", num_generated);

        
    // After the loop finishes:
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\nProcessing time: %f seconds\n", elapsed_time);

    // Otherwise generate the quadrature rule by calling to algoim.

    // Append the rule to the output file.

    // tear down
    free(ids_in);
    free(numbers_in);
    free(ids_out);
    free(numbers_out);
    free(in_out_map);
    fclose(file_input);  // Close the file when done
    fclose(file_output);  // Close the file when done

    return 0;  // Success

}