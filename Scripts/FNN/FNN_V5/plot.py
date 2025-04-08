import re
import matplotlib.pyplot as plt

# Lists to store trial information
trial_numbers = []
val_losses = []
trial_types = []  # will be "pruned" or "finished"

# Open and parse the output file line by line
with open("Test.out.47729818.txt", "r") as f:
    for line in f:
        if line.startswith("Trial") and "val_loss:" in line:
            # Extract trial number (assumes format "Trial <number> ...")
            parts = line.split()
            trial_num = int(parts[1])
            
            # Determine trial type: pruned or finished
            if "pruned" in line:
                ttype = "pruned"
            else:
                ttype = "finished"
                
            # Extract validation loss using regex
            match = re.search(r"val_loss:\s*([\d\.e\-\+]+)", line)
            if match:
                loss = float(match.group(1))
            else:
                continue

            trial_numbers.append(trial_num)
            val_losses.append(loss)
            trial_types.append(ttype)

# Identify the best finished trial (lowest val_loss among finished trials)
finished_trials = [(tn, loss) for tn, loss, typ in zip(trial_numbers, val_losses, trial_types) if typ == "finished"]
if finished_trials:
    best_trial_num, best_loss = min(finished_trials, key=lambda x: x[1])
else:
    best_trial_num, best_loss = None, None

# Prepare the plot
plt.figure(figsize=(10, 6))



# Plot finished trials (blue dots)
finished_x = [tn for tn, typ in zip(trial_numbers, trial_types) if typ == "finished"]
finished_y = [loss for loss, typ in zip(val_losses, trial_types) if typ == "finished"]
plt.scatter(finished_x, finished_y, color='blue', label="Finished Trials")

# Mark the best trial (green star)
if best_trial_num is not None:
    plt.scatter([best_trial_num], [best_loss], color='green', s=150, marker='*', label="Best Trial")

# Customize the plot
plt.xlabel("Trial Number")
plt.ylabel("Validation Loss")
plt.title("Validation Loss per Trial")
plt.ylim(-1, 1.5)  # Set y-axis limits
plt.legend()
plt.grid(True)

# Save the plot to a file instead of displaying it
plt.savefig("validation_loss_plot.png")
plt.close()
