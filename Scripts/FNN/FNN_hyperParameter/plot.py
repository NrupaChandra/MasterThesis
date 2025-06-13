import re
import matplotlib.pyplot as plt

# Path to the log file
log_file_path = "/work/home/ng66sume/MasterThesis/Scripts/FNN_V5.1/Test.out.47756567.txt"


# Regular expression to match finished trials and their final val_loss
pattern = re.compile(
    r"Trial\s+(\d+)\s+finished\s+with\s+parameters:.*?and final val_loss:\s*([0-9\.eE+-]+)"
)

trial_numbers = []
val_losses = []

# Read and parse the log file
with open(log_file_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            trial_num = int(match.group(1))
            val_loss = float(match.group(2))
            trial_numbers.append(trial_num)
            val_losses.append(val_loss)

# Sort by trial number
sorted_data = sorted(zip(trial_numbers, val_losses), key=lambda x: x[0])
trials_sorted, losses_sorted = zip(*sorted_data)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(trials_sorted, losses_sorted, label="Final Val Loss")
# Mark the best trial
min_loss = min(losses_sorted)
best_trial = trials_sorted[losses_sorted.index(min_loss)]
plt.scatter([best_trial], [min_loss], marker="*", s=200, label=f"Best Trial ({best_trial})")
plt.xlabel("Trial Number")
plt.ylabel("Final Validation Loss")
plt.yscale("log")
plt.title("Final Validation Loss per Trial")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()

# Save the figure instead of showing it
output_path = "final_val_loss_per_trial.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to {output_path}")