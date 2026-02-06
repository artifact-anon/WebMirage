# Optimization & Attack Scripts

This framework provides three distinct optimization strategies for generating adversarial patches. These scripts are available for each supported VLM (LLaVA, MiniCPM-o, Phi-3).

Choose the script that best balances your need for convergence speed versus attack robustness.

## Script Variants

### 1. Fixed Position Attack
* **Script:** `train_with_mask.py`
* **Description:** Optimizes a perturbation in a single, fixed location on the image.
* **Use Case:** Best for initial testing and debugging. It converges the fastest but is the least robust to spatial changes (e.g., scrolling or resizing).

### 2. Two-Position Patch
* **Script:** `train_with_patch_two_position.py`
* **Description:** Optimizes the patch to be effective across two different spatial coordinates simultaneously.
* **Use Case:** A balanced approach that improves transferability without significantly increasing training instability.

### 3. Three-Position Patch (Recommended for Robustness)
* **Script:** `train_with_patch_three_position.py`
* **Description:** Optimizes the patch to be effective across three distinct positions.
* **Trade-off:**
    * **Pros:** Produces the most robust adversarial attacks, capable of withstanding significant shifts in the agent's viewport.
    * **Cons:** Harder to converge. Requires careful tuning of hyperparameters (learning rate, iterations) and may take longer to find a successful adversarial example. 