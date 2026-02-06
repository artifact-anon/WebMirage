# Red Teaming Framework for VLM Web Agents

A modular framework designed for red teaming and evaluating the robustness of Vision-Language Models (VLMs). This project specifically focuses on generating adversarial inputs and testing the security of web agents powered by multimodal models.

## Critical Prerequisites

**Before installing this framework, you must manually configure the base models.**

Due to conflicting library versions and environment requirements, you need to set up the environments for **MiniCPM-o**, **LLaVA**, and **Phi-3** separately and download their model weights to your local machine.

### 1. MiniCPM-o
* **Repo:** [OpenBMB/MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)
* **Setup:** Clone the repo, create a dedicated environment (e.g., `conda create -n minicpm`), and install dependencies.
* **Weights:** Download the official weights locally and record the path.

### 2. LLaVA
* **Repo:** [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
* **Setup:** Clone the repo and create a dedicated environment (e.g., `conda create -n llava`).
* **Weights:** Download LLaVA-v1.5/v1.6 weights locally and record the path.

### 3. Phi-3 Vision
* **Repo:** [microsoft/Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
* **Setup:** Ensure you have the compatible `transformers` version installed in a dedicated environment.
* **Weights:** Download the `Phi-3-vision-128k-instruct` weights locally.

---

## Workflow Example: LLaVA

After ensuring you have created a unique Conda environment for each model above, follow these steps to run the training workflow. We will use LLaVA as the primary example.

### 1. Clone LLaVA to Home Directory
You must clone the official LLaVA repository to your **home directory** (`~/`), as the scripts rely on local code references from this repository.

```bash
cd ~
git clone [https://github.com/haotian-liu/LLaVA.git](https://github.com/haotian-liu/LLaVA.git)
```

### 2. Configure the Training Script
Navigate to the training script located at: `scripts/LLaVA/train_with_mask.sh`

Open the file and configure each parameter's path (model paths, data paths, output directories) to match your local environment configuration.

3. Run the Script
Once the paths are configured, execute the script:


```bash
bash scripts/LLaVA/train_with_mask.sh
```

