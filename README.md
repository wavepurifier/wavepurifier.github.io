
# Welcome to WavePurifier, a diffusion based audio adversarial defense approach.

## 1. Prepare the environment

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone git@github.com:wavepurifier/wavepurifier.github.io.git
cd wavepurifier.github.io
```

### Step 2: Create the Conda Environment

You can recreate the environment by running the following command:

```bash
conda env create -f environment.yml
```

This will create a Conda environment with all the necessary dependencies as specified in the `environment.yml` file.

### Step 3: Activate the Environment

Once the environment is created, activate it:

```bash
conda activate torch1.6
```
---
## 2. Download the Diffusion Model

### Step 1: Download the Pre-trained Diffusion Model

We have a pre-trained diffusion model that can be used directly. You can download it by following these steps:

1. [Download the pre-trained model here](https://drive.google.com/drive/folders/14dh-aDxIncLd-cviZAmqvZ-uld8zrBTJ?usp=sharing) and place it in the `pretrained/guided_diffusion` folder in your local repository.


2. Once the model is in place, you can use it directly in your project for inference or further experiments.

### Optional: Train Your Own Diffusion Model

If you want to train your own diffusion model from scratch, please refer to the official OpenAI Guided Diffusion repository: [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion).

To train the diffusion model, you need to set the following parameters:
- `diffusion_step = 1000`
- `image_size = 256`

Additionally, you need to prepare your own **audio spectrogram** data. Each spectrogram should have the shape of `256 x 256` (indicating Frequency * Time). The spectrogram represents the audio in the frequency domain over time and will be used as input to the model.

```
TRAIN_FLAGS="--lr 1e-4 --batch_size 1 --image_size=256"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
python scripts/image_train.py --data_dir ~/data/TIMIT_spec_256/TRAIN $DIFFUSION_FLAGS $TRAIN_FLAGS
```
---
## 3. Launch the Purification
You can specify the input and output wav file. Also, you can change the purification noise by modify `t_s, t_m, t_l` in `config.py`.

Next run:
```
python purify.py
```
---
## 4. [Optional] Craft more adversarial audios

### 1. **C&W Attack**

To build and use the **C&W attack**:
- Refer to the official GitHub repository: [https://github.com/carlini/audio_adversarial_examples](https://github.com/carlini/audio_adversarial_examples).
- This attack optimizes perturbations in the audio domain to force the model to misclassify the input audio.

You can follow the steps provided in the repository to integrate this attack into your project.

### 2. **SpecPatch Attack**

To implement the **SpecPatch attack**:
- Read the paper: [SpecPatch: A New Adversarial Attack on Audio Models](https://cse.msu.edu/~qyan/paper/SpecPatch_CCS22.pdf).
- SpecPatch is built on top of the C&W attack with the following modifications:
  1. **Frequency Filter & RIR Filter**: Adds these filters to the perturbation, allowing for more realistic adversarial examples.
  2. **Changed Optimization Goal**: The optimization goal is modified to include **empty symbols**, making the patch affect longer transcription.
  3. **Optimizing with Different Backgrounds**: SpecPatch takes into account different background noises during optimization to improve robustness.

### 3. **QIN-I Attack**

To implement the **QIN-I attack**:
- Refer to the official CleverHans repository: [https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/adversarial_asr](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/adversarial_asr).
- The QIN-I attack demonstrates a state-of-the-art approach for attacking speech recognition models by generating adversarial inputs that force incorrect transcription.

Follow the instructions in the CleverHans repository to implement and use this attack in your system.

---


