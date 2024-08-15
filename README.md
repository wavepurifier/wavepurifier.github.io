## 1. Prepare Attack
Switch the conda Env
```
conda activate DL
```
### 1.1 CW attack: (nc)
cd ~/SpecPatch/ctc_attack/audio_adversarial_examples/
```cfgrlanguage
python attack_carlini.py --in source_audio/SA1.WAV.wav 
--target "open the door" --out nc_3_1_1000.wav --lr 20 --iterations 1000 
--restore_path DeepSpeech/deepspeech-0.4.1-checkpoint/model.v0.4.1
```
To generate many AEs, run the following script

```cfgrlanguage
bash generate_CW_AEs.sh
```

### 1.2 SpecPatch attack: (sp)
cd ~/SpecPatch/ctc_attack/audio_adversarial_examples/
```cfgrlanguage
conda activate DL
```
First, generate the muted signals for the original audio
```cfgrlanguage
./generate_SP_Muted.sh
```
This will produce the muted audios for the benign input, and save it to the 
"/data/pure_audio/SP_AEs/Muted/". The detailed process can be found in 
**cd ~/SpecPatch/ctc_attack/audio_adversarial_examples/result2/mute/Muted_**


Next, we need to use the muted audio to generate the SpecPatch Attack:
```cfgrlanguage
./generate_SP_AEs.sh
```
The generated AE process can be found in "/data/pure_audio/SP_AEs/Muted/AE"

### 1.3 Qin-I attack: (qi)
#### 1.3.1 Generate Qin-I AE
```cfgrlanguage
conda activate qinattack
cd ~/intern/adversarial_asr
```
change the **read_data.txt**, then the adversarial example will be generated 
to the **same directory** of the original audio.
```cfgrlanguage
python generate_imperceptible_adv.py
```
Note that the batch_size need to modify based on number of source audio

#### 1.3.2 Classify the audio with Lingvo Model
```cfgrlanguage
conda activate qinattack
python -W ignore::Warning classify.py --inputaudio "/data/pure_audio/QIN_AEs/5_1.wav"
```



## 2. Train the DM
Switch the conda Env
```
conda activate torch1.6
```
2.1 Prepare dataset
```cfgrlanguage
python create_dataset.py
```
The dataset is convert the audios from TIMIT to the TIMIT_spec Folder, saved to the __/data/TIMIT_spec__

2.2 Train a Diffusion Model

cd  **~/intern/guided_diffusion/**
```cfgrlanguage
TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
python scripts/image_train.py --data_dir /data/TIMIT_spec/TRAIN $DIFFUSION_FLAGS $TRAIN_FLAGS
```
For image size as 128*128, simply add --image_size 128 at the end of the previous command
```cfgrlanguage
TRAIN_FLAGS="--lr 1e-4 --batch_size 1"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
python scripts/image_train.py --data_dir /data/TIMIT_spec_middle/TEST $DIFFUSION_FLAGS $TRAIN_FLAGS --image_size=128
```

For server, the trained model is saved in ~/code/guided-diffusion/trained_model


2.3 Copy Trained model to the local
```cfgrlanguage
$ cd /tmp/openai-[data]
$ cp model.pt ~/intern/PureAudio/pretrained/guided_diffusion/
```
2.4 Change the load checkpoint path
Go to the __PurAudio/diffmodel.py__, change model140000.pt to your model.pt
```cfgrlanguage
model.load_state_dict(torch.load(f'{model_dir}/model140000.pt', map_location='cpu'))
```
2.5 Change config file accordingly
    
1. create config file in PureAudio/config/XXXX.yml
2. change the middle.yml model.image_size to 128
3. change PureAudio/configs.py --config to XXXX.yml

## 3. Purify the AE
### TODO: Merge the purify scripts to one

cd ~/intern/PureAudio/
```cfgrlanguage
conda activate torch1.6
```

Purify single AE with 256 window
```
python purify_256.py --t 5 --config _256.yml
``` 

This will do a purification for all the AEs, and save the purified spectrogram in 
**~/intern/PureAudio/logs/[foldername]**
```cfgrlanguage
[foldername] = [purify window]_[attack]_[purify_depth]_[original audio]_[target text]
256_CW_40_3_5 = [256]_[CW]_[40]_[3]_[5]
```

-------------------------------------------------------------------------
__[Deprecated!!]__ Purify single AE with 128 window
```
python main.py --t 10 --config XXX.yml
```
__[Deprecated!!]__ Purify AE with hierarchy window:
```cfgrlanguage
python hierarchy_purify.py --t 5 --config _256.yml
```
Note here _256.yml only set the window size, however, the size can be controlled in the purify process by set:
```cfgrlanguage
config.data.dataset = '_'+str(purify_level)
```
----------------------------------------------------------------------------------

## 4. Evaluate the purification
### 4.1 Evaluate on the DeepSpeech Model
Once the audio is purified by models, it will generate purified audios in ```
```
/data/pure_audio/
```
use:
```cfgrlanguage
conda activate DL
```

Change the purified path in the __evaluation_wer.py__, and run 
```cfgrlanguage
python evaluation_wer.py
```
This script generate transcriptions in ****, the 
```cfgrlanguage
Purified transcription: /data/pure_audio/texts/
Purified wer cer: /data/pure_audio/EER_result/
```

### 4.2 Evaluate on the Lingvo Model
Complete this part before go home!!!
```cfgrlanguage
conda activate qinattack
```



## 5. Optimize the t
```cfgrlanguage
conda activate torch1.6
```
### 5.1 Optimize a fixed t for the global:
```cfgrlanguage
cd ~/intern/PureAudio/
python -W ignore::Warning optimize_global.py --config _256.yml
```
This will do a purification for all the AEs, and save the purified spectrogram in 
**~/intern/PureAudio/logs/[foldername]**
```cfgrlanguage
[foldername] = [purify window]_[attack]_[purify_depth]_[original audio]_[target text]
256_CW_40_3_5 = [256]_[CW]_[40]_[3]_[5]
```
Then, in each folder, it will have the following items
```cfgrlanguage
init_0_40_0.png: original image; 40 steps noise; at first(idx=0) window
original_input_0.png: origianl image; at first(idx=0) window
samples_0_0_png: purified image; at first(idx=0) window
```
Next, this script re-construct audios based on the spectrograms, and save the audios in **[/data/pure_audio/purified_audio/QIN_global]**.

For DeepSpeech API, we use 
```
python eva_deepspeech.py
```
to get the transcriptions, and the metrics for WER, CER in both ep1 and ep2 setting. The transcription is saved in 
```cfgrlanguage
/data/pure_audio/transcription
```
The ep1, ep2 result is saved in the script's folder.

If we use the Lingvo Model to recognize, we need to 
```cfgrlanguage
conda activate qinattack
cd ~/intern/adversarial_asr/
python classify_many.py
```

Finally, the 
```cfgrlanguage
optimize_global.collect_para() 
```

Last, use
```cfgrlanguage
python python_figures/optimize_t.py
```

save the metric result to **~/intern/PureAudio/results**.
### 5.2 Optimize the hierarchy t:
```cfgrlanguage
cd ~/intern/PureAudio/
python -W ignore::Warning optimize_hier.py
```

For **DeepSpeech** model, run
```cfgrlanguage
python eva_deepspeech.py
```

For **Lingvo** model 
```cfgrlanguage
conda activate qinattack
cd ~/intern/adversarial_asr/
python classify_many.py
```
Finally, the 
```cfgrlanguage
optimize_hier.collect_para() 
```
Last, use
```cfgrlanguage
python python_figures/optimize_ht.py
```

### 6 Evaluation on Commercial APIs
### 6.1 Evaluate attacks
Devils whisper attack:
```
/data/pure_audio/Devil-Whisper-Attack-master/AEs/*.wav
```

Phantom attack:
```
/data/Phantom_USENIX/AE4Diffusion/*/*/*.wav
```

Evaluate the Attacks on different Speech2text APIs:

```
conda activate torch1.7
cd ~/NIH_SimAmp/ASRs/
```
Then, modify the attack audio dir path, and run:
```
python IBM_API.py
```
Then transcription will be saved at __~/NIH_SimAmp/ASRs/ASR\_\[attack name]\/\[APIname]_\[attack].txt__

### 6.2 Evaluate defenses


```
python purify_256.py --t 40 --config _256.yml
```

### 6.3 Evaluate WaveGuard defense
```
cd ~/intern/PureAudio/waveguard_defense/DEfender
conda activate torch1.6
```
For __Downsampling__ defense
```
python defender_multiple.py --in_dir /data/pure_audio/Devil-Whisper-Attack-master/AEs/ --out_base /data/pure_audio/waveGuard_defended --defender_type downsample_upsample --defender_hp 2000
```

For __LPC__ defense
```
python defender_multiple.py --in_dir /data/pure_audio/Devil-Whisper-Attack-master/AEs/ --out_base /data/pure_audio/waveGuard_defended --defender_type lpc --defender_hp 10
```
For __Quantization__ defense
```
python defender_multiple.py --in_dir /data/pure_audio/Devil-Whisper-Attack-master/AEs/ --out_base /data/pure_audio/waveGuard_defended --quant --defender_hp 8
```

### 6.4 Evaluate Noise reduction defense
Same directory:
```
python noise_reduce.py
```

