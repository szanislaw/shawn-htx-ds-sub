# shawn-htx-ds-sub

## Submission of HTX Take Home Technical Test Questions

This repository contains all notebooks and codebases, but not the datasets and finetuned model.


**Setting Up:**
The repository consists of 3 main folders:
**asr, asr-train and hotword-detection**

These 3 folders correspond to the 3 tasks as stated in the question paper.

Additional files include **training-report.pdf** (for Task 4) and **essay-ssl.pdf** (for Task 5) and a **cmd.txt** for the running of the Docker task (for Task 1) as well as a requirements.txt, which provides all the necessary packages needed to run the code.

The custom common-voice dataset provided within the question paper should be downloaded and extracted into the folder in which this repository is cloned. 

Additionally, the fine-tuned model can be accessed publicly here: https://huggingface.co/szanislaw/wav2vec2-large-960h-cv and should be cloned into the asr-train folder for usage.


Task 1:
To run Task 1, cd to the asr directory and run the two commands from cmd.txt, namely:
```
docker build -t asr-api -f asr.Dockerfile .

docker run --gpus all -p 8001:8001 asr-api
```

Task 2:
All tasks are in the notebook, with explanations with respect and reference to the report (training-report.pdf)

Task 3:
All tasks are in the notebook, with the corresponding explanations for each cell.

