# Automatic Speech Recognition project

Based on [Deep Speech 2](https://arxiv.org/abs/1512.02595) implementation

## Installation guide

Load trained model:
```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r ./requirements.txt
python load_model.py
```

## Reproduction
Train for 79 epochs with config hw_asr/configs/deepspeech2_train.json (train-clean-100)

Resume and train until 253 epochs reached with config hw_asr/configs/deepspeech2_fine.json (train-other-500)
```
python train.py -c [config]
```

## Test
LibriSpeech __test-clean__:
```
python test.py -c hw_asr/configs/deepspeech2_test_clean.json -r model_best.pth
```

```
WER (argmax) 0.23683128579124377
CER (argmax) 0.0693261912516432
WER (beam search + LM) 0.068444553594604
CER (beam search + LM) 0.025498943780145625
```

LibriSpeech __test-other__:
```
python test.py -c hw_asr/configs/deepspeech2_test_other.json -r model_best.pth
```
```
WER (argmax) 0.3892966125238747
CER (argmax) 0.14703915366469356
WER (beam search + LM) 0.26958574306431143
CER (beam search + LM) 0.11945911234636582
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
