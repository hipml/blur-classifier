# Neural blur classifier

Pragmatically sorting through my home lab. A simple neural classifier, trained on [acozma/imagenet-1k-rand_blur](https://huggingface.co/datasets/acozma/imagenet-1k-rand_blur) from Hugging Face.

## To Run

### Environment

To build and run the project, first initialize your environment.

#### PIP:
```bash
python -m venv .myvenv
source .myvenv/bin/activate
pip install -r requirements.txt
```

Note: it is _highly_ recommended to use CUDA. Ensure you are installing the GPU-accelerated versions of PyTorch.

### To train:
```bash
python classifier.py train --steps 1000
```

### To run:
```bash
python classifier.py inf --source /path/to/images  # image directory to be sorted
                         --output /path/to/output  # sorted images are cp'd to here in `normal`, `blurred` subdirs
                         --threshold 0.8           # confidence threshold for classifying an image as blurry (range: 0 to 1. default: 0.8)
```

## Contributing

 [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com) 
 
 * Thanks to [@mlindgren](https://github.com/mlindgren) for adding HEIC support
