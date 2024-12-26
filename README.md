# Neural blur classifier

Pragmatically sorting through my home lab. A simple neural classifier, trained on [acozma/imagenet-1k-rand_blur](https://huggingface.co/datasets/acozma/imagenet-1k-rand_blur) from Hugging Face.

### To train:
```bash
python classifier.py train --steps 1000
```

### To run:
```bash
python classifier.py inf --source /path/to/images 
                         --output /path/to/output
```
