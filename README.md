This code was adapted from Denny Britz's implementation of a CNN for Text Classification, which can be found here https://github.com/dennybritz/cnn-text-classification-tf

One key difference is that there are two additional datasets that can be used with this code: Reddit dataset called SARC ( Self-Annotated Reddit Corpus) and Debanjan Ghosh's Twitter sarcasm dataset.

Download these files from here:
Twitter: https://drive.google.com/drive/folders/1-AD928kZess59nSsClDkWfagu8ErPaOT?usp=sharing
Reddit: https://drive.google.com/file/d/1CBWn5nSaKHaXrZNVE4R1XdWg8vbsB0AC/view?usp=sharing

Then place them in the data folder before running and make sure to add the dataset parameter to each of the commands

For Twitter data
```
python train.py --dataset=ghosh
python eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/ --dataset=ghosh
```
For Reddit data
```
python train.py --dataset=sarc
python eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/ --dataset=sarc
```

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py --dataset=ghosh
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/ --dataset=ghosh"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
