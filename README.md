# CNN camera dataset

## Directory layout

```text
cnn_camera_capture/
├── capture_images.py
├── record_video.py
├── extract_frames.py
├── raw_videos/
│   ├── 가위/
│   ├── 국자/
│   ├── 뒤집개/
│   ├── 집게/
│   └── 칼/
├── raw_dataset/
│   ├── 가위/
│   ├── 국자/
│   ├── 뒤집개/
│   ├── 집게/
│   └── 칼/
└── dataset/
    ├── train/<class>/
    ├── val/<class>/
    └── test/<class>/
```

`raw_videos` stores original recordings. `raw_dataset` stores extracted
frames. Copy frames into `dataset` only after deciding the train, validation,
and test split.

## Record a video

The preview starts first. Press `r` to begin recording, then press `q` or
`ESC` to stop and save. Pressing `ESC` before recording exits without saving.

```bash
python3 record_video.py raw_videos/칼
python3 record_video.py raw_videos/가위 --camera 0 --width 640 --height 480 --fps 30
```

## Extract frames

The following command saves one image for every 10 video frames:

```bash
python3 extract_frames.py \
  raw_videos/칼/video_YYYYMMDD_HHMMSS.mp4 \
  raw_dataset/칼 \
  --every-n-frames 10
```

Useful options:

```bash
# Extract every 30 frames.
python3 extract_frames.py INPUT.mp4 OUTPUT_DIR --every-n-frames 30

# Extract only frames 300 through 1800.
python3 extract_frames.py INPUT.mp4 OUTPUT_DIR \
  --every-n-frames 10 --start-frame 300 --end-frame 1800

# Stop after 200 output images.
python3 extract_frames.py INPUT.mp4 OUTPUT_DIR \
  --every-n-frames 10 --max-images 200
```

## Training tips

1. Record several separate videos for each class, changing background,
   distance, angle, lighting, and object orientation.
2. Keep the number of images per class similar. A large class imbalance can
   make accuracy misleading.
3. Do not randomly split neighboring frames from one video across train and
   validation. Split by recording session so nearly identical frames cannot
   leak into both sets.
4. A practical starting split is train 70%, validation 15%, and test 15%,
   based on whole videos rather than individual frames.
5. Adjacent video frames are highly repetitive. Start with one image every
   10 to 30 frames and inspect the result instead of extracting every frame.
6. Remove blurred, empty, badly cropped, and incorrectly labeled images.
   Dataset quality matters more than simply increasing image count.
7. Resize in the training pipeline, commonly to 224x224, while preserving
   original extracted images. Use random crop, small rotation, brightness
   changes, and horizontal flip only when those transforms preserve the
   correct class meaning.
8. Start with transfer learning such as MobileNetV3 or ResNet18 instead of
   training a CNN from scratch. First freeze the backbone, then fine-tune it
   with a lower learning rate.
9. Track validation loss as well as accuracy and use early stopping. Always
   inspect a confusion matrix to find classes the model mixes up.
10. Test with a completely new recording environment before trusting the
    model in real camera input.

## Train the kitchen-tool classifier

Every source video is divided into contiguous train, validation, and test
sections. The defaults are 70%, 15%, and 15%. Using contiguous sections is
safer than randomly mixing neighboring frames, although a completely separate
test recording is still recommended for the final evaluation.

```bash
python3 prepare_video_dataset.py --clean \
  --backgrounds 흰색 나무색 \
  --every-n-frames 15 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15

python3 train_classifier.py \
  --data dataset \
  --output models/kitchen_cnn.pt \
  --device cuda
```

The CNN uses ReLU in its hidden layers. It produces raw logits during training
because `CrossEntropyLoss` already includes the log-softmax operation.
Softmax is applied only when probabilities are needed for evaluation and live
camera inference. By default, training augmentation samples rotations from
-180 to +180 degrees so top-down tools can appear at any in-plane orientation.
Horizontal flipping is disabled because mirroring is not equivalent to
rotation. Enable it only when a mirrored tool image is valid for the task:

```bash
python3 train_classifier.py \
  --data dataset \
  --rotation-degrees 180 \
  --horizontal-flip-probability 0
```

Run live camera inference:

```bash
python3 camera_classifier.py \
  --model models/kitchen_cnn.pt \
  --camera 0
```

The preview displays the predicted class and its smoothed probability. Press
`r` to start recording the annotated preview and press `r` again to stop and
save it. Videos are saved under `recordings/classifier/` by default. Press `q`
or `ESC` to exit; an active recording is saved before shutdown.

Use a different directory or output frame rate when needed:

```bash
python3 camera_classifier.py \
  --recordings-dir recordings/experiment_01 \
  --recording-fps 30
```





python3 camera_classifier.py --device cuda --camera 2# AI-Machine-Learning-Project
