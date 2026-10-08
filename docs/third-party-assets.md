# External assets and third-party dependencies

The project MIT license covers the project source, project documentation
(including the paper-derived README Figure 1), and the 90 certified proposals
stored as `q_final` JSON artifacts. The five joint rightsholders are
Paweł Lorek, Rafał Nowak, Rafał Topolnicki, Tomasz Trzciński, and Maciej Zięba.
The grant excludes the prepared g7 tensor, ResNet weights, ImageNetV2 image
bytes, and third-party dependency code or assets.

## External g7 inputs

The standalone release does not bundle, redistribute, or host the prepared g7
tensor, torchvision ResNet-18 pretrained weights, or ImageNetV2 image bytes.
Users who run g7 supply the following exact files through explicit local paths:

- `imagenetv2-matched-frequency-format-val_first24.pt`, 7,227,208 bytes,
  SHA-256
  `3eed840225285c10291673dea3a247504d6c225fd617648ba7b139ee5aa63c32`;
- `resnet18-f37072fd.pth`, 46,830,571 bytes, SHA-256
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

The evaluator checks these identities before use and remains offline. It does
not download either file or fall back to a torchvision cache. The earlier
preparation script has status `DOES_NOT_EXACTLY_REPRODUCE`: it does not exactly
reproduce the required prepared tensor. Supplying a local file does not make
that file part of this project or place it under this project's license.

## Python dependencies

The project declares NumPy, PyYAML, scikit-learn, and PyTorch as runtime
dependencies; torchvision is an optional g7 dependency, and pytest is a
development dependency. Their code and assets are not relicensed under this
project's MIT license. Each remains subject to its own upstream license and
terms. The project source tree does not vendor those dependency bytes.
