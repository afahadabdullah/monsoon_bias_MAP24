# Model checkpoints

No trained weights were included in the archive. The notebooks use:

- `best_model_weights.h5`: baseline CNN.
- `best_model_weights2.h5`: augmented seven-channel CNN.
- `best_model_weights_unet.h5`: deep-supervision experiment.

Place compatible weights here or set `PRECIP_WEIGHTS_DIR`. Training may write
checkpoints to this directory. The augmented CNN's training cell is commented
out and its later weight load is unconditional, so supply its weights or enable
and run training first. Checkpoints are ignored by Git.

The original filenames are retained. Newer Keras releases can require the
`.weights.h5` suffix for `save_weights_only=True`; adapt all load/save references
together or use the original compatible TensorFlow/Keras environment. Renaming
a file alone does not guarantee checkpoint compatibility. No dependency lock
or checkpoint-format metadata was supplied.
