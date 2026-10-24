import tensorflow as tf

print(f"Tensorflow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs avaialbele: {len(gpus)}")

for gpu in gpus:
    print(gpu)