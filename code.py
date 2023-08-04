import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components

EXPERIMENT_NAME = 'digit recognition note pipeline'        # Name of the experiment in the UI
BASE_IMAGE = '****/modeltraining:1.6'           
VOLUME_NAME = "modeltraining-data"
VOLUME_MOUNT_PATH = "/mnt/raw_data"

@dsl.python_component(
    name='Training',
    description='Training the model.',
    base_image=BASE_IMAGE,
    volume_mounts=[dsl.VolumeMount(volume_name=VOLUME_NAME, mount_path=VOLUME_MOUNT_PATH)]
)
def training() -> None:
    import tensorflow as tf
    import numpy as np
    import gzip, os
    import mlflow
    from mlflow.keras import autolog

    os.environ["MLFLOW_TRACKING_URI"] = " "       #replace
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = ""     #replace
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "admin123"

    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIR = f"{VOLUME_MOUNT_PATH}/model"
    DATA_DIR = f"{VOLUME_MOUNT_PATH}/data"

    def load_local_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_local_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    train_images = load_local_mnist_images(f'{DATA_DIR}/train-images-idx3-ubyte.gz')
    train_labels = load_local_mnist_labels(f'{DATA_DIR}/train-labels-idx1-ubyte.gz')

    # Load test images and labels from local files
    test_images = load_local_mnist_images(f'{DATA_DIR}/t10k-images-idx3-ubyte.gz')
    test_labels = load_local_mnist_labels(f'{DATA_DIR}/t10k-labels-idx1-ubyte.gz')

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Start the MLflow run
    mlflow.start_run()

    # Enable autologging for Keras to automatically log parameters and metrics
    autolog()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model (no need to explicitly log metrics, autolog will handle it)
    model.fit(train_images, train_labels, epochs=20)

    # Evaluate the model (no need to explicitly log metrics, autolog will handle it)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    # Save the model and log the model artifact
    model.save(MODEL_DIR)
    mlflow.log_artifacts(MODEL_DIR, artifact_path='model')

    # End the MLflow run
    mlflow.end_run()

training_op = components.func_to_container_op(
    training,
    base_image=BASE_IMAGE, 
)

dsl.pipeline(
   name='Training pipeline',
   description='A Pipeline to train a model.'
)
def training_pipeline():
    add_task = training_op()

kfp.Client().create_run_from_pipeline_func(
    training_pipeline, arguments={}, 
    experiment_name=EXPERIMENT_NAME
)
----


data_loading_task = data_loading_op(data_path)
                            .add_volume(k8s_client.V1Volume(name='data-processing')) \
                            .add_volume_mount(k8s_client.V1VolumeMount(
