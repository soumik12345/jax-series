import os


def setup_tpu():
    if "google.colab" in str(get_ipython()) and "COLAB_TPU_ADDR" in os.environ:
        # Make sure the Colab Runtime is set to Accelerator: TPU.
        import requests

        if "TPU_DRIVER_MODE" not in globals():
            url = (
                "http://"
                + os.environ["COLAB_TPU_ADDR"].split(":")[0]
                + ":8475/requestversion/tpu_driver0.1-dev20191206"
            )
            resp = requests.post(url)
            TPU_DRIVER_MODE = 1

        # The following is required to use TPU Driver as JAX's backend.
        from jax.config import config

        config.FLAGS.jax_xla_backend = "tpu_driver"
        config.FLAGS.jax_backend_target = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        print("Registered TPU:", config.FLAGS.jax_backend_target)
    else:
        print('No TPU detected. Can be changed under "Runtime/Change runtime type".')
