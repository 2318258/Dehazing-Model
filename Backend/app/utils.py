import tensorflow as tf

def ssim_metric(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) metric.

    Args:
        y_true (tf.Tensor): Ground truth image.
        y_pred (tf.Tensor): Predicted image.

    Returns:
        tf.Tensor: Mean SSIM value.
    """
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))