import tensorflow as tf
from tensorflow.keras import layers

def unet_encoder_block(input_tensor, num_filters, dropout_rate=0.1):
    """
    U-Net encoder block with Conv2D, BatchNorm, LeakyReLU, and MaxPooling.

    Args:
        input_tensor: Input tensor to the encoder block.
        num_filters: Number of filters for convolutional layers.
        dropout_rate: Dropout rate for regularization.

    Returns:
        A tuple of:
        - output_tensor: Output tensor of the encoder block (after MaxPooling).
        - skip_connection: Output tensor before MaxPooling (used for skip connections).
    """
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = layers.SeparableConv2D(num_filters, (3, 3), dilation_rate=(2,2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    skip_connection = x

    output_tensor = layers.MaxPooling2D((2, 2))(x)

    return output_tensor, skip_connection


class CustomAttentionLayerWithPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, **kwargs):
        super(CustomAttentionLayerWithPositionalEncoding, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.pre_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.post_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.positional_embedding = None

    def build(self, input_shape):
        _, height, width, channels = input_shape
        num_tokens = height * width

        # Initialize positional embedding with zeros for stability
        self.positional_embedding = self.add_weight(
            shape=(num_tokens, channels),  # (256, 128) for (16x16) input
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="positional_embedding"
        )

    def call(self, inputs):
        batch_size, height, width, channels = (
            tf.shape(inputs)[0],
            tf.shape(inputs)[1],
            tf.shape(inputs)[2],
            tf.shape(inputs)[3]
        )
        # Flatten inputs
        flattened_inputs = tf.reshape(inputs, [batch_size, height * width, channels])  # (batch_size, 256, 128)

        # Add positional embedding
        flattened_inputs_with_position = flattened_inputs + self.positional_embedding  # Broadcasting

        # Pre-attention normalization
        normalized_inputs = self.pre_norm(flattened_inputs_with_position)

        # Multi-head attention
        attention_output = self.attention(
            query=normalized_inputs,
            value=normalized_inputs,
            key=normalized_inputs
        )  # (batch_size, 256, 128)

        # Post-attention normalization
        attention_output = self.post_norm(attention_output)

        # Reshape back to original spatial dimensions
        reshaped_output = tf.reshape(attention_output, [batch_size, height, width, channels])

        # Modulate output by the original inputs
        output = reshaped_output + inputs

        return output

    def get_config(self):
        config = super(CustomAttentionLayerWithPositionalEncoding, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return config
    

class ImprovedFusionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ImprovedFusionLayer, self).__init__(**kwargs)
        self.filters = None 
    def build(self, input_shape):
        self.filters = input_shape[0][-1]

        # Convolutional blocks with dilation and separable convolutions
        self.sep_conv_3x3 = tf.keras.layers.SeparableConv2D(self.filters, (3, 3), dilation_rate=2, padding='same')
        self.sep_conv_3x3_no_dilation = tf.keras.layers.SeparableConv2D(self.filters, (3, 3), padding='same')

        # 1x1 convolution for dimensionality reduction
        self.conv1x1 = tf.keras.layers.Conv2D(self.filters, (1, 1), padding='same')

        # Normalization layers
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Global pooling
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

        # Gated fusion mechanism
        self.gate = tf.keras.layers.Dense(self.filters, activation='sigmoid')

    def call(self, inputs):
        encoder_input, decoder_input = inputs

        # Concatenate inputs
        concatenated = tf.concat([encoder_input, decoder_input], axis=-1)

        # Convolutions
        conv_3x3 = self.sep_conv_3x3(concatenated)
        conv_3x3_no_dilation = self.sep_conv_3x3_no_dilation(concatenated)

        # Summed feature maps with normalization
        summed_features = self.norm_1(conv_3x3 + conv_3x3_no_dilation)

        # Additional 1x1 convolution for feature refinement
        refined_features = self.conv1x1(summed_features)

        # Global attention
        global_avg = self.global_avg_pool(refined_features)  # Shape: (batch, 1, 1, filters)
        global_attention = tf.keras.activations.sigmoid(global_avg)  # Shape: (batch, 1, 1, filters)

        # Spatial attention
        spatial_attention = tf.keras.activations.sigmoid(refined_features)  # Shape: (batch, h, w, filters)

        # Weighted contributions of encoder and decoder inputs
        gated_decoder = self.gate(decoder_input) * decoder_input
        gated_encoder = (1 - self.gate(encoder_input)) * encoder_input

        # Apply attention and combine
        attention_applied = spatial_attention * gated_decoder + global_attention * gated_encoder

        # Residual connection
        output = self.norm_2(attention_applied + decoder_input)

        return output

    def get_config(self):
        config = super(ImprovedFusionLayer, self).get_config()
        config.update({
            "filters": self.filters,
        })
        return config
    @classmethod
    def from_config(cls, config):
        """
        Creates a layer instance from its config.

        This method is called by Keras when loading a model that contains this layer.
        It needs to be able to reconstruct the layer based on the configuration
        dictionary that was saved in the model.

        Args:
            config (dict): The configuration dictionary for the layer.

        Returns:
            ImprovedFusionLayer: A new instance of the layer.
        """
        # This handles the special case of the 'filters' argument
        filters = config.pop('filters', None) 
        layer = cls(**config)
        layer.filters = filters  # Set the filters attribute after initialization
        return layer

def attention_decoder(input_tensor, skip_connection, num_filters):
    """
    U-Net decoder block with UpSampling, Conv2D, and concatenation with skip connection.

    Args:
        input_tensor: Input tensor from the previous layer (decoder input).
        skip_connection: Tensor from the corresponding encoder block (skip connection).
        num_filters: Number of filters for convolutional layers.

    Returns:
        Output tensor of the decoder block.
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same", kernel_initializer="he_normal")(input_tensor)

    x = ImprovedFusionLayer()([x, skip_connection])

    x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def build_unet(input_shape):
    """
    Builds a U-Net model using encoder and decoder blocks.

    Args:
        input_shape: Tuple specifying the input image shape (height, width, channels).
        num_classes: Number of output classes for segmentation.

    Returns:
        A Keras Model representing the U-Net.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    e1, skip1 = unet_encoder_block(inputs, 32) #s: 256, e:128
    e2, skip2 = unet_encoder_block(e1, 64)#s:128, e:64
    e3, skip3 = unet_encoder_block(e2, 128)#s:64, e:32

    # Bottleneck
    bottleneck = CustomAttentionLayerWithPositionalEncoding()(e3)

    # Decoder
    d3 = attention_decoder(bottleneck, skip3, 128)
    d2 = attention_decoder(d3, skip2, 64)
    d1 = attention_decoder(d2, skip1, 32)

    # Output Layer
    outputs = layers.Conv2D(3, (1, 1), activation="sigmoid")(d1)

    return tf.keras.Model(inputs, outputs)
