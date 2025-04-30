import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import math
import numpy as np
from PIL import Image

# Add constants
LAYER_NORM_EPS = 1e-6

def bottleneck0(inputs):
    # EfficientNetV2 
    backbone = tf.keras.applications.EfficientNetV2B0(
        weights='imagenet', 
        include_top=False,
        input_tensor=inputs  # directly specify input tensor
    )
    
    # Feature extraction and processing
    x = backbone.output  # use backbone output
    x = layers.GlobalAveragePooling2D(
        name='efficient_gap'
    )(x)
    x = layers.BatchNormalization(
        name='efficient_bn'
    )(x)
    x = layers.Dense(
        256, 
        activation='relu',
        name='efficient_dense'
    )(x)
    x = layers.Flatten(
        name='efficient_flatten'
    )(x)
    
    return x

class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        num_patches,
        projection_dim,
        vanilla=False,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.vanilla = vanilla
        
        self.patch_norm = layers.BatchNormalization(
            name=f"{name}_patch_norm" if name else "patch_norm"
        )
        
        self.patch_proj_1 = layers.Dense(
            projection_dim, 
            activation='gelu',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name}_patch_proj_1" if name else "patch_proj_1"
        )
        
        self.patch_proj_2 = layers.Dense(
            projection_dim,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{name}_patch_proj_2" if name else "patch_proj_2"
        )
        
        self.patch_dropout = layers.Dropout(
            0.1,
            name=f"{name}_patch_dropout" if name else "patch_dropout"
        )
        
        self.patch_layer_norm = layers.LayerNormalization(
            epsilon=LAYER_NORM_EPS,
            name=f"{name}_patch_layer_norm" if name else "patch_layer_norm"
        )

    def extract_patches(self, images):
        noise = tf.random.normal(
            shape=tf.shape(images), 
            mean=0.0, 
            stddev=0.01,
            name="patch_noise"
        )
        images = images + noise
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [tf.shape(images)[0], -1, self.patch_size * self.patch_size * 3])
        return patches

    def call(self, images, training=None):
        if self.vanilla:
            patches = self.extract_patches(images)
            patches = self.patch_norm(patches, training=training)
            patches = self.patch_proj_1(patches)
            patches = self.patch_dropout(patches, training=training)
            patches = self.patch_layer_norm(patches)
        else:
            patches = self.extract_patches(images)
            patches = self.patch_norm(patches, training=training)
            patches = self.patch_proj_1(patches)
            
            shifted_images = tf.roll(
                images, 
                shift=[self.half_patch, self.half_patch], 
                axis=[1, 2],
                name="shifted_images"
            )
            shifted_patches = self.extract_patches(shifted_images)
            shifted_patches = self.patch_norm(shifted_patches, training=training)
            shifted_patches = self.patch_proj_1(shifted_patches)
            
            attention_weights = tf.nn.softmax(
                tf.matmul(patches, shifted_patches, transpose_b=True),
                name="attention_weights"
            )
            attention_output = tf.matmul(attention_weights, shifted_patches)
            
            patches = tf.concat(
                [patches, attention_output], 
                axis=1,
                name="concat_patches"
            )
            
            patches = tf.cond(
                tf.greater(tf.shape(patches)[1], self.num_patches),
                lambda: patches[:, :self.num_patches, :],
                lambda: patches
            )
            
            patches = self.patch_dropout(patches, training=training)
            patches = self.patch_proj_2(patches)
            patches = self.patch_layer_norm(patches)
        
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "vanilla": self.vanilla,
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(
        self,
        num_patches,
        projection_dim,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        
        self.pos_embedding = layers.Embedding(
            input_dim=num_patches, 
            output_dim=projection_dim,
            name=f"{name}_pos_embedding" if name else "pos_embedding"
        )

    def call(self, patch_input):
        position_indices = tf.range(
            start=0, 
            limit=self.num_patches, 
            delta=1,
            name="position_indices"
        )
        
        position_encodings = self.pos_embedding(position_indices)
        encoded_patches = patch_input + position_encodings
        return encoded_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config

# Multi-head attention layer with LSA scaling
class MultiHeadAttentionLSA(layers.MultiHeadAttention):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lsa_temperature = tf.Variable(
            math.sqrt(float(self._key_dim)), 
            trainable=True,
            name=f"{name}_lsa_temperature" if name else "lsa_temperature"
        )
    
    
    def _compute_attention(
        self, 
        query, 
        key, 
        value, 
        attention_mask=None, 
        training=None,
        use_causal_mask=False
    ):
        
        
        # use LSA scaling
        query = query * (1.0 / self.lsa_temperature)
        
        # calculate attention scores
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # scale attention scores
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        
        # apply attention mask(if any)
        if attention_mask is not None:
            logits += attention_mask
            
        # apply causal mask(if any)
        if use_causal_mask:
            size = tf.shape(logits)[-2]
            causal_mask = 1.0 - tf.linalg.band_part(
                tf.ones((size, size)), -1, 0
            )
            causal_mask = tf.cast(causal_mask, logits.dtype)
            logits -= 1.e9 * causal_mask
            
        # attention weights
        attention_weights = tf.nn.softmax(logits, axis=-1)
        
        # use dropout(if training)
        if training:
            attention_weights = self._dropout_layer(attention_weights)
            
        # calculate output
        attention_output = tf.matmul(attention_weights, value)
        
        return attention_output, attention_weights

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "lsa_temperature": self.lsa_temperature.numpy()
        })
        return base_config

def mlp(x, hidden_units, dropout_rate=0.1, name_prefix="mlp"):
   # MLP block
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units, 
            activation=tf.nn.gelu,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f'{name_prefix}_dense_{i}'
        )(x)
        x = layers.Dropout(
            dropout_rate,
            name=f'{name_prefix}_dropout_{i}'
        )(x)
    return x

def preprocess_image(image_path):
    
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image = image.resize((160, 160))
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def create_model():
    
    # basic configuration parameters
    input_shape = (160, 160, 3)
    patch_size = 16
    num_patches = (160 // patch_size) ** 2
    
    # match original model
    projection_dim = 64   
    num_heads = 4        
    num_classes = 2
    transformer_layers = 6
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    weight_decay = 0.001  
    
    inputs = layers.Input(shape=input_shape)
    
    # EfficientNetV2 分支
    efficient_branch = bottleneck0(inputs)  
    
    # ViT 分支
    tokens = ShiftedPatchTokenization(
        image_size=160,
        patch_size=patch_size,
        num_patches=num_patches * 2,  
        projection_dim=projection_dim,
        vanilla=False,
        name='spt_layer'
    )(inputs)
    
    
    encoded_patches = PatchEncoder(
        num_patches=num_patches * 2,  
        projection_dim=projection_dim,
        name='patch_encoder'
    )(tokens)

    # Transformer blocks
    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(
            epsilon=LAYER_NORM_EPS,
            name=f'transformer_norm1_{i}'
        )(encoded_patches)
        
        
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=projection_dim, 
            dropout=0.1,
            name=f'mha_lsa_{i}'
        )(x1, x1)
        
        x2 = layers.Add(name=f'skip_connection1_{i}')([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(
            epsilon=LAYER_NORM_EPS,
            name=f'transformer_norm2_{i}'
        )(x2)
        
        
        x3 = mlp(
            x3, 
            hidden_units=transformer_units, 
            dropout_rate=0.1,
            name_prefix=f'mlp_{i}'
        )
        
        encoded_patches = layers.Add(name=f'skip_connection2_{i}')([x3, x2])

    
    vit_features = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS,
        name='final_layer_norm'
    )(encoded_patches)
    vit_features = layers.Flatten(name='flatten')(vit_features)  
    vit_features = layers.Dropout(0.2, name='vit_dropout')(vit_features)
    
    
    combined_features = layers.Concatenate(name='feature_concat')([efficient_branch, vit_features])
    
    
    kernel_regularizer = regularizers.l2(weight_decay * 0.5)
    
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=regularizers.l1(1e-6),
        name='dense_1'
    )(combined_features)
    x = layers.BatchNormalization(momentum=0.9, name='bn_1')(x)
    x = layers.Dropout(0.2, name='dropout_1')(x)
    
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=regularizers.l1(1e-6),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(momentum=0.9, name='bn_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        kernel_regularizer=kernel_regularizer,
        name='output'
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def main():
    
    weights_path = "/Users/carson/Desktop/code/GUI/vit_hybrid_spt_20250329_154643/best_model.h5"
    
    try:
        print("Building model structure...")
        model = create_model()
        
        print("Model summary:")
        model.summary()  # print model structure
        
        print("\nLoading model weights...")
        try:
            # try normal loading
            model.load_weights(weights_path)
            print("Model weights loaded successfully (normal mode).")
        except:
            # if failed, try loading by name
            print("Failed to load normal weights, trying to load by name...")
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("Loaded model weights successfully (by name).")
        
        # 测试图像预测 - 请修改为您的实际图像路径
        test_image_path = "/Users/carson/Desktop/Pitcture1.png"
        print(f"\nProcessing test image: {test_image_path}")
        
        # 预处理图像
        processed_image = preprocess_image(test_image_path)
        if processed_image is None:
            raise Exception("Image processing failed")
            
        print("Processed image shape:", processed_image.shape)
        
        # 进行预测
        print("\nPredicting...")
        prediction = model.predict(processed_image)
        
        # 显示预测结果
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx] * 100
        result_text = "(Benign)" if class_idx == 0 else "(Malignant)"
        
        print("\nResult:")
        print(f"Result: {result_text}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Prediction: {prediction[0]}")
        
    except Exception as e:
        print(f"error: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息

if __name__ == "__main__":
    main()