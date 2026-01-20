"""Tests for TensorFlow/Keras support."""

import pytest


@pytest.fixture(autouse=True)
def check_tensorflow():
    """Skip all tests in this module if TensorFlow is not available."""
    pytest.importorskip("tensorflow")


class TestCameraAdaptorLayer:
    """Tests for TensorFlow CameraAdaptor layer."""

    def test_adaptor_required(self):
        """Test adaptor parameter is required."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        with pytest.raises(TypeError):
            CameraAdaptor()

    def test_explicit_adaptor(self):
        """Test explicit adaptor configuration."""
        from edgefirst.cameraadaptor import ColorSpace
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("yuyv")
        assert layer.adaptor == "yuyv"
        assert layer.color_space == ColorSpace.YUYV

    def test_invalid_adaptor(self):
        """Test invalid adaptor raises ValueError."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        with pytest.raises(ValueError, match="Invalid camera adaptor"):
            CameraAdaptor("invalid")

    def test_unsupported_adaptor(self):
        """Test unsupported adaptor raises ValueError."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        with pytest.raises(ValueError, match="not yet supported"):
            CameraAdaptor("nv12")

    def test_forward_rgb_passthrough(self):
        """Test forward pass with RGB is passthrough."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgb")
        x = tf.random.normal((1, 224, 224, 3))
        y = layer(x)
        tf.debugging.assert_equal(x, y)

    def test_forward_rgba_drops_alpha(self):
        """Test forward pass with RGBA drops alpha channel."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgba")
        x = tf.random.normal((1, 224, 224, 4))
        y = layer(x)
        assert y.shape == (1, 224, 224, 3)
        tf.debugging.assert_equal(x[:, :, :, :3], y)

    def test_forward_bgra_drops_alpha(self):
        """Test forward pass with BGRA drops alpha channel."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("bgra")
        x = tf.random.normal((1, 224, 224, 4))
        y = layer(x)
        assert y.shape == (1, 224, 224, 3)

    def test_compute_output_shape_rgba(self):
        """Test compute_output_shape for RGBA."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgba")
        input_shape = tf.TensorShape([None, 224, 224, 4])
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape.as_list() == [None, 224, 224, 3]

    def test_compute_output_shape_passthrough(self):
        """Test compute_output_shape for passthrough formats."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgb")
        input_shape = tf.TensorShape([None, 224, 224, 3])
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape.as_list() == [None, 224, 224, 3]


class TestCameraAdaptorSerialization:
    """Tests for layer serialization."""

    def test_get_config(self):
        """Test get_config method."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("yuyv")
        config = layer.get_config()
        assert config["adaptor"] == "yuyv"

    def test_from_config(self):
        """Test from_config class method."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgba")
        config = layer.get_config()
        restored = CameraAdaptor.from_config(config)
        assert restored.adaptor == "rgba"

    def test_model_save_load(self, tmp_path):
        """Test saving and loading a model with CameraAdaptor."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        # Create model
        inputs = tf.keras.Input(shape=(224, 224, 4))
        x = CameraAdaptor("rgba")(inputs)
        x = tf.keras.layers.Conv2D(16, 3, padding="same")(x)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
        model = tf.keras.Model(inputs, outputs)

        # Save and load
        model_path = tmp_path / "test_model.keras"
        model.save(model_path)
        loaded_model = tf.keras.models.load_model(model_path)

        # Test inference
        x = tf.random.normal((1, 224, 224, 4))
        y1 = model(x)
        y2 = loaded_model(x)
        tf.debugging.assert_near(y1, y2)


class TestCameraAdaptorChannelsFirst:
    """Tests for channels_first parameter."""

    def test_channels_first_property(self):
        """Test channels_first property."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgb", channels_first=True)
        assert layer.channels_first is True

    def test_channels_first_default(self):
        """Test channels_first defaults to False."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgb")
        assert layer.channels_first is False

    def test_forward_channels_first_rgb(self):
        """Test forward pass with channels-first RGB input."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgb", channels_first=True)
        x = tf.random.normal((1, 3, 224, 224))  # NCHW
        y = layer(x)
        assert y.shape == (1, 224, 224, 3)  # NHWC output

    def test_forward_channels_first_yuyv(self):
        """Test forward pass with channels-first YUYV input."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("yuyv", channels_first=True)
        x = tf.random.normal((1, 2, 224, 224))  # NCHW
        y = layer(x)
        assert y.shape == (1, 224, 224, 2)  # NHWC output

    def test_forward_channels_first_rgba(self):
        """Test forward pass with channels-first RGBA drops alpha."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgba", channels_first=True)
        x = tf.random.normal((1, 4, 224, 224))  # NCHW
        y = layer(x)
        assert y.shape == (1, 224, 224, 3)  # NHWC output, alpha dropped

    def test_forward_channels_first_bgra(self):
        """Test forward pass with channels-first BGRA drops alpha."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("bgra", channels_first=True)
        x = tf.random.normal((1, 4, 224, 224))  # NCHW
        y = layer(x)
        assert y.shape == (1, 224, 224, 3)  # NHWC output, alpha dropped

    def test_channels_first_permute_correctness(self):
        """Test that channels-first permute is correct."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgb", channels_first=True)

        # Create tensor with known values
        x = tf.reshape(tf.range(12, dtype=tf.float32), (1, 3, 2, 2))  # NCHW
        y = layer(x)

        # Verify shape
        assert y.shape == (1, 2, 2, 3)  # NHWC

        # Verify values are correctly permuted
        # NCHW[0,c,h,w] -> NHWC[0,h,w,c]
        for c in range(3):
            assert float(y[0, 0, 0, c]) == float(x[0, c, 0, 0])

    def test_compute_output_shape_channels_first_rgba(self):
        """Test compute_output_shape for channels-first RGBA."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgba", channels_first=True)
        input_shape = tf.TensorShape([None, 4, 224, 224])  # NCHW
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape.as_list() == [None, 224, 224, 3]  # NHWC

    def test_compute_output_shape_channels_first_passthrough(self):
        """Test compute_output_shape for channels-first passthrough."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgb", channels_first=True)
        input_shape = tf.TensorShape([None, 3, 224, 224])  # NCHW
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape.as_list() == [None, 224, 224, 3]  # NHWC

    def test_config_includes_channels_first(self):
        """Test get_config includes channels_first."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("yuyv", channels_first=True)
        config = layer.get_config()
        assert config["channels_first"] is True

    def test_from_config_with_channels_first(self):
        """Test from_config restores channels_first."""
        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgba", channels_first=True)
        config = layer.get_config()
        restored = CameraAdaptor.from_config(config)
        assert restored.channels_first is True
        assert restored.adaptor == "rgba"

    def test_batch_processing_channels_first(self):
        """Test batch processing with channels_first."""
        import tensorflow as tf

        from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

        layer = CameraAdaptor("rgba", channels_first=True)
        batch_size = 8
        x = tf.random.normal((batch_size, 4, 224, 224))  # NCHW
        y = layer(x)
        assert y.shape == (batch_size, 224, 224, 3)  # NHWC
