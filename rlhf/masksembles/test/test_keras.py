import unittest

import masksembles.keras
import tensorflow as tf


class TestCreation(unittest.TestCase):
    def test_init_failed(self):
        layer = masksembles.keras.Masksembles2D(4, 11.0)
        self.assertRaises(ValueError, layer, tf.ones([4, 10, 4, 4]))

    def test_init_success(self):
        layer = masksembles.keras.Masksembles2D(4, 11.0)
