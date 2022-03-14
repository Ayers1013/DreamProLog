from .latent import *
import tensorflow as tf

def test_latent_NormalSpace(logger):
    latent_layer = NormalSpace()
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()
    mode_2 = x.extract(False)

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = NormalSpace()
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)
    #mode and mode_2 should be equal
    assert tf.reduce_sum(mode-mode_2) <0.01

    return 'misc.latent.NormalSpace checks out.'

def test_latent_ScaledNormalSpace(logger):
    latent_layer = ScaledNormalSpace(6)
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = ScaledNormalSpace(6)
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)

    logprob = x.base_dist.log_prob(sample)
    logger('\n\n')
    logger(logprob)

    assert logprob.shape ==(4,)

    return 'misc.latent.ScaledNormalSpace checks out.'

def test_latent_DiscrateSpace(logger):
    latent_layer = DiscrateSpace(8, 6)
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = DiscrateSpace(8, 6)
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)

    return 'misc.latent.DiscrateSpace checks out.'

def test_latent_GumbleSpace(logger):
    temp = 0.5
    latent_layer = GumbleSpace(8, 6, temp)
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = GumbleSpace(8, 6, temp)
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)

    return 'misc.latent.GumbleSpace checks out.'

from .autoconfig import *
def test_autoconfig_0(logger):
    class A(ConfiguredModule):
        def __init__(self, *args, **kwargs):
            ConfiguredModule.__init__(self, *args, **kwargs)
            self.z = self.x + self.y

        def __call__(self):
            return f'x:{self.x}, y:{self.y}, z:{self.z}.'
        
    class B(ConfiguredModule):
        def __init__(self, *args, **kwargs):
            ConfiguredModule.__init__(self, *args, **kwargs)
            self.a = self.configure(A)

        def __call__(self):
            return f'B.z:{self.z}, A.{self.a()}'

    b = B(x=4, y=5, z=420)
    assert b() == 'B.z:420, A.x:4, y:5, z:9.'
    #b._stats()
    params = {'A': {'x': 69}, 'x': 707, 'z': 42}
    b2 = B(y=5, z=420, params=params)
    #b2._stats()
    logger(b2())
    assert b2() == 'B.z:420, A.x:69, y:5, z:74.'

    return 'misc.autoconfig test 0 passed.'

def test_autoconfig_1(logger):
    class NN(ConfiguredModule, tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.layers = [self.configure(tf.keras.layers.Dense, self.dims, activation='relu') for i in range(self.num_layers)]
            self.end_layer = self.configure(tf.keras.layers.Dense, self.dims, activation='sigmoid')

        def call(self, x):
            for l in self.layers:
                x = l(x)
            x = self.end_layer(x)
            return x

    nn = NN(num_layers=2, dims=8)
    inp = tf.zeros((8,8))
    x = nn(inp)
    assert x.shape == (8,8)
    return 'misc.autoconfig test 1 passed.'

def test_autoconfig_2(logger):
    class NN(ConfiguredModule, tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.layers = [self.configure(tf.keras.layers.Dense, self.dims, activation='relu') for i in range(self.num_layers)]
            self.end_layer = self.configure(tf.keras.layers.Dense, self.dims, activation='sigmoid')

        def call(self, x):
            for l in self.layers:
                x = l(x)
            x = self.end_layer(x)
            return x

    class NN2(NN):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.end_layer2 = self.configure(tf.keras.layers.Dense, 1, activation = self.act)
            self.nn = self.configure(NN)

        def call(self, x):
            x = super().call(x)
            x = self.end_layer2(x)

    
    nn = NN2(num_layers=2, dims=8, act='')
    inp = tf.zeros((8,8))
    x = nn(inp)
    logger(nn._name_structure(full_depth = False))
    return 'misc.autoconfig test 2 passed.'

def test_autoconfig_3(logger):
    'Test whether param inheritenc correctly works.'
    # NOTE autoconfig does not support param inheritence, therefore this test is disabled
    class NN(ConfiguredModule, tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.layers = [self.configure(tf.keras.layers.Dense, self.dims, activation='relu') for i in range(self.num_layers)]
            self.end_layer = self.configure(tf.keras.layers.Dense, self.dims, activation='sigmoid')
            self.string = f'{self.x}'

        def call(self, x):
            for l in self.layers:
                x = l(x)
            x = self.end_layer(x)
            return x

    class NN2(NN):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.end_layer2 = self.configure(tf.keras.layers.Dense, 1, activation = self.act)
            self.nn1 = self.configure(NN)
            self.nn2 = self.configure(NN, unique_name = 'dr_strange')
            self.nn3 = self.configure(NN, unique_name = 'spiderman')

        def call(self, x):
            x = super().call(x)
            x = self.end_layer2(x)

    params = {
        'dims': 8,
        'act': 'relu',
        'num_layers': 3,
        'x': 42,
        'NN': {
            'x': 3
        },
        'dr_strange': {
        },
        'spiderman': {
            'x': 7
        },
    }

    nn = NN2(params=params)
    logger(nn._name_structure())

    assert nn.nn1.x == 3
    # NOTE nn.nn2.x should be 3 when param inheritence is supported
    assert nn.nn2.x == 42
    assert nn.nn3.x == 7
    return 'misc.autoconfig test 3 passed.'