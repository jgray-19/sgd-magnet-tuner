import numpy as np

from aba_optimiser.optimisers.adam import AdamOptimiser
from aba_optimiser.optimisers.amsgrad import AMSGradOptimiser


class TestAMSGradOptimiser:
    def test_initialization(self):
        shape = (5,)
        optim = AMSGradOptimiser(
            shape=shape, beta1=0.9, beta2=0.999, eps=1e-12, weight_decay=0.01
        )
        assert optim.beta1 == 0.9
        assert optim.beta2 == 0.999
        assert optim.eps == 1e-12
        assert optim.weight_decay == 0.01
        assert optim.t == 0
        assert np.array_equal(optim.m, np.zeros(shape))
        assert np.array_equal(optim.v, np.zeros(shape))
        assert np.array_equal(optim.v_hat_max, np.zeros(shape))

    def test_v_hat_max_never_decreases(self):
        """Test the key AMSGrad property: v_hat_max is the running maximum."""
        shape = (1,)
        optim = AMSGradOptimiser(shape=shape, beta1=0.9, beta2=0.9, eps=0.0)
        params = np.array([1.0])
        lr = 0.1

        # First step with large gradient
        grads1 = np.array([2.0])
        optim.step(params, grads1, lr)
        v_hat_max_1 = optim.v_hat_max.copy()

        # Second step with smaller gradient
        grads2 = np.array([0.5])
        optim.step(params, grads2, lr)
        v_hat_max_2 = optim.v_hat_max.copy()

        # v_hat_max should not decrease
        assert np.all(v_hat_max_2 >= v_hat_max_1)

        # Third step with even smaller gradient
        grads3 = np.array([0.1])
        optim.step(params, grads3, lr)
        v_hat_max_3 = optim.v_hat_max.copy()

        assert np.all(v_hat_max_3 >= v_hat_max_2)

    def test_amsgrad_vs_adam_smaller_updates_after_large_grad(self):
        """After a large gradient, AMSGrad's running max v_hat makes later small-gradient updates <= Adam's."""
        shape = (1,)
        amsgrad = AMSGradOptimiser(shape=shape, beta1=0.9, beta2=0.9, eps=0.0)
        adam = AdamOptimiser(shape=shape, beta1=0.9, beta2=0.9, eps=0.0)
        params_ams = np.array([1.0])
        params_adam = np.array([1.0])
        lr = 0.1

        # Step 1: large gradient
        g1 = np.array([2.0])
        params_ams = amsgrad.step(params_ams, g1, lr)
        params_adam = adam.step(params_adam, g1, lr)

        # Step 2: small gradient — AMSGrad denominator uses max(v_hat), so update should be <= Adam's
        g2 = np.array([0.1])
        params_ams_prev2 = params_ams.copy()
        params_adam_prev2 = params_adam.copy()
        params_ams = amsgrad.step(params_ams, g2, lr)
        params_adam = adam.step(params_adam, g2, lr)

        upd_ams_2 = np.abs(params_ams_prev2 - params_ams)[0]
        upd_adam_2 = np.abs(params_adam_prev2 - params_adam)[0]
        assert upd_ams_2 <= upd_adam_2 + 1e-12

        # Step 3: another small gradient — same reasoning
        params_ams_prev3 = params_ams.copy()
        params_adam_prev3 = params_adam.copy()
        params_ams = amsgrad.step(params_ams, g2, lr)
        params_adam = adam.step(params_adam, g2, lr)

        upd_ams_3 = np.abs(params_ams_prev3 - params_ams)[0]
        upd_adam_3 = np.abs(params_adam_prev3 - params_adam)[0]
        assert upd_ams_3 <= upd_adam_3 + 1e-12

    def test_weight_decay(self):
        shape = (1,)
        optim = AMSGradOptimiser(shape=shape, weight_decay=0.1)
        params = np.array([1.0])
        grads = np.array([2.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # Just check it runs without error and updates
        assert new_params.shape == params.shape
        assert not np.array_equal(new_params, params)

    def test_zero_gradients(self):
        shape = (2,)
        optim = AMSGradOptimiser(shape=shape)
        params = np.array([1.0, 2.0])
        grads = np.array([0.0, 0.0])
        lr = 0.01

        new_params = optim.step(params, grads, lr)

        # With zero gradients, little change
        assert np.allclose(new_params, params, atol=1e-6)
