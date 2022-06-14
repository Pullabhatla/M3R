<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# M3R

A suite of optimisers for an Multilayer Perceptron using Leaky ReLU layer activations, Softmax output activation, and Categorical Cross Entropy Loss. This repository was built for my Undergraduate Research Project at Imperial College London titled "A Computational Review of the Performance of a Suite of Optimisers for Multilayer Perceptron Supervised Learning".

## Reported Optimisers
1. Gradient Descent (Section 2.1): [`gd_train`](../Main/Req/mlp.py)
2. Stochastic Gradient Descent (Section 2.2): [`sgd_train`](../Main/Req/mlp.py)
3. Momentum (Section 2.3): [`momentum_sgd_train`](../Main/Req/mlp.py)
4. Nesterov Accelerated Gradient (Section 2.4): [`nesterov_sgd_train`](../Main/Req/mlp.py)
5. Adaptive Gradient (Section 2.5): [`adagrad_train`](../Main/Req/mlp.py)
6. Root Mean Squared Propagation (Section 2.6): [`rmsprop_train`](../Main/Req/mlp.py)
7. Adaptive Moments (Section 2.7): [`adam_train`](../Main/Req/mlp.py)
8. Noisy Stochastic Gradient Descent (Section 2.8): [`noisy_sgd_train`](../Main/Req/mlp.py)
10. Elastic Net Weight Regularisation (Section 2.9): [`elastic_net_train`](../Main/Req/mlp.py)
11. Carrillo Consensus Based Optimiser (Section 3): : [`carrillo_cbo`](../Main/Req/carrillo.py)

## Unreported Optimisers
1. Weight Decay: [`decay_train`](../Main/Req/mlp.py)
2. $L_{1}$ Weight Regularistion: [`l1_regularise_train`](../Main/Req/mlp.py)
3. $L_{2}$ Weight Regularistion: [`l2_regularise_train`](../Main/Req/mlp.py)
4. Simple Majority Voting: [`simple_majority_voting`](../Main/Req/ensemble.py)
5. Decorrelated Majority Voting: [`decorrelated_majority_voting`](../Main/Req/ensemble.py)
6. Weighted Voting: [`weighted_voting`](../Main/Req/ensemble.py)
