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

A suite of optimisers for an Multilayer Perceptron using ReLU layer activations, Softmax output activation, and Categorical Cross Entropy Loss. This repository was built for my Undergraduate Research Project at Imperial College London titled "".

## Optimisers
1. Batch Gradient Descent
2. Mini-batch Gradient Descent
3. Momentum
4. Nesterov Accelerated Gradient
5. Adaptive Gradient
6. Root Mean Squared Propagation
7. Adaptive Moments
8. Noisy Mini-batch Gradient Descent 
9. Weight Decay
10. Weight Regularisation ($L_{2}$ and $L_{1}$)

