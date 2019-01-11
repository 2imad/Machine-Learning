const features = tf.tensor([
    [10],
    [20],
    [35],
    [90]
])

// mean and variance are tensor object properties
const { mean , variance } = tf.moments(features, 0)
features.sub(mean).div(variance.pow(0.5)) 

