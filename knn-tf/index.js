require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const dataFile = 'kc_house_data.csv'

function knn(features, labels , predctionPoint , k){
  const { mean, variance } = tf.moments(features, 0);
 // mean returns the average
  const scaledPrediction = predctionPoint.sub(mean).div(variance.pow(.5))
   return features
    .sub(mean)
    .div(variance.pow(.5))
    .sub(scaledPrediction)
    .pow(2) // to the power of 2 
    .sum(1) // sum along the X axis 
    .pow(0.5) // Square root 
    .expandDims(1) // expand dimenstions along the X axis
    .concat(labels,1) // concat along the X axis 
    .unstack() // unfold the tensors to vanilla arrays 
    .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1)
    .slice(0,k)
    .reduce((acc, pair) => acc + pair.get(1), 0) / k
}
let { features, labels, testFeatures, testLabels } = loadCSV(dataFile,{
    shuffle : true,
    splitTest: 10,
    dataColumns : ['lat','long','sqft_lot','sqft_living'],
    labelColumns: ['price']
})
features = tf.tensor(features)
labels = tf.tensor(labels)

testFeatures.forEach((testPoint, i) =>{
    const result = knn(features, labels, tf.tensor(testPoint), 10)
    const err =( testLabels[i][0] - result ) / testLabels[i][0]
    console.log("Analysis off by",err*100,"%")
})