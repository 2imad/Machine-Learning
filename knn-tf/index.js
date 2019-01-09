require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const dataFile = 'kc_house_data.csv'

function knn(features, labels , predctionPoint , k){
  const { mean, variance } = tf.moments(features, 0);
 
  const scaledPrediction = predctionPoint.sub(mean).div(variance.pow(.5))
   return features
    .sub(mean)
    .div(variance.pow(.5))
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels,1)
    .unstack()
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