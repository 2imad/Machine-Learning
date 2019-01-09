require('@tensorflow/tfjs')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const dataFile = 'kc_house_data.csv'

let { features, labels, testFeatures, testLabels } = loadCSV(dataFile,{
    shuffle : true,
    splitTest: 10,
    dataColumns : ['lat','long'],
    labelColumns: ['price']
})