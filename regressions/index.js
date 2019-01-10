require('@tensorflow/tfjs-node');
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv")
const CSVFile = './cars.csv'


let {features, labels, testFeatures, testLabels  } = loadCSV(CSVFile,{
    shuffle: true,
    splitTest: 50,
    dataColumns:['horsepower'],
    labelColumns:['mpg']   
})

console.log(features)