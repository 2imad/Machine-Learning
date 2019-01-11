require('@tensorflow/tfjs-node');
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv")
const CSVFile = './cars.csv'
const LinearRegression = require("./linear-regression")


let {features, labels, testFeatures, testLabels  } = loadCSV(CSVFile,{
    shuffle: true,
    splitTest: 50,
    dataColumns:['horsepower'],
    labelColumns:['mpg']   
})

const regression = new LinearRegression(features, labels,{
    iterations : 100,
    learningRate : 0.001
    }
  )


console.log("rate for b is :" + regression.b, "and rate for m is :" + regression.m)
