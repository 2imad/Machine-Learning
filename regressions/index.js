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

const regression = new LinearRegression(features, labels, {
    learningRate : 0.0001,
    iterations : 100
  })
  regression.train() 
  const R2 =  regression.test(testFeatures, testLabels)
  console.log("R2 is " , R2)

