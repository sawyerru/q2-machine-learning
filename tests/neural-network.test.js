/*
Neural Network Tests
Author: Sawyer Ruben
 */
const NeuralNetwork = require('../neural-network/neural-net');
const DataManipulator = require('../data-manipulator/data-manipulator');

const got = require('got');

// Defining config object for data
const config = {
  labelHeading: 'G3',
  labelType: 'numeric',
  excludeColumns: [],
  categoricalCols: {
    school: {GP: 0, MS: 1},
    sex: {F: 0, M: 1},
    address: {R: 0, U: 1},
    famsize: {LE3: 0, GT3: 1},
    Pstatus: {A: 0, T: 1},
    Mjob: {teacher: 0, health: 1, services: 2, at_home: 3, other: 4},
    Fjob: {teacher: 0, health: 1, services: 2, at_home: 3, other: 4},
    reason: {home: 0, reputation: 1, course: 2, other: 3},
    guardian: {mother: 0, father: 1, other: 2},
    schoolsup: {no: 0, yes: 1},
    famsup: {no: 0, yes: 1},
    paid: {no: 0, yes: 1},
    activities: {no: 0, yes: 1},
    nursery: {no: 0, yes: 1},
    higher: {no: 0, yes: 1},
    internet: {no: 0, yes: 1},
    romantic: {no: 0, yes: 1},
  }
};

// Test Neural Net with Numeric Classes
const test_num = async () => {
  console.log('Neural Network Tests (Numerical Classes)');
  const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-1class.csv';
  const resp = await got(url);

  const manipulator = new DataManipulator();
  await manipulator.loadCsv(resp.body); // load csv into manipulator object

  manipulator.processData(config); // process data and convert into Tensors
  manipulator.normalize(); // Normalize data points
  console.assert(manipulator.X.length > 0 && manipulator.X[0].length > 0, "X not processed");
  console.assert(manipulator.Y.length > 0, "X not processed");

  const [features, labels] = manipulator.exportData(); // Export data
  console.assert(features.shape[0] === labels.shape[0], "unevent sample size");

  const sample = features.slice([0, 0], [1, 43]); // extract sample
  const feats = features.shape[1];

  const neuralNet = new NeuralNetwork(); // create neural net object

  // define neural net configuration
  const nn_config = {
    inputDims: feats,         //should be number of features
    architecture: [60, 20],   // array length denotes number of hidden layers, elements are units in layer
    outputClasses: 1,         // final output classes (output layer)
    activationF: 'relu',      // unit activation function (more options defined in Tensorflow.js specs)
    optimizerF: 'adam',       // neural net optimizer function (more options defined in Tensorflow.js specs)
    lossF: 'meanSquaredError' // loss function defined in Tensorflow.js
  };

  try { neuralNet.getModel(); }
  catch(err) { console.assert(err !== undefined, 'Error not caught'); }

  // build and compile neural network config
  neuralNet.build(nn_config);

  console.assert(neuralNet.getModel() !== undefined, 'Model still not defined');
  await neuralNet.train(features, labels); // train model with features and labels

  const label = manipulator.Y[0];

  const pred = neuralNet.predict(sample); // predict a numerical score from a sample
  console.assert(label-1 <= pred <= label+1, 'Label too far off');

  neuralNet.clean(); // clean unnecessary tensors for browser memory
}

// Neural Network testing classification
const test_category = async () => {
  console.log('Neural Network Tests (5 Classes)');

  const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-5_class.csv';
  const manipulator = new DataManipulator();
  const resp = await got(url);
  await manipulator.loadCsv(resp.body);

  // label type change in configuration object to change to classification output
  config.labelType = 'categorical';
  config.categoricalCols["G3"] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4};

  manipulator.processData(config);
  manipulator.normalize();
  console.assert(manipulator.X.length > 0 && manipulator.X[0].length > 0, "X not processed");
  console.assert(manipulator.Y.length > 0, "X not processed");

  const [features, labels] = manipulator.exportData();
  console.assert(features.shape[0] === labels.shape[0], "uneven sample size");

  const sample = features.slice([0, 0], [1, 43]);

  const neuralNet = new NeuralNetwork();
  const build_config = {
    inputDims: features.shape[1],
    architecture: [60, 20],
    outputClasses: 5,
    activationF: 'relu',
    optimizerF: 'adam',
    lossF: 'categoricalCrossentropy' // change loss function for classification (if not defined this will be changed automatically)
  };

  try { neuralNet.getModel(); }
  catch(err) { console.assert(err !== undefined, 'Error not caught'); }

  neuralNet.build(build_config);

  console.assert(neuralNet.getModel() !== undefined, 'Model still not defined');

  await neuralNet.train(features, labels);
  const label = manipulator.Y[0];

  const pred = neuralNet.predict(sample);
  console.assert(0 <= pred <= 4, 'Label too far off');

  neuralNet.clean();
}


const test = async () => {
  await test_num();
  await test_category();
}

test();
