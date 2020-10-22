const NeuralNetwork = require('../neural-network/neural-net');
const DataManipulator = require('../data-manipulator/data-manipulator');


const test_num = async () => {
  console.log('Neural Network Tests (Numerical Classes)');
    // const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-1class.csv';
    // const manipulator = new DataManipulator();
    // await manipulator.loadCsv(url);
    // this.config.labelType = 'numeric';
    // manipulator.processData(this.config);
    // manipulator.normalize();
    // const [features, labels] = manipulator.exportData();

    // const neuralNet = new NeuralNetwork(url, this.config);
    // const config = {
    //   inputDims: 43,
    //   architecture: [60, 20],
    //   outputClasses: 1,
    //   activationF: 'relu',
    //   optimizerF: 'adam',
    //   lossF: 'meanSquaredError'
    // };

    // neuralNet.build(config);
    // await neuralNet.train(features, labels, document.getElementById('lossCanvas'));

    // let errorSum = 0;
    // for (let i = 0; i < features.shape[0]; i++ ) {
    //   const sample = features.slice([i, 0], [1, 43]);
    //   const prediction = neuralNet.predict(sample);

    //   this.jsonData[i].nn_num = prediction;
    //   errorSum += Math.pow(this.jsonData[i].G3 - prediction, 2);
    // }
    // this.nn_num_acc = Math.pow(errorSum / features.shape[0], 0.5);
    // neuralNet.clean();
}

const test_5class = async () => {
  console.log('Neural Network Tests (5 Classes)');

    // const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-5_class.csv';
    // const manipulator = new DataManipulator();
    // await manipulator.loadCsv(url);
    
    // manipulator.processData(this.config);
    // manipulator.normalize();
    // const [features, labels] = manipulator.exportData();

    // const neuralNet = new NeuralNetwork();
    // const config = {
    //   inputDims: 43,
    //   architecture: [60, 20],
    //   outputClasses: 5,
    //   activationF: 'relu',
    //   optimizerF: 'adam',
    //   lossF: 'categoricalCrossentropy'
    // };

    // neuralNet.build(config);
    // await neuralNet.train(features, labels, document.getElementById('lossCanvas'));

    // let correctClassCount = 0;
    // for (let i = 0; i < features.shape[0]; i++ ) {
    //   const sample = features.slice([i, 0], [1, 43]);
    //   const prediction = neuralNet.predict(sample);

    //   this.jsonData[i].class5 = Number(manipulator.jsonData[i].G3);
    //   this.jsonData[i].nn_5class = Number(prediction);

    //   if (this.jsonData[i].class5 === this.jsonData[i].nn_5class) {
    //     correctClassCount += 1;
    //   }
    // }
    // this.nn_5class_acc = correctClassCount / features.shape[0] * 100;
    // neuralNet.clean();
}

const test_10class = async () => {
  console.log('Neural Network Tests (10 Classes)');

    // const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-10class.csv';
    // const manipulator = new DataManipulator();
    // await manipulator.loadCsv(url);
    // this.config.labelType = 'categorical';
    // this.config.categoricalCols['G3'] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9};
    // manipulator.processData(this.config);
    // manipulator.normalize();
    // const [features, labels] = manipulator.exportData();

    // const neuralNet = new NeuralNetwork();
    // const config = {
    //   inputDims: 43,
    //   architecture: [60, 20],
    //   outputClasses: 10,
    //   activationF: 'relu',
    //   optimizerF: 'adam',
    //   lossF: 'categoricalCrossentropy'
    // };

    // neuralNet.build(config);
    // await neuralNet.train(features, labels, document.getElementById('lossCanvas'));

    // let correctClassCount = 0;
    // for (let i = 0; i < features.shape[0]; i++ ) {
    //   const sample = features.slice([i, 0], [1, 43]);
    //   const prediction = neuralNet.predict(sample);

    //   this.jsonData[i].class10 = Number(manipulator.jsonData[i].G3);
    //   this.jsonData[i].nn_10class = Number(prediction);

    //   if (this.jsonData[i].class10 === this.jsonData[i].nn_10class) {
    //     correctClassCount += 1;
    //   }
    // }
    // this.nn_10class_acc = correctClassCount / features.shape[0] * 100;
    // neuralNet.clean();
}


test_num();
test_5class();
test_10class();