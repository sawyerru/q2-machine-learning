const NaiveBayesClassifier = require('../naive-bayes/naive-bayes');
const DataManipulator = require('../data-manipulator/data-manipulator');


const test = async () => {
  console.log('Naive Bayes Tests');
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/heartDisease.csv';
    const manipulator = new DataManipulator();
    await manipulator.loadCsv(url);
    const dconfig = {
      labelHeading: 'illness',
      labelType: 'categorical',
      excludeColumns: [],
      categoricalCols: {
        city: {dallas: 0, nyc: 1},
        gender: {female: 0, male: 1},
        illness: {no: 0, yes: 1}
      }
    };
    manipulator.processData(dconfig);
    const [features, labels] = manipulator.exportData();

    const naiveBayes = new NaiveBayesClassifier();
    const cconfig = {
      numericalCols: ['income'],
      categoricalCols: ['city', 'gender']
    };
    naiveBayes.train(features, labels, manipulator.getColumnHeaders(), cconfig);
    const x = [0, 0, 100000]; // ['dallas', 'female', 100000]
    const pred = naiveBayes.predict(x);
    console.log('All Tests Passed');
}

test();