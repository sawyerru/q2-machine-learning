const NaiveBayesClassifier = require('../naive-bayes/naive-bayes');
const DataManipulator = require('../data-manipulator/data-manipulator');

const got = require('got');

const test = async () => {
  console.log('Naive Bayes Tests');
  const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/heartDisease.csv';
  const manipulator = new DataManipulator();
  const resp = await got(url);
  await manipulator.loadCsv(resp.body);
  const config = {
    labelHeading: 'illness',
    labelType: 'categorical',
    excludeColumns: [],
    categoricalCols: {
      city: {dallas: 0, nyc: 1},
      gender: {female: 0, male: 1},
      illness: {no: 0, yes: 1}
    }
  };
  manipulator.processData(config);
  console.assert(manipulator.X.length > 0, 'no samples read');
  const [features, labels] = manipulator.exportData();
  console.assert(features.shape[0] === labels.shape[0], 'sample size mismatch');

  const naiveBayes = new NaiveBayesClassifier();
  console.assert(Object.keys(naiveBayes.headers).length === 0, 'headers not null');

  const modelConfig = {
    numericalCols: ['income'],
    categoricalCols: ['city', 'gender']
  };
  naiveBayes.train(features, labels, manipulator.getColumnHeaders(), modelConfig);
  console.assert(Object.keys(naiveBayes.headers).length !== 0, 'headers null');

  const x = [0, 0, 100000]; // ['dallas', 'female', 100000]
  const pred = naiveBayes.predict(x);
  console.assert(pred === 0 || pred === 1, 'invalid class label');

  console.log('All Tests Passed');
}

test();
