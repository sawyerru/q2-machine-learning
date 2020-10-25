const RandomForestClassifier = require('../decision-trees/random-forest');
const DataManipulator = require('../data-manipulator/data-manipulator');

const got = require('got');

const test = async () => {
  console.log('Random Forest Tests');
  const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-5_class.csv';
  const resp = await got(url);
  const manipulator = new DataManipulator();
  await manipulator.loadCsv(resp.body);
  console.assert(manipulator.fullData.length > 0, 'Data not loaded successfully ');

  const trainData = manipulator.fullData.slice(1, 101);
  const testData = manipulator.fullData.slice(101, manipulator.fullData.length);

  const forest = new RandomForestClassifier();
  console.assert(forest.forest.length === 0, 'invalid construction');
  await forest.build(trainData);
  console.assert(forest.forest.length > 0, 'invalid construction');

  const pred = forest.predict(testData);
  console.assert(pred > 0, 'invalid prediction');

  console.log('All Tests Passed');
}

test();
