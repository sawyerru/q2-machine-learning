const dt = require('../decision-trees/decision-tree');
const DataManipulator = require('../data-manipulator/data-manipulator');

const got = require('got');

const test = async () => {
  console.log('Decision Tree Tests');
  const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-5_class.csv';
  const res = await got(url);
  const manip = new DataManipulator();

  await manip.loadCsv(res.body);
  console.assert(manip.fullData.length > 0, 'Data not loaded successfully ');

  const train = manip.fullData.slice(100, manip.fullData.length);
  const test = manip.fullData.slice(1, 100);

  const decisionTree = new dt.DecisionTreeClassifier();

  try {
    const model = decisionTree.getModel();
  }
  catch(err) {
    console.assert(err !== undefined, 'Error since model is not defined yet')
  }

  decisionTree.construct(train, true);

  const root = decisionTree.getModel()
  console.assert( root !== undefined, 'Model defined post training');

  // decisionTree.display();
  const pred = decisionTree.predict(test);
  console.assert(pred > 0, 'Prediction invalid');

  console.log('ALL Tests Passed');
}

test();
