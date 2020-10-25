/*
Apriori Association Tests
Author: Sawyer Ruben
 */

const LinearRegression = require('../linear-regression/linear-regression');
const DataManipulator = require('../data-manipulator/data-manipulator');

const got = require('got');

const test = async () => {
  console.log('Linear Regression Tests');
  const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-1class.csv';
  const resp = await got(url);
  const manip = new DataManipulator();

  await manip.loadCsv(resp.body); // process csv into proper datatypes

  const config = { // define config about the data
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

  manip.processData(config); // process raw data into configured data
  manip.normalize();         // normalize points with default z-score method

  console.assert(manip.X[0].length > 0 && manip.X.length > 0, 'X not defined');
  console.assert(manip.Y.length > 0, 'Y not defined');
  console.assert(manip.X.length === manip.Y.length, 'data lost');

  const [feats, labels] = manip.exportData(); // export data as tensors
  console.assert(feats.shape[0] === labels.shape[0], 'X and Y have different number of samples');
  const dims = feats.shape[1]; // get number of dimensions

  // create linear regression model
  const linearRegression = new LinearRegression();

  try { linearRegression.getModel(); }
  catch(err) { console.assert(err !== undefined, 'Error not caught'); }

  linearRegression.build(dims); //build linear regression model based on number of input dimensions

  const model = linearRegression.getModel();
  console.assert(model !== undefined, 'model not created');

  console.assert(linearRegression.trainingLogs === undefined, 'training logs not null');
  await linearRegression.train(feats, labels); // Train linear regression model based on features and labels
  console.assert(linearRegression.trainingLogs !== undefined, 'training logs null');

  const [weight, bias] = linearRegression.getWeightsAndBias();
  console.assert(weight.length !== 0 && bias.length !== 0, 'no weight or bias vectors');

  // *** Clean model to remove unneeded tensors and lighten browser load in memory ***
  linearRegression.clean();

  console.log('All Tests Passed');
}

test();
