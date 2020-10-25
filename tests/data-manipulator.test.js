/*
Data Manipulation Tests
Author: Sawyer Ruben
 */
const DataManipulator = require('../data-manipulator/data-manipulator');
const got = require('got')

const test = async () => {
    console.log('Data Manipulation Tests');
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-1class.csv';
    const res = await got(url);
    const manipulator = new DataManipulator();
    await manipulator.loadCsv(res.body);

    console.assert(manipulator.fullData.length > 0, 'Did not load data properly');
    console.assert(manipulator.getColumnHeaders() !== undefined, 'Column Headers not returned properly');
    console.assert(manipulator.getFeatureSize()[0] === 649, 'Samples not all gathered');
    console.assert(manipulator.getFeatureSize()[1] === 30, 'dimensons not all gathered');

    const config = {
        labelHeading: 'G3', // define which column is label
        labelType: 'numeric', // define what type of label (numeric or categorical)
        excludeColumns: [], // exclude columns (list of headings)
        categoricalCols: { // Categorical columns object (keys are column headings)
          school: {GP: 0, MS: 1}, // column should have all options for category (key is name of option, value is translated value)
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
    }
    manipulator.processData(config); // process data with config to do one-hot encoding, translation, and observation

    console.assert(manipulator.X.length > 0, 'Feature matrix not processed');
    console.assert(manipulator.Y.length > 0, 'Label matrix not processed');
    console.assert(manipulator.jsonData.length > 0, 'Json data not processed');

    const normalizationTechniques = ['zscore', 'clip'] // zscore is most common and default in normalize function
    manipulator.normalize(); // normalize data with techniques

    const preShuffle = manipulator.fullData[0];
    manipulator.shuffle(); // shuffle data
    const postShuffle = manipulator.fullData[0];

    console.assert(preShuffle !== postShuffle, 'Not Shuffled');

    const [features, labels] = manipulator.exportData(); // export data as tensor objects for next steps
    const [xTrain, yTrain, xTest, yTest] = manipulator.split(); // split data into train, test, and can pass a threshold to get validation

    console.log('All Tests Passed');
};

test();


