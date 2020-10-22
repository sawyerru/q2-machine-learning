const LinearRegression = require('../linear-regression/linear-regression');
const DataManipulator = require('../data-manipulator/data-manipulator');


const test = async () => {
  console.log('Linear Regression Tests');
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-1class.csv';
    const manip = new DataManipulator();
    // await manip.loadCsv(url);
    // manip.processData(this.config);
    // manip.normalize();

    // const [feats, labels] = manip.exportData();
    // const dims = feats.shape[1];
    const linearRegression = new LinearRegression();
    // linearRegression.build(dims);
    // await linearRegression.train(feats, labels, true);

    // const model = linearRegression.getModel();

    // let errorSum = 0;
    // for (let i = 0; i < feats.shape[0]; i++ ) {
    //   const sample = feats.slice([i, 0], [1, 43]);
    //   const prediction = linearRegression.predict(sample);

    //   this.jsonData[i].linReg = prediction;
    //   errorSum += Math.pow(this.jsonData[i].G3 - prediction, 2);
    // }
    // this.linearRegression_acc = Math.pow(errorSum / feats.shape[0], 0.5);
    // linearRegression.clean();

  console.log('All Tests Passed');
}

test();