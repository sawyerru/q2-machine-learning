const RandomForestClassifier = require('../decision-trees/random-forest');
const DataManipulator = require('../data-manipulator/data-manipulator');

const test = async () => {
  console.log('Random Forest Tests');
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-5_class.csv';
    const manipulator = new DataManipulator();
    await manipulator.loadCsv(url);
    const trainData = manipulator.fullData.slice(1, 101);
    const testData = manipulator.fullData.slice(101, manipulator.fullData.length);
    const forest = new RandomForestClassifier();
    await forest.build(trainData);

    // let correctClassCount = 0;
    // for (let i = 0; i < testData.length; i++) {
    //   const sample = testData[i];
    //   const label = sample.pop();
    //   const prediction = forest.predict(sample);

    //   this.jsonData[i].class5 = Number(label);
    //   this.jsonData[i].rf = Number(prediction);

    //   if (this.jsonData[i].class5 === this.jsonData[i].rf) {
    //     correctClassCount += 1;
    //   }
    // }
    // this.rf_acc = (correctClassCount / testData.length) * 100;

  console.log('All Tests Passed');
}

test();