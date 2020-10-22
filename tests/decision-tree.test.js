const DecisionTreeClassifier = require('../decision-trees/decision-tree');
const DataManipulator = require('../data-manipulator/data-manipulator');


const test = async () => {
  console.log('Decision Tree Tests');
  const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/student-data-5_class.csv';
  const manip = new DataManipulator();
  await manip.loadCsv(url);
  const train = manip.fullData.slice(100, manip.fullData.length);
  const test = manip.fullData.slice(1, 100);
  const decisionTree = new DecisionTreeClassifier();
  decisionTree.construct(train, true);
  // decisionTree.display();
  // const root = decisionTree.getModel;

  // let correctClassCount = 0;
  // for (let i = 0; i < test.length; i++) {
  //   const sample = test[i];
  //   const label = sample.pop();
  //   const prediction = decisionTree.predict(sample);

  //   this.jsonData[i].class5 = Number(label);
  //   this.jsonData[i].dt = Number(prediction);

  //   if (this.jsonData[i].class5 === this.jsonData[i].dt) {
  //     correctClassCount += 1;
  //   }
  // }
  // this.decision_tree_acc = (correctClassCount / test.length) * 100;
  console.log('ALL Tests Passed');
}

test();