const SupportVectorMachine = require('../svm/svm');
const DataManipulator = require('../data-manipulator/data-manipulator');

const test = async () => {
    console.log('SVM Tests');
    const manip = new DataManipulator();
    const svm = new SupportVectorMachine();

}

test();