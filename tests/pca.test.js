const PCASelector = require('../pca/pca');
const DataManipulator = require('../data-manipulator/data-manipulator');


const test = async () => {
    console.log('PCA Tests');
    const manip = new DataManipulator();
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/wine.csv';
    await manip.loadCsv(url);

    const pca = new PCASelector();
    const data = manip.exportData(false);
    const [principleComps, principleCols] = pca.select(data);
    console.log('All Tests Passed');
}

test();