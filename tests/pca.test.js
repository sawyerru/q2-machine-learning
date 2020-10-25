const PCASelector = require('../pca/pca');
const DataManipulator = require('../data-manipulator/data-manipulator');

const got = require('got');

const test = async () => {
    console.log('PCA Tests');
    const manip = new DataManipulator();
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/wine.csv';
    const resp = await got(url);

    await manip.loadCsv(resp.body);

    const pca = new PCASelector();
    console.assert(pca.principleColumns.length === 0 && pca.principleComponents.length === 0,
        'Invalid principle sizes');
    const data = manip.exportData(false);

    const [principleComps, principleCols] = pca.select(data);
    console.assert(principleComps.length > 0 && principleCols.length > 0,
        'No princple components selected');

    console.log('All Tests Passed');
}

test();
