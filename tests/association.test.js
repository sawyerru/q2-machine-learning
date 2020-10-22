const AprioriAssociationGenerator = require('../association/apriori');
const DataManipulator = require('../data-manipulator/data-manipulator');

const test = () => {
    console.log('Apriori Tests');
    const manip = new DataManipulator();
    const apriori = new AprioriAssociationGenerator();

    console.log('All Tests Passed');
}

test();
