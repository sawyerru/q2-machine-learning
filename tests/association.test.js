/*
Apriori Association Tests
Author: Sawyer Ruben
 */

const AprioriAssociationGenerator = require('../association/apriori');
const DataManipulator = require('../data-manipulator/data-manipulator');

const test = () => {
    console.log('Apriori Tests');
    const manip = new DataManipulator();
    const apriori = new AprioriAssociationGenerator();
    /*
    Implement test cases
     */

    console.log('All Tests Passed');
}

test();
