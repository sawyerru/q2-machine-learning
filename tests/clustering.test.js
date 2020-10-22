const ClusterClassifier = require('../clustering/clustering');
const DataManipulator = require('../data-manipulator/data-manipulator');


const test = async () => {
    console.log('Clustering Tests')
    const manip = new DataManipulator();
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/wine.csv';
    await manip.loadCsv(url);
    const clust = new ClusterClassifier();
    // clust.train(manip.exportData(false), 3);
    // const pred = clust.predict(manip.fullData[0]);
    clust.clean();

    console.log('Tests Passed');
}

test();