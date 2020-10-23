const ClusterClassifier = require('../clustering/clustering');
const DataManipulator = require('../data-manipulator/data-manipulator');

const got = require('got');

const test = async () => {
    console.log('Clustering Tests')
    const manip = new DataManipulator();
    const url = 'https://raw.githubusercontent.com/sawyerru/q2-data/master/wine.csv';
    const res = await got(url);
    await manip.loadCsv(res.body);
    console.assert(manip.fullData.length > 0, 'Did not load data properly');

    const clust = new ClusterClassifier();
    console.assert(clust.centroids.length === 0, 'Cluster Class not initialized');
    await clust.train(manip.exportData(false), 3);
    console.assert(clust.centroids.length === 3, 'Cluster Class not created 3 cetroids');

    const pred = clust.predict(manip.fullData[0]);
    console.assert(pred > 3 || pred < 0, 'Prediction not within clasification range')

    clust.clean();

    console.log('Tests Passed');
}

test();
