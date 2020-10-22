const DataManipulator = require('./data-manipulator/data-manipulator');
const ClusterClassifier = require('./clustering/clustering');
const DecisionTreeClassifier = require('./decision-trees/decision-tree');
const RandomForestClassifier = require('./decision-trees/random-forest');
const LinearRegression = require('./linear-regression/linear-regression');
const NaiveBayesClassifier = require('./naive-bayes/naive-bayes');
const NeuralNetwork = require('./neural-network/neural-net');
const {runTrainAndValidation, runTest} = require('./neural-network/modelSelection');
const PCASelector = require('./pca/pca');
const AprioriAssociationGenerator = require('./association/apriori');
const SupportVectorMachine = require('./svm/svm');

module.exports = {
    DataManipulator,
    ClusterClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    LinearRegression,
    NaiveBayesClassifier,
    NeuralNetwork,
    runTrainAndValidation, runTest,
    PCASelector,
    AprioriAssociationGenerator,
    SupportVectorMachine
}