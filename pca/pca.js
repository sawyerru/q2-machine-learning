/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 */
const tf = require('@tensorflow/tfjs');
const math = require('mathjs');
const BaseModel = require('../base-model');


/**
 * @private
 * Create the covariance matrix 
 * @param {Tensor} dataSet Tensor of data
 * @returns covar matrix (sigma)
 */
function createCovarianceMatrix(dataSet) {
    const [n, dimensions] = dataSet.shape;
    let xCentered = null;
    // get mean for each feature
    const means = tf.mean(dataSet, 0).dataSync();
    for (let d = 0; d < dimensions; d++) {
        const col = dataSet.slice([0,d], [n,1]);
        const centeredCol = tf.sub(col, tf.scalar(means[d]))
        if (xCentered === null) {
            xCentered = centeredCol;
        } else {
            xCentered = tf.concat([xCentered, centeredCol], 1);
        }
    }
    const sigma = tf.dot(xCentered.transpose(), xCentered).div(tf.scalar(n));
    return sigma;
}

/**
 * @private
 * use mathjs lib to compute eigen values and vectors
 * @param {Array} matrix
 * @returns eigen values and eigen vectors
 */
function computeEigen(matrix) {
    matrix = matrix.arraySync();
    const ans = math.eigs(matrix);
    return [ans.values, ans.vectors]

}

/**
 * @private
 * Select the principle components based on largest Eigen values
 * @param {Array} eigVals 
 * @param {Array} eigVects 
 * @param {Number} threshold parameter to answer how tight or loss we want the components to represent the data
 * @returns [principle components, principle Columns]
 */
function eigenSelection(eigVals, eigVects, threshold) {
    const eigsum = tf.sum(tf.tensor(eigVals)).dataSync()[0];
    const hashTable = {}

    for (let i = 0; i < eigVals.length; i++) { 
        hashTable[eigVals[i]] = [i, eigVects[i]]; 
    }

    let sortedEigs = math.sort(eigVals, 'desc');
    const principleComponents = []
    const principleColumns = []
    let i = 0;
    let acc = 0;
    while (acc/eigsum < threshold) {
        acc += sortedEigs[i];
        principleColumns.push(hashTable[sortedEigs[i]][0])
        principleComponents.push(hashTable[sortedEigs[i]][1])
        i++;
    }
    return [principleComponents, principleColumns]

}

/**
 * @private
 * Get original data values from principle Columns
 * @param {Tensor} data original data
 * @param {Array} cols columns
 * @returns matrix of original data with only principle features
 */
function getPrincipleColumns(data, cols) {
    const [n, dims] = data.shape;
    const princCols = []
    for (let c = 0; c < dims; c++) {
        if (cols.indexOf(c) > -1) {
            const col = data.slice([0,c], [n,1]).arraySync();
            princCols.push(col);
        }
    }

    return princCols;

}

/**
 * @public 
 * Principle Component Analysis Selector
 */
class PCASelector extends BaseModel {
    constructor() {
        super();
        this.principleColumns = [];
        this.principleComponents = [];

    }

    /**
     * @public
     * Select principle components from data
     * @param {Tensor} data Data tensor with all original data
     * @param {Number} threshold number to represent % of variance we want to explain
     * @returns [matrix of original data with only principle features, list of principle columns]
     */
    select(data, threshold=0.95) {
       const covar = createCovarianceMatrix(data);
       const [eigVals, eigVects] = computeEigen(covar);
       const [principleComponents, principleColumns] = eigenSelection(eigVals, eigVects, threshold);
       this.principleColumns = principleColumns;
       this.principleComponents = getPrincipleColumns(data, this.principleColumns);
       return [this.principleComponents, this.principleColumns];
       
    }
}

module.exports = PCASelector;