/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 * Content: DataManipulator Class to handle csv data and help with manipulation of types, categorical variables, etc.
 */

const tf = require('@tensorflow/tfjs');

/**
 * @private
 * Swap 2 elements in an array
 * @param {Array} array array object for things to be swapped within
 * @param {Number} idx1 element to be moved
 * @param {Number} idx2 location to be placed (current element in this spot gets placed )
 */
function swap(arr, idx1, idx2) {
    const temp = arr[idx1];
    arr[idx1] = arr[idx2];
    arr[idx2] = temp;
}

/**
 * @private
 * Helper function to convert categorical variables to one-hot array
 * @param {Number} value which class
 * @param {Number} num_of_classes total number of classes
 * @returns {Array} one-hot array containing where 1 is positive example and 0 is negative
 */
function convertCategorical(value, num_of_classes) {
    const onehot = new Array(num_of_classes).fill(0);
    onehot[value] = 1;
    return onehot;
}

/**
 * @private
 * Convert array to tensor - can handle 1d and 2d array likes 
 * @param {Array} M matrix/vector to be converted to Tensor object
 * @returns {Tensor} Tensor Object representing M
 */
function convertToTensor(M) {
    if (typeof(M[0]) === 'object') {
        return tf.tensor(M, [M.length, M[0].length]);
    } else {
        return tf.tensor(M, [M.length, 1]);
    }
}

/**
 * @private
 * Normalize variable using a few different methods - returns a normalized value
 * @param {Number} x value to be normalized
 * @param {string} method can be one of four strings: 'zscore' | 'scale' | 'log' | 'clip'
 * @param {Object} config object used to define normalization parameters (max/min or mean/std)
 * @returns {Number} normalized value
 */
function normalizePoint(x, method = 'zscore', config) {
    const {
        max = NaN,
        min = NaN,
        mean = NaN,
        std = NaN,
    }  = config;
    switch (method) {
        case 'zscore':
            if (isNaN(mean) || isNaN(std)) {
                throw Error('Mean and Standard Deviation not defined in config object')
            }
            return (x - mean) / std;
        case 'scale':
            if (isNaN(min) || isNaN(max)) {
                throw Error('Max and Min not defined in config object')
            }
            return (x - min) / (max - min);
        case 'log':
            return Math.log(x);
        case 'clip':
            if (isNaN(min) || isNaN(max)) {
                throw Error('Max and Min not defined in config object')
            }
            let xPrime = (x > max) ? max : x;
            xPrime = (xPrime < min) ? min : xPrime;
            return xPrime;
        case 'mean':
            if (isNaN(mean) || isNaN(std)) {
                throw Error('Mean and Standard Deviation not defined in config object')
            }
            return (x - mean)
        default:
            throw Error("Normalization method not defined please try: 'zscore' | 'scale' | 'log' | 'clip' ")
    }
}

/**
 * @public
 * Data manipulation class to handle data rerieval and parsing
 */
class DataManipulator {
    constructor() {
        // private
        this._dimensions = -1;
        this._samples = -1;
        this._config = undefined;
        this._numericalColumns = [];
        this._colHeaders = [];

        //public 
        this.fullData = [];
        this.X = [];
        this.Y = [];
        this.jsonData = [];
    }

    /**
     * @public
     * Getter to return size and shape of feature matrix
     * @returns {Array} [number of samples, number of features]
     */
    getFeatureSize() {
        return [this._samples, this._dimensions];
    }

    /**
     * @public
     * Getter for returning the column headers 
     * @returns {Array} column headers as strings
     */
    getColumnHeaders() {
        const cols = {}
        this._colHeaders.forEach( (ele, i) => { cols[i] = ele }); 
        return cols;
    }

    /**
     * @public
     * Take URL string/file as a csv and a config object to create a Feature Matrix, Label Matrix and JSON Object for items
     * @param {Array} data of raw data (rows should be comma delimited strings)
     * @returns nothing
     */
    async loadCsv(data) {

        const allLines = data.split('\n');
        allLines.forEach( (sample, idx) => {
            const row = sample.split(',').map( v => {
                return (isNaN(Number(v))) ? v.trim() : Number(v);
            });
            if (idx === 0) {
                this._colHeaders = row
                this._dimensions = this._colHeaders.length - 1;
            }
            if (row.length === this._colHeaders.length) {
                this.fullData.push(row);
            }
        });

        // Removing header row and label column
        this._samples = this.fullData.length - 1;
    }

    /**
     * @public
     * Process Data - converting categorical to one-hot and organizing objects 
     * @param {Object} config object that must define: labelHeading, labelType, excludeColumns, CategoricalCols. 
     * CategoricalCols should be an object with header: {option1: 0, option2: 1, option3: 2}
     * Even Binary variables (yes/no) should be defined, these are handled in the function 
     */
    processData(config) {       
        const {
            labelHeading = null,
            labelType = 'numeric',
            excludeColumns = [],
            categoricalCols = {}
        } = config;
        this._config = config;

        // Put all data into appropriate matricies/objects
        for (let i = 1; i < this.fullData.length; i++) {
            const sample = this.fullData[i]
            const obj = {};
            let ys = [];
            let xs = [];
            let s = 0;
            for (let j = 0; j < sample.length; j++) {

                const value = sample[j];
                const attribute = this.fullData[0][j];

                // Add label to ys 
                if (attribute === labelHeading) {
                    ys = (labelType === 'numeric') ? Number(value) : convertCategorical(categoricalCols[attribute][value], Object.keys(categoricalCols[attribute]).length);
                }
                // if Category - convert to number from obj and make oneHot put in placeholder for normalization
                else if (attribute in categoricalCols) {
                    const translatedValue = categoricalCols[attribute][value];
                    const numOptions = Object.keys(categoricalCols[attribute]).length
                    if (numOptions > 2) {
                        const onehot = convertCategorical(translatedValue, numOptions);
                        xs = [...xs, ...onehot];
                        onehot.forEach( () => {
                            this._numericalColumns[s] = 0;
                            s++;
                        })
                    } else {
                        xs.push(Number(translatedValue));
                        this._numericalColumns[s] = 0;
                        s++;
                    }
                } 
                // ignore exlusions
                else if (excludeColumns.includes(attribute)){
                    continue;
                } 
                // for numeric values add and update max/min/mean/std
                else {
                    xs.push(Number(value));
                    this._numericalColumns[s] = 1;
                    s++;

                }
                obj[attribute] = value;
        }
        this.X.push(xs);
        this.Y.push(ys);
        this.jsonData.push(obj);
        }
        this._dimensions = this.X[0].length; // update dimensions to support one-hot variables
    }

    /**
     * @public
     * Normalize all numeric columns saved from processing
     * @param {string} method normalization method can be: 'zscore' | 'scale' | 'log' | 'clip' -- Default is 'zscore'
     * @param {number} min (optional) use if you are using clipping normalization
     * @param {number} max (optional) use if you are using clipping normalization
     */
    normalize(method='zscore', min=NaN, max=NaN) {
        // tf.tidy( ()=> {
            const tensor = convertToTensor(this.X);
            const maxArray = tensor.max(0).dataSync();
            const minArray = tensor.min(0).dataSync();
            const meanArray = tensor.mean(0).dataSync();

            // Loop through all features to check normalize
            for(let i = 0; i < this._dimensions; i++) {
                if (this._numericalColumns[i] === 1) {

                    const config = {
                        max: (isNaN(max) && method !== 'clip') ? maxArray[i] : max,
                        min: (isNaN(min && method !== 'clip')) ? minArray[i] : min,
                        mean: meanArray[i]
                    }

                    if (method === 'zscore') {
                        let totalVariation = 0;
                        this.X.forEach( sample => {
                            totalVariation += Math.pow(sample[i] - meanArray[i], 2);
                        });
                        const variance = totalVariation / this._samples;
                        const std = Math.sqrt(variance);
                        config['std'] = std;
                    }

                    if (method === 'clip' && (isNaN(min) || isNaN(max)) ) {
                        throw Error('To use clip normalization you must define min and max values');
                    }
                    
                    this.X.forEach( sample => {
                        sample[i] = normalizePoint(sample[i], method, config)
                    })
                }
            }
        // });        
    }

    /**
     * @public
     * Export X and Y arrays as tensors returns [X, Y]
     * @returns {Array} [Feature Matrix, Label Matrix]
     */
    exportData(labels = true) {
        if (labels) {
            return [
                convertToTensor(this.X),
                convertToTensor(this.Y)
                ];
        } else {
            return convertToTensor(this.fullData);
        }
        
    }

    /**
     * @public
     * Shuffle all samples into different order 
     */
    shuffle() {
        const indicies = tf.util.createShuffledIndices(this._samples);
        indicies.forEach( (oldIdx, newIdx) => {
            swap(this.X, oldIdx, newIdx);
            swap(this.Y, oldIdx, newIdx);
            swap(this.jsonData, oldIdx, newIdx);
            swap(this.fullData, oldIdx, newIdx)
        });
    }

    /**
     * @public
     * Split Data into training and testing
     * @param {Array} threshold (optional) % of data in each set. if only 2 elements - no validation set will be created
     * @returns {Array} Array in shape of [Xtrain, Ytrain, Xtest, Ytest] or [Xtrain, Ytrain, Xvalidation, Yvalidation, Xtest, Ytest]
     * all objects are Tensor objects
     */
    split(thresholds=[0.6, 0.2, 0.2]) {
        // Error Checking
        const sumArray = (accumulator, currentValue) => accumulator + currentValue;
        const getValidation = (thresholds.length === 3);
        if (thresholds.reduce(sumArray) !== 1) {
            throw Error('Sum of passed %ages do not sum to 1!')
        } 
        if (thresholds.length > 3) {
            throw Error(`Too many elements in thresholds array. Max is 3 you passed ${thresholds.length}`);
        }
        if (thresholds.length < 1) {
            throw Error('Too few elements in thresholds array. Min is 1 you passed an empty array');
        }

        return tf.tidy( () => {
            const [X, Y] = this.exportData();
            const [n, d] = this.getFeatureSize();

            // Get Train and Test Always
            const numTrain = Math.round(this._samples * thresholds.shift());
            const Xtrain = X.slice([0, 0], [numTrain, d]);
            const Ytrain = Y.slice([0, 0], [numTrain, Y.shape[1]]);

            const numTest = Math.round(this._samples * thresholds.pop())
            const Xtest = X.slice([numTrain, 0], [numTest, d]);
            const Ytest = Y.slice([numTrain, 0], [numTest, Y.shape[1]]);

            // Conditionally get and return validation set
            if (getValidation) {
                const numValidation = Math.round(this._samples * thresholds[0]);
                const Xvalidation = X.slice([numTest, 0], [numValidation, d]);
                const Yvalidation = Y.slice([numTest, 0], [numValidation, Y.shape[1]]);

                if (numTrain + numTest + numValidation === n) {
                    return [Xtrain, Ytrain, Xvalidation, Yvalidation, Xtest, Ytest]
                } else {
                    throw Error('Missing some values')
                }
            }
            if (numTrain + numTest === n) {
                console.log(numTrain, numTest)
                return [Xtrain, Ytrain, Xtest, Ytest]
            } else {
                throw Error('Missing some values')
            }
        });        
    }

    /**
     * @public
     * concatenate X and Y matricies to get single matrix
     * @returns {Array} Full dataset
     */
    concatenateXY() {
        const fullData = [];
        for (let i = 0; i < this._samples; i++) { 
            let row = (Array.isArray(this.Y[i])) ? [...this.X[i], ...this.Y[i]] : [...this.X[i], this.Y[i]];
            fullData.push(row);
        }
        return fullData;
    }
}

module.exports = DataManipulator;
