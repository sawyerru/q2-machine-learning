/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 * Sources:
 * https://medium.com/analytics-vidhya/use-naive-bayes-algorithm-for-categorical-and-numerical-data-classification-935d90ab273f
 */
const BaseModel = require('../base-model');


/**
 * @private 
 * Bayesian Equation to calculate probability - only use for categorical vars
 * @param {Number} pClass probability of the class
 * @param {Number} pCategory probability of the category
 * @param {Number} pConditional probability of P(category|class)
 * @returns {Number} Bayesian probability 
 */
function BayesianEquation(pClass, pCategory, pConditional) {
    return (pClass * pConditional) / pCategory;
}

/**
 * @private
 * Normal Distribution equation - only use for numerical vars
 * @param {Number} x value observation
 * @param {Number} std standard deviation of the feature
 * @param {Number} mu mean of the feature
 * @returns {Number} normal distribution value
 */
function NormalDist(x, mu, std) {
    const exp = -1 * ((x - mu)**2)/(2*std**2);
    const frac = 1/ (std * Math.sqrt(2 * Math.PI));
    return frac * Math.pow(Math.E, exp);
}


/**
 * Classifier based on Bayes Theorem:
 * P(A|B) = ( P(B|A) * P(A) ) / P(B)
 */
class NaiveBayesClassifier extends BaseModel {
    constructor() {
        super();
        this.headers = {}
        this._pClass = {} // P(class)
        this._pCategory = [] // P(feature)
        this._pConditional = [] // P(class|feature)
        this._pStatistics = []
    }

    /**
     * @public
     * Train the model by counting probability and calculating conditional probabilites to store
     * @param {Tensor} X feature Tensor object
     * @param {Tensor} Y label tensor object
     * @param {Array} headers array of headers
     * @param {Object} config config object to define numerical and categorical columns
     */
    train(X, Y, headers, config) {
        const {numericalCols, categoricalCols} = config;
        this.headers = config;
        const [n, dims] = X.shape;

        // Get Probability of labels
        const labels = Y.arraySync();

        for (let i = 0; i < n; i++) {
            const cl = labels[i].indexOf(1);
            if (cl in this._pClass) { this._pClass[cl] += 1/n; } 
            else { this._pClass[cl] = 1/n; }
        }

        const data = X.arraySync();
        // Construct probabilities
        for (let f = 0; f < dims; f++) {
            // Add new objects to category and conditionals - associated with features
            if (numericalCols.indexOf(headers[f]) > -1) { // col is numerical
                const means = {};
                const deviations = {};

                // get sums for mean
                for (let i = 0; i < n; i++ ) {
                    const label = labels[i].indexOf(1);
                    if (label in means) { means[label].push(data[i][f]); }
                    else { means[label] = [data[i][f]]; }
                }

                // adjust mean
                for (let prop in means) {
                    let sum = means[prop].reduce( (a, b) => { return a+b; });
                    means[prop] = sum / means[prop].length;
                }

                // get sums for std
                for (let i = 0; i < n; i++ ) {
                    const label = labels[i].indexOf(1);
                    if (label in deviations) { deviations[label].push((data[i][f] - means[label])**2); }
                    else { deviations[label] = [(data[i][f] - means[label])**2]; }
                }
                // adjust stds
                for (let prop in deviations) {
                    let sum = deviations[prop].reduce( (a, b) => { return a+b; });
                    deviations[prop] = Math.sqrt(sum / (deviations[prop].length - 1));
                }

                this._pCategory.push(-1);
                this._pConditional.push(-1);
                this._pStatistics.push(
                    {mean: means,
                    std: deviations}
                    
                )
            }

            if (categoricalCols.indexOf(headers[f]) > -1) { // col in categorical
                const featCount = {};
                const condObj = {};

                for (let i = 0; i < n; i++ ) {
                    const feat = data[i][f];
                    if (feat in featCount) { featCount[feat] += 1/n; }
                    else { featCount[feat] = 1/n; }
                }
                for (let i = 0; i < n; i++) {
                    const feat = data[i][f];
                    const label = labels[i].indexOf(1);

                    const catCount = featCount[feat] * n
                    const cond = String(feat) + '|' + String(label) 
                    if (cond in condObj) { 
                        condObj[cond] += 1/catCount; 
                    }
                    else { 
                        condObj[cond] = 1/catCount; 
                    }
                }
                this._pCategory.push(featCount);
                this._pConditional.push(condObj); 
                this._pStatistics.push(-1)               
            }
        } 
    }

    /**
     * @public
     * Prediction method to classify a sample
     * @param {Array} X features - must be passed as processed
     * @returns label output as processed
     */
    predict(X) {
        const numLabels = Object.keys(this._pClass).length;
        const predictionScores = [];
        for (let i = 0; i < numLabels; i++) {
            // calculate bayes probability for each label;
            let p = 1
            for (let f = 0; f < X.length; f++) {
                if (this._pCategory[f] !== -1 && this._pConditional !== -1 && this._pStatistics[f] === -1) { // Categorical Variable
                    const cond = String(X[f]) + '|' + String(i);
                    p *= this._pConditional[f][cond];
                }
                else if (this._pStatistics[f] !== -1) { // Numerical Variable
                    const ndist = NormalDist(X[f], this._pStatistics[f].mean[i], this._pStatistics[f].std[i]);
                    p *= ndist;

                }
                else {
                    throw new Error('Invalid feature')
                }
            }
            predictionScores.push(p);

        }

        return predictionScores.indexOf(Math.max(...predictionScores));
    }
}

module.exports = NaiveBayesClassifier;
