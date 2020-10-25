/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 */

const BaseModel = require('../base-model');
const dt = require('./decision-tree');

/**
 * 
 * @param {Array} data full data set to generate random bootstrapped data set from
 * @returns {Array} bootstrap set
 */
async function generateBootstrap(data) {
    const bootstrapData = [];
    const n = data.length;
    
    for (let i = 0; i < n; i++) {
        const randomIdx = Math.floor((Math.random() * n)); // Select random 0-n
        const sample = data.slice(randomIdx, randomIdx+1);
        bootstrapData.push(sample[0]);        
    }

    return bootstrapData;
}

/**
 * @public
 * Random Forest classifier using base decision tree models
 */
class RandomForestClassifier extends BaseModel {
    constructor() {
        super();
        this._treeDepth = 6;
        this._numEstimators = 10;
        this.forest = [];
    }

    /**
     * Build the Random Forest containing many randomly structured trees
     */
    async build(fullData) {
        for (let i = 0; i < this._numEstimators; i++) {
            // Create Bootstrapped Dataset
            const bootstrapData = await generateBootstrap(fullData);

            // Construct Tree using bootstrapped data and random subset of variables
            const tree = new dt.DecisionTreeClassifier();
            tree.construct(bootstrapData, false, true);

            // Add tree to list 
            this.forest[i] = tree;
        }
    }

    predict(X) {
        if (this.forest.length === 0) {
            throw Error('Forest not created');
        }
        let output = [];
        for (let i = 0; i < this.forest.length; i++) {
            output[i] = this.forest[i].predict(X);
        }

        return dt.getMostFrequent(output, true)
    }
}

module.exports = RandomForestClassifier;
