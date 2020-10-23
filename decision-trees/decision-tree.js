/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 * Sources: https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
 */

const BaseModel = require('../base-model');
const {LeafNode, DecisionNode, Condition, countLabels} = require('./nodes');

/**
 * @public
 * Get the most frequent object 
 * @param {*} obj 
 * @param {*} fromArray 
 */
function getMostFrequent(obj, fromArray=false) {
    let table = obj;
    if (fromArray) {
        table = {}
        for (let i = 0; i < obj.length; i++) {
            if (obj[i] in table) {
                table[obj[i]] += 1;
            } else {
                table[obj[i]] = 1;
            }
        }
    }
    let max = 0;
    let assignment = 0;
    Object.keys(table).forEach( (ele,i) => {
        if (table[ele] > max) {
            max = table[ele];
            assignment = i
        }
    });
    return Object.keys(table)[assignment];
}

/**
 * Generate random array
 * @param {Number} size 
 * @param {Number} dims 
 */
function randomArray(size, dims) {
    const random = [];
    for (let i = 0; i < size; i++) {
        let idx = Math.floor(Math.random() * dims);
        while (random.indexOf(idx) !== -1) {
            idx = Math.floor(Math.random() * dims);
        }
        random[i] = idx;
    }
    return random;
}

/**
 * @private
 * Create and return a unique set of values from a feature matrix 
 * @param {Array} data matrix to be processed 
 * @param {Number} col the column number of the feature to gather unique set
 * @returns {Set} set of unique responses for single feature
 */
function createSet(data, col) {
    const set = new Set(); // Create set of responses from each sample
    data.forEach( row => {set.add(row[col]); } );
    return set
}

/**
 * @private
 * Calculate the Gini Impurity based on formula from:
 * https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
 * @param {Array} data matrix to calculate gini for each branch
 * @returns {Number} gini impurity
 */
function calculateGini(data) {
    let sumOfLabelProbabilities = 0;

    // Get counts of samples (histogram)
    const labelCounts = countLabels(data);

    for (let label in labelCounts) {
        const probabilityOfLabel = labelCounts[label] / data.length;
        sumOfLabelProbabilities += Math.pow(probabilityOfLabel, 2);
    }
    return 1 - sumOfLabelProbabilities;
}

/**
 * Calculate the information gain from one layer in the tree to the next - helps quantify decisions on nodes. 
 * Desc: https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain
 * @param {Array} trueSet array of sample that are true from condition
 * @param {Array} falseSet array of sample that are false from condition
 * @param {Number} currentUncertainty current tree gini uncertainty
 * @returns {Number} information gain
 */
function calculateInfoGain(trueSet, falseSet, currentUncertainty) {
    const pTrue = trueSet.length / (trueSet.length + falseSet.length);
    return currentUncertainty - (pTrue * calculateGini(trueSet) + (1-pTrue) * calculateGini(falseSet));
}

/**
 * Split the data based on the condition into true and false sets
 * @param {Array} data full data matrix to be split
 * @param {Condition Instance} condition condition Object to split on
 * @returns {Array} [true Set, false Set]
 */
function partition(data, condition) {
    const trueSet = [];
    const falseSet = [];
    
    // Compare data
    data.forEach( row => {
        const result = condition.compareTo(row);
        if (result) {
            trueSet.push(row);
        } else {
            falseSet.push(row)
        }
    });

    return [trueSet, falseSet];
}

/**
 * make a decision on which node to place in the tree - calculating and comparing all conditions and information gains before making decision
 * @param {Array} data matrix to be used in decision
 * @param {Bool} random_subspace indicator to use all of the features (normal decision Tree = false) or to generate a random tree (=true)
 * @returns {Array} [bestGain, bestConditionObject]
 */
function calculateDecision(data, random_subspace) {
    let bestGain = -1;
    let bestCondition = undefined;
    const uncertainty = calculateGini(data);
    let dims = data[0].length - 1;
    const subset = randomArray(2, dims);
    dims = (random_subspace) ? subset.length : dims; // removing label

    for (let col = 0; col < dims; col++) { // loop through all features
        if (random_subspace) {
            col = subset[col];
        }
        const uniqueVals = createSet(data, col);
        for (let val of uniqueVals) {
            const cond = new Condition(col, val);
            
            // partition based on condition into true and false
            const [trueSet, falseSet] = partition(data, cond);

            // recalculate info gain and update bests if neccessary
            const gain = calculateInfoGain(trueSet, falseSet, uncertainty);

            if (gain > bestGain) {
                bestGain = gain;
                bestCondition = cond;
            }
        }
    }
    
    return [bestGain, bestCondition];
}

/**
 * Decision Tree built using CART (Classification and regression trees) algorithm
 */
class DecisionTreeClassifier extends BaseModel{
    constructor() {
        super();
        this.data;
        this.root;
    }

    /**
     * Construct the tree recursively using a CART algorithm.
     * @param {*} data Matrix to build tree on
     * @param {bool} hasHeaders bool to indicate if data has headers or not
     * @returns {Node Instance} root node
     */
    construct(data, hasHeaders=true, random_subspace=false) {
        if (hasHeaders) {
            this.headers = data.shift()
            this.data = data;
        }

        // Recursive wrapper to encapsulate method
        let constructHelper = (data, random_subspace) => {
            // Find best question and split by iterating over feature/value and calculating info gain
            const [infoGain, condition] = calculateDecision(data, random_subspace);
    
            // Base case:
            if (infoGain === 0) {
                return new LeafNode(data);
            }
            
            // Partition data into false and true branches
            const [trueData, falseData] = partition(data, condition);
            
            const trueBranch = constructHelper(trueData, random_subspace);
            const falseBranch = constructHelper(falseData, random_subspace);
    
            return new DecisionNode(condition, trueBranch, falseBranch); 
        }
        this.root = constructHelper(data, random_subspace);
        this._model = this.root;
    }

    /**
     * Predict a single example
     * @param {Array} X sample to be predicted, passing no labels
     */
    predict(X) {
        // Recursive wrapper to encapsulate method
        let predictHelper = (X, node) => {
            if (node.isLeaf) {
                return node.classCount;
            }

            if (node.condition.compareTo(X)){
                return predictHelper(X, node.trueBranch);
            } else {
                return predictHelper(X, node.falseBranch);
            }
        }

        const obj = predictHelper(X, this.root);
        return getMostFrequent(obj, false);
    }
       

    /**
     * Recursive Printing of the decision tree created in console
     * @param {Node Instance} tree parent node of tree/sub-tree
     * @param {string} spacing string used in formatting
     */
    display() {
        // Recursive wrapper to encapsulate method
        let displayHelper = (node, spacing='') => {
            if (node.isLeaf) {
                console.log(`${spacing} Predict`, node.classCount);
                return;
            }
            console.log(`${spacing} ${node.condition.print(this.headers)}`);
    
            console.log(`${spacing} --> True:`)
            displayHelper(node.trueBranch, spacing + ' ')
            console.log(`${spacing} --> False:`)
            displayHelper(node.falseBranch, spacing + ' ')
        }
        return displayHelper(this.root);
    }
   
}

module.exports = DecisionTreeClassifier;
