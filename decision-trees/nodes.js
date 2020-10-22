/**
 * Author: Sawyer Ruben
 * Date: August 2020
 * License: MIT
 */

 /**
  * @private 
  * Helper function to create histogram of labels in data
  * Labels MUST be in final column of Matrix
  * @param {Array} data Matrix to be counted 
  * @returns {Object} {label: count, ... , ...}
  */
function countLabels(data) {
    const labelCounts = {};

    for (let i = 0; i < data.length; i++) {
        const sample = data[i];
        const label = sample[sample.length - 1];
        if (!(label in labelCounts)) {
            labelCounts[label] = 0;
        }
        labelCounts[label] += 1;
    }
    return labelCounts;
}

/**
 * @private 
 * Helper Class to hold condition and assist with comparing and printing
 */
class Condition {
    constructor(col, value) {
        this._col = col;
        this._value = value;
        this._isNumeric = (!isNaN(Number(value))) ? true : false;
    }

    /**
     * @private
     * Compare a single sample's value to the condition and return bool
     * @param {Array} sample array of values
     * @returns {Bool} meets or fails condition 
     */
    compareTo(sample) {
        const compValue = sample[this._col];
        if (typeof(compValue) === Number) {
            return compValue >= this._value;
        } else {
            return compValue == this._value;
        }
    }

    /**
     * @private
     * Print the condition into proper format
     * @param {Array} headers headers array stored in the Decision Tree class on construction
     * @returns {string} to be output
     */
    print(headers) {
        const cond = (this._isNumeric) ? '>=' : '==';
        return `Is ${headers[this._col]} ${cond} ${this._value}`;
    }
}


/**
 * @private
 * Helper class to represent Leaf Node in tree (no children)
 */
class LeafNode {
    constructor(data){
        this.isLeaf = true;
        this.classCount = countLabels(data);
    } 
    print() {
        return this.classCount.toString();
    }
}

/**
 * @private
 * Helper class to represent Decision Node in Tree (has children and splits data)
 */
class DecisionNode {
    constructor(condition, trueBranch, falseBranch) {
        this.isLeaf = false;
        this.condition = condition;
        this.trueBranch = trueBranch;
        this.falseBranch = falseBranch;
    }
}

module.exports = {
    LeafNode, 
    DecisionNode, 
    Condition,
    countLabels
}